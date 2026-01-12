"""
LFM2.5-Audio with Gradio - EXACT official pattern
Based on: https://github.com/Liquid4All/liquid-audio/blob/main/src/liquid_audio/demo/chat.py
"""
from queue import Queue
from threading import Thread
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr
import numpy as np
import torch
from fastrtc import AdditionalOutputs, ReplyOnPause, WebRTC

from liquid_audio import ChatState, LFMModality, LFM2AudioModel, LFM2AudioProcessor

def log(msg):
    """Log to both stdout and file."""
    print(msg, flush=True)
    with open("/tmp/gradio_init.log", "a") as f:
        f.write(msg + "\n")
        f.flush()

log("Loading LFM2.5-Audio...")
MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"

device = "mps" if torch.backends.mps.is_available() else "cpu"
log(f"Device: {device}")

log("Loading processor...")
processor = LFM2AudioProcessor.from_pretrained(MODEL_ID, device="cpu").eval()
log("Loading model...")
model = LFM2AudioModel.from_pretrained(MODEL_ID, device="cpu").eval()

log("Extracting mimi codec...")
# CRITICAL: Extract the Mimi audio codec from processor
mimi = processor.mimi.eval()

log("Moving to device...")
if device != "cpu":
    processor = processor.to(device)
    model = model.to(device)
    mimi = mimi.to(device)  # Move audio codec to device too

log("✅ Ready\n")


def chat_producer(q: Queue, chat: ChatState):
    """Producer thread - generates tokens"""
    print("  [producer] Starting generation")
    with torch.no_grad():
        with mimi.streaming(1):
            for t in model.generate_interleaved(
                **chat,
                max_new_tokens=512,
                audio_temperature=1.0,
                audio_top_k=4,
            ):
                q.put(t)

                # Decode audio tokens
                if t.numel() > 1:  # Audio token
                    if (t == 2048).any():  # Skip padding
                        continue
                    wav_chunk = mimi.decode(t[None, :, None])[0]
                    q.put(wav_chunk)

    q.put(None)  # Signal end


def chat_response(audio: tuple[int, np.ndarray], _id: str, chat: ChatState):
    """Main response handler - exact official pattern"""
    # Log to file since Gradio might swallow stdout
    with open("/tmp/gradio_handler.log", "a") as f:
        f.write(f"\n📥 chat_response called at {__import__('time').time()}\n")
        if audio is None:
            f.write("   audio is None\n")
            return
        if audio[1] is None:
            f.write("   audio[1] is None\n")
            return
        f.write(f"   samples: {len(audio[1])}\n")
        f.flush()

    print(f"\n📥 chat_response called with {len(audio[1]) if audio[1] is not None else 0} samples")

    sample_rate, wav = audio

    # Initialize chat on first turn
    if len(chat.text) == 1:
        chat.new_turn("system")
        chat.add_text("Respond with interleaved text and audio.")
        chat.end_turn()
        chat.new_turn("user")

    # Add user audio - convert to float and add batch dimension
    wav_float = torch.tensor(wav / 32_768.0, dtype=torch.float32)
    if wav_float.dim() == 1:
        wav_float = wav_float.unsqueeze(0)  # Add batch dimension [1, num_samples]
    chat.add_audio(wav_float, sample_rate)
    chat.end_turn()

    # Assistant turn
    chat.new_turn("assistant")

    # Producer-consumer pattern
    q = Queue()
    producer = Thread(target=chat_producer, args=(q, chat))
    producer.start()

    out_text = []
    out_audio = []
    out_modality = []

    print("  🎯 Generating...")

    while True:
        t = q.get()
        if t is None:
            break

        elif t.numel() == 1:  # Text token
            out_text.append(t)
            out_modality.append(LFMModality.TEXT)
            char = processor.text.decode(t)
            print(char, end="", flush=True)
            cur_string = processor.text.decode(torch.cat(out_text)).removesuffix("<|text_end|>")
            yield AdditionalOutputs(cur_string)

        elif t.numel() == 8:  # Audio token
            out_audio.append(t)
            out_modality.append(LFMModality.AUDIO_OUT)

        elif t.numel() == 1920:  # Audio chunk (24kHz, 80ms)
            np_chunk = (t.cpu().numpy() * 32_767).astype(np.int16)
            yield (24_000, np_chunk)
            print("🔊", end="", flush=True)

    print("\n  ✅ Done")

    # Append to chat history
    if out_text and out_audio:
        chat.append(
            text=torch.stack(out_text, 1),
            audio_out=torch.stack(out_audio, 1),
            modality_flag=torch.tensor(out_modality),
        )

    chat.end_turn()


def clear_chat():
    gr.Info("Cleared", duration=1)
    return ChatState(processor), None


# Build UI - EXACT official pattern
log("Creating gr.Blocks()...")
with gr.Blocks() as demo:
    log("Creating Markdown...")
    gr.Markdown("# LFM2.5-Audio Chat\n\nOfficial pattern with MPS acceleration")

    log("Creating chat_state...")
    chat_state = gr.State(ChatState(processor))
    log("Creating WebRTC component...")
    webrtc = WebRTC(modality="audio", mode="send-receive")
    log("Creating text output...")
    text_output = gr.Textbox(lines=3, label="Response", interactive=False)
    log("Creating clear button...")
    clear_btn = gr.Button("Clear")

    # Wire up - EXACT pattern from official repo
    def log_stream_call(*args, **kwargs):
        print(f"🔌 stream() called with args={len(args)}, kwargs={kwargs}")
        import traceback
        traceback.print_stack(limit=5)

    print(f"📌 Setting up webrtc.stream()...")
    print(f"   webrtc type: {type(webrtc)}")
    print(f"   chat_state type: {type(chat_state)}")

    webrtc.stream(
        ReplyOnPause(
            chat_response,
            input_sample_rate=24_000,
            output_sample_rate=24_000,
            can_interrupt=False,
        ),
        inputs=[webrtc, chat_state],
        outputs=[webrtc],
    )

    print(f"✅ webrtc.stream() configured")

    webrtc.on_additional_outputs(lambda s: s, outputs=[text_output])
    clear_btn.click(clear_chat, outputs=[chat_state, text_output])


if __name__ == "__main__":
    log("🚀 About to launch Gradio...")
    log("Calling demo.launch()...")
    demo.launch(share=False, show_error=True)
    log("✅ Gradio launched successfully!")
