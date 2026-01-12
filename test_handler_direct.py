"""
Direct test of the chat_response handler without Gradio/WebRTC
Simulates what ReplyOnPause should do
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch
import asyncio
from pathlib import Path

# Import everything from gradio_official_pattern
from liquid_audio import ChatState, LFMModality, LFM2AudioModel, LFM2AudioProcessor

print("🧪 Direct Handler Test\n")

# Load models
print("1️⃣  Loading models...")
MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"   Using device: {device}")

processor = LFM2AudioProcessor.from_pretrained(MODEL_ID, device="cpu").eval()
model = LFM2AudioModel.from_pretrained(MODEL_ID, device="cpu").eval()

# CRITICAL: Extract the Mimi audio codec
mimi = processor.mimi.eval()

if device != "cpu":
    processor = processor.to(device)
    model = model.to(device)
    mimi = mimi.to(device)

print("   ✅ Models loaded\n")

# Generate test audio
print("2️⃣  Generating test audio...")
SAMPLE_RATE = 24000
duration_sec = 1
num_samples = int(SAMPLE_RATE * duration_sec)
t = np.arange(num_samples) / SAMPLE_RATE
test_audio = 0.3 * np.sin(2 * np.pi * 500 * t).astype(np.float32)
print(f"   ✅ Generated 1s test audio\n")

# Create chat state
print("3️⃣  Creating chat state...")
chat = ChatState(processor)
print("   ✅ Chat state created\n")

# Initialize chat
print("4️⃣  Initializing chat...")
chat.new_turn("system")
chat.add_text("Respond with interleaved text and audio.")
chat.end_turn()

chat.new_turn("user")
wav_float = torch.tensor(test_audio / 32_768.0, dtype=torch.float32)
# Add batch dimension
wav_float = wav_float.unsqueeze(0)  # [1, num_samples]
chat.add_audio(wav_float, SAMPLE_RATE)
chat.end_turn()

chat.new_turn("assistant")
print("   ✅ Chat initialized\n")

# Test generation
print("5️⃣  Testing interleaved generation...")
text_out = []
audio_chunks = []
chunk_count = 0

with torch.no_grad():
    with mimi.streaming(1):
        for t in model.generate_interleaved(
            **chat,
            max_new_tokens=128,  # Shorter for testing
            audio_temperature=1.0,
            audio_top_k=4,
        ):
            # Text token
            if t.numel() == 1:
                text_out.append(t)
                char = processor.text.decode(t)
                print(char, end="", flush=True)

            # Audio token (8 codebooks)
            elif t.numel() == 8:
                audio_chunks.append(t)

                # Decode incrementally
                if (t == 2048).any():
                    print("(pad)", end="", flush=True)
                    continue

                wav_chunk = mimi.decode(t[None, :, None])[0]
                print(f"🔊", end="", flush=True)
                chunk_count += 1

print("\n\n6️⃣  Results:")
print(f"   ✅ Generated {len(text_out)} text tokens")
print(f"   ✅ Generated {len(audio_chunks)} audio tokens")
print(f"   ✅ Decoded {chunk_count} audio chunks")

if len(audio_chunks) > 0:
    print(f"   ✅ SUCCESS - Handler produces audio!")
else:
    print(f"   ❌ FAILURE - No audio tokens generated!")

print("\n" + "="*50)
