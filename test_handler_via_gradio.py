"""
Simulate what Gradio/ReplyOnPause should do:
1. Get audio from WebRTC
2. Call handler with audio data
3. Collect handler output
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

print("🧪 Simulating Gradio Handler Call\n")

# Setup (copied from gradio_official_pattern.py)
MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"
device = "mps" if torch.backends.mps.is_available() else "cpu"

processor = LFM2AudioProcessor.from_pretrained(MODEL_ID, device="cpu").eval()
model = LFM2AudioModel.from_pretrained(MODEL_ID, device="cpu").eval()
mimi = processor.mimi.eval()

if device != "cpu":
    processor = processor.to(device)
    model = model.to(device)
    mimi = mimi.to(device)

print(f"✅ Models loaded on {device}\n")

# Import the actual handler
print("Importing chat_response handler...")
import importlib.util
spec = importlib.util.spec_from_file_location("gradio_pattern", "/Users/rodneyfranklin/Development/personal/liquid-audio/gradio_official_pattern.py")
gradio_module = importlib.util.module_from_spec(spec)

# Make models available to the module
gradio_module.processor = processor
gradio_module.model = model
gradio_module.mimi = mimi
gradio_module.device = device

spec.loader.exec_module(gradio_module)
print("✅ Handler imported\n")

# Create test audio
print("1️⃣  Creating test audio...")
SAMPLE_RATE = 24000
duration_sec = 1
num_samples = int(SAMPLE_RATE * duration_sec)
t = np.arange(num_samples) / SAMPLE_RATE
test_audio_float = 0.3 * np.sin(2 * np.pi * 500 * t).astype(np.float32)
# Convert to int16 (what microphone input would be)
test_audio_int16 = (test_audio_float * 32767).astype(np.int16)
print(f"✅ Generated test audio ({len(test_audio_int16)} samples)\n")

# Prepare inputs like Gradio would
print("2️⃣  Calling handler like Gradio/ReplyOnPause would...")
chat_state = ChatState(processor)
audio_input = (SAMPLE_RATE, test_audio_int16)
handler_id = "test_session_123"

try:
    print("   Invoking: chat_response(audio, _id, chat)")
    outputs = []

    # Call the generator
    gen = gradio_module.chat_response(audio_input, handler_id, chat_state)

    output_count = 0
    text_frames = []
    audio_frames = []

    # Collect all outputs
    for output in gen:
        output_count += 1
        outputs.append(output)

        # Output could be:
        # - (sample_rate, audio_chunk) for WebRTC
        # - AdditionalOutputs(text) for display

        if isinstance(output, tuple) and len(output) == 2:
            if isinstance(output[0], int) and isinstance(output[1], np.ndarray):
                # Audio frame
                audio_frames.append(output)
                print(f"   🔊 Audio frame {len(audio_frames)}: {len(output[1])} samples")
        else:
            # Check if it's AdditionalOutputs
            print(f"   📤 Output {output_count}: {type(output).__name__}")
            text_frames.append(output)

    print(f"\n✅ Handler completed!")
    print(f"   Text outputs: {len(text_frames)}")
    print(f"   Audio frames: {len(audio_frames)}")
    print(f"   Total outputs: {output_count}")

    if len(audio_frames) > 0:
        total_audio_samples = sum(len(f[1]) for f in audio_frames)
        print(f"   Total audio samples: {total_audio_samples}")
        print(f"\n✅ TEST PASSED - Handler works!")
    else:
        print(f"\n❌ TEST FAILED - No audio frames returned!")

except Exception as e:
    print(f"   ❌ Handler failed with error:")
    import traceback
    traceback.print_exc()
    print(f"\n❌ TEST FAILED")
