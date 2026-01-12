"""Test just the model loading to find the hangpoint."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
from liquid_audio import ChatState, LFMModality, LFM2AudioModel, LFM2AudioProcessor

print("1️⃣  Loading processor...")
MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

processor = LFM2AudioProcessor.from_pretrained(MODEL_ID, device="cpu").eval()
print("✅ Processor loaded")

print("2️⃣  Loading model...")
model = LFM2AudioModel.from_pretrained(MODEL_ID, device="cpu").eval()
print("✅ Model loaded")

print("3️⃣  Extracting mimi codec...")
mimi = processor.mimi.eval()
print("✅ Mimi extracted")

print("4️⃣  Moving to device...")
if device != "cpu":
    print(f"   Moving processor to {device}...")
    processor = processor.to(device)
    print(f"   ✅ Processor on {device}")

    print(f"   Moving model to {device}...")
    model = model.to(device)
    print(f"   ✅ Model on {device}")

    print(f"   Moving mimi to {device}...")
    mimi = mimi.to(device)
    print(f"   ✅ Mimi on {device}")
else:
    print("   Staying on CPU")

print("\n✅ All models loaded successfully!")
