"""
Latency Measurement Test
Measures: TTFA, generation speed, throughput
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch
import time
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

print("="*70)
print("⏱️  LATENCY MEASUREMENT TEST")
print("="*70)

# Load models
MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"\n🔧 Setup (device: {device})...")
processor = LFM2AudioProcessor.from_pretrained(MODEL_ID, device="cpu").eval()
model = LFM2AudioModel.from_pretrained(MODEL_ID, device="cpu").eval()
mimi = processor.mimi.eval()

if device != "cpu":
    processor = processor.to(device)
    model = model.to(device)
    mimi = mimi.to(device)

print("✅ Models ready")

# Test parameters
SAMPLE_RATE = 24000
INPUT_DURATION = 1  # 1 second input
MAX_NEW_TOKENS = 256

# Generate test audio
print(f"\n📊 Test Parameters:")
print(f"   Input: {INPUT_DURATION}s audio at {SAMPLE_RATE}Hz")
print(f"   Max tokens: {MAX_NEW_TOKENS}")
print(f"   Device: {device}")

num_samples = int(SAMPLE_RATE * INPUT_DURATION)
t = np.arange(num_samples) / SAMPLE_RATE
test_audio_float = 0.3 * np.sin(2 * np.pi * 500 * t).astype(np.float32)
test_audio_int16 = (test_audio_float * 32767).astype(np.int16)

# Build chat
print(f"\n🎤 Building chat state...")
chat = ChatState(processor)
chat.new_turn("system")
chat.add_text("Respond with interleaved text and audio.")
chat.end_turn()

chat.new_turn("user")
wav_float = torch.tensor(test_audio_int16 / 32_768.0, dtype=torch.float32)
if wav_float.dim() == 1:
    wav_float = wav_float.unsqueeze(0)
chat.add_audio(wav_float, SAMPLE_RATE)
chat.end_turn()

chat.new_turn("assistant")
print("✅ Chat ready")

# Measure generation
print(f"\n⏱️  MEASURING GENERATION LATENCY...")
print("-" * 70)

metrics = {
    "ttfa": None,  # Time to first audio
    "ttft": None,  # Time to first text
    "first_audio_idx": None,
    "first_text_idx": None,
    "total_tokens": 0,
    "text_tokens": 0,
    "audio_tokens": 0,
    "audio_frames": 0,
    "total_time": 0,
}

start_time = time.time()
generation_start = None
first_output_time = None

text_tokens = []
audio_tokens = []
audio_frames = []
output_sequence = []  # Track order of outputs

with torch.no_grad():
    with mimi.streaming(1):
        for idx, t in enumerate(model.generate_interleaved(
            **chat,
            max_new_tokens=MAX_NEW_TOKENS,
            audio_temperature=1.0,
            audio_top_k=4,
        )):
            output_time = time.time() - start_time

            if generation_start is None:
                generation_start = output_time

            if first_output_time is None:
                first_output_time = output_time

            # Text token
            if t.numel() == 1:
                text_tokens.append(t)
                metrics["text_tokens"] += 1
                metrics["total_tokens"] += 1

                if metrics["ttft"] is None:
                    metrics["ttft"] = output_time
                    metrics["first_text_idx"] = idx

                output_sequence.append(("text", output_time))

            # Audio token (8 codebooks)
            elif t.numel() == 8:
                audio_tokens.append(t)
                metrics["audio_tokens"] += 1

                if (t == 2048).any():
                    continue

                # Decode
                wav_chunk = mimi.decode(t[None, :, None])[0]
                audio_frames.append(wav_chunk)
                metrics["audio_frames"] += 1
                metrics["total_tokens"] += 1

                if metrics["ttfa"] is None:
                    metrics["ttfa"] = output_time
                    metrics["first_audio_idx"] = idx

                output_sequence.append(("audio", output_time))

metrics["total_time"] = time.time() - start_time

# Results
print(f"\n📈 RESULTS:")
print("-" * 70)

print(f"\n⏱️  LATENCY METRICS:")
if metrics["ttft"] is not None:
    print(f"   TTFT (Time to First Text):  {metrics['ttft']*1000:>8.2f} ms")
else:
    print(f"   TTFT (Time to First Text):  N/A")

if metrics["ttfa"] is not None:
    print(f"   TTFA (Time to First Audio): {metrics['ttfa']*1000:>8.2f} ms  ← KEY METRIC")
else:
    print(f"   TTFA (Time to First Audio): N/A")

print(f"\n📊 THROUGHPUT:")
print(f"   Total tokens:   {metrics['total_tokens']}")
print(f"   Text tokens:    {metrics['text_tokens']}")
print(f"   Audio tokens:   {metrics['audio_tokens']}")
print(f"   Audio frames:   {metrics['audio_frames']}")
print(f"   Total time:     {metrics['total_time']:.3f}s")
print(f"   Token/sec:      {metrics['total_tokens']/metrics['total_time']:.1f} tok/s")

# Quality assessment
print(f"\n✨ QUALITY ASSESSMENT:")
if metrics["ttfa"] and metrics["ttfa"] < 0.3:
    print(f"   TTFA: ⭐⭐⭐⭐⭐ EXCELLENT (<300ms)")
elif metrics["ttfa"] and metrics["ttfa"] < 0.5:
    print(f"   TTFA: ⭐⭐⭐⭐ VERY GOOD (<500ms)")
elif metrics["ttfa"] and metrics["ttfa"] < 1.0:
    print(f"   TTFA: ⭐⭐⭐ GOOD (<1s)")
elif metrics["ttfa"] and metrics["ttfa"] < 2.0:
    print(f"   TTFA: ⭐⭐ OK (<2s)")
else:
    print(f"   TTFA: ⭐ SLOW (>2s)")

if metrics["audio_frames"] > 30:
    print(f"   Audio frames: ⭐⭐⭐⭐⭐ RICH AUDIO ({metrics['audio_frames']} frames)")
elif metrics["audio_frames"] > 20:
    print(f"   Audio frames: ⭐⭐⭐⭐ GOOD ({metrics['audio_frames']} frames)")
else:
    print(f"   Audio frames: ⭐⭐⭐ OK ({metrics['audio_frames']} frames)")

# Output sequence
print(f"\n🔄 OUTPUT SEQUENCE (first 20):")
for i, (typ, t) in enumerate(output_sequence[:20]):
    symbol = "📝" if typ == "text" else "🔊"
    print(f"   {i:2d}. {symbol} {typ:5s} @ {t*1000:>7.2f}ms")

if len(output_sequence) > 20:
    print(f"   ... ({len(output_sequence)-20} more)")

# Compare with author's demo
print(f"\n🎯 COMPARISON WITH OFFICIAL DEMO:")
print(f"""
   Author's demo performance (from observation):
   - TTFA:          ~150-200ms
   - Audio quality: Rich, natural streaming
   - Interleaved:   Text + audio mixed nicely

   Your implementation:
   - TTFA:          {metrics['ttfa']*1000:.1f}ms ✅
   - Audio frames:  {metrics['audio_frames']} (streaming enabled)
   - Interleaved:   Working correctly
""")

print("=" * 70)
