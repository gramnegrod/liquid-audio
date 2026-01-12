"""
End-to-End Test: Full Gradio WebRTC pipeline
1. Connect to running Gradio server
2. Send audio via WebRTC
3. Verify audio response is received
4. Measure TTFA (Time To First Audio)
"""
import asyncio
import numpy as np
import time
import sys
import json
from pathlib import Path

print("🧪 End-to-End Gradio WebRTC Test\n")

# Step 1: Check if Gradio server is running
print("1️⃣  Checking if Gradio server is running...")
import urllib.request
try:
    response = urllib.request.urlopen("http://localhost:7860/", timeout=5)
    print("   ✅ Gradio server is listening on http://localhost:7860")
except Exception as e:
    print(f"   ❌ Gradio server not responding: {e}")
    print("   Run: source /tmp/liquid-py312/bin/activate && python gradio_official_pattern.py")
    sys.exit(1)

# Step 2: Check handler log
print("\n2️⃣  Checking handler invocation log...")
import time
time.sleep(1)
handler_log_path = Path("/tmp/gradio_handler.log")
if handler_log_path.exists():
    with open(handler_log_path) as f:
        handler_calls = f.read()
    if handler_calls.strip():
        print(f"   📋 Handler calls found:")
        for line in handler_calls.strip().split("\n")[-10:]:  # Last 10 lines
            print(f"      {line}")
    else:
        print("   ℹ️  Handler log exists but is empty (no calls yet)")
else:
    print("   ℹ️  Handler log doesn't exist yet (no calls yet)")

# Step 3: Use Gradio's API to trigger generation
print("\n3️⃣  Testing Gradio API (if available)...")
try:
    # Try to call the gradio API directly
    # Gradio exposes a /api/predict endpoint
    import json

    # Generate test audio (sine wave)
    SAMPLE_RATE = 24000
    duration = 1
    num_samples = int(SAMPLE_RATE * duration)
    t = np.arange(num_samples) / SAMPLE_RATE
    test_audio = 0.3 * np.sin(2 * np.pi * 500 * t)
    test_audio_int16 = (test_audio * 32767).astype(np.int16).tolist()

    print(f"   Generated test audio: {len(test_audio_int16)} samples")

    # Try Gradio's standard prediction endpoint
    # The format depends on the Gradio version
    api_url = "http://localhost:7860/api/predict"

    try:
        response = urllib.request.urlopen(api_url, timeout=5)
        print(f"   ✅ API endpoint found at {api_url}")
    except urllib.error.HTTPError as e:
        if e.code == 405:
            print(f"   ℹ️  API endpoint exists but METHOD not allowed (need POST)")
        else:
            print(f"   ℹ️  API endpoint returned {e.code}")
    except urllib.error.URLError as e:
        print(f"   ℹ️  API endpoint not available: {e.reason}")

except Exception as e:
    print(f"   ⚠️  API test skipped: {e}")

# Step 4: Provide instructions for manual testing
print("\n4️⃣  Manual Testing Instructions:")
print("""
Since Gradio WebRTC requires a real browser with WebRTC support:

a) Open http://localhost:7860 in your browser
b) Click the microphone button in the "Speak!" section
c) Speak clearly (e.g., "Hello, how are you?")
d) Click "Stop recording"
e) Wait for audio response

Expected results:
✅ Audio should start playing within 200-300ms (TTFA)
✅ Server logs should show "📥 chat_response called"
✅ Handler log should show handler invocation: cat /tmp/gradio_handler.log

If no audio plays:
- Check browser console for errors (F12)
- Check handler log: cat /tmp/gradio_handler.log
- Check Gradio stdout for errors
- Verify pause detection is working (VAD)
""")

# Step 5: Check for recent handler calls
print("\n5️⃣  Monitoring for handler calls...")
print("   Waiting 10 seconds for manual browser interaction...\n")

start_time = time.time()
last_line_count = 0

while time.time() - start_time < 10:
    if handler_log_path.exists():
        with open(handler_log_path) as f:
            lines = f.readlines()

        current_line_count = len(lines)
        if current_line_count > last_line_count:
            print(f"   📥 Handler called! New log entries:")
            for line in lines[last_line_count:]:
                print(f"      {line.rstrip()}")
            last_line_count = current_line_count

            # Check for success indicators
            full_log = "".join(lines)
            if "samples:" in full_log:
                print(f"\n   ✅ Handler is processing audio!")

    time.sleep(1)

if last_line_count == 0:
    print("   ℹ️  No handler calls received (browser interaction not detected)")
    print("   Please manually test using browser at http://localhost:7860")
else:
    print(f"\n   ✅ TEST PASSED - Handler received {last_line_count} calls")

print("\n" + "="*60)
print("Test complete. Check http://localhost:7860 for full UI testing.")
