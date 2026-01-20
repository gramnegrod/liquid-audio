#!/usr/bin/env python3
"""
Mac STS Client - True Push-to-Talk
Hold 'p' to record, release to send to Pi
"""

import sounddevice as sd
import numpy as np
import requests
import struct
import threading
import io
import sys

PI_HOST = "http://raspberrypilfm.local:5002"
SAMPLE_RATE = 16000
MIC_DEVICE = 1  # MacBook Pro Microphone

# Recording state
recording = False
audio_chunks = []

def create_wav(audio_data):
    """Create WAV from float32 audio"""
    int16 = (np.clip(audio_data, -1, 1) * 32767).astype(np.int16)
    size = len(int16) * 2
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + size, b'WAVE', b'fmt ', 16, 1, 1,
        SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16, b'data', size)
    return header + int16.tobytes()

def audio_callback(indata, frames, time, status):
    """Capture audio while recording"""
    if recording:
        audio_chunks.append(indata.copy())

def send_audio():
    """Send recorded audio to Pi"""
    if not audio_chunks:
        print("No audio recorded")
        return

    audio = np.concatenate(audio_chunks).flatten()
    duration = len(audio) / SAMPLE_RATE

    if duration < 0.3:
        print("Too short")
        return

    print(f"Sending {duration:.1f}s to Pi...")

    try:
        wav = create_wav(audio)
        files = {'audio': ('rec.wav', io.BytesIO(wav), 'audio/wav')}
        resp = requests.post(f"{PI_HOST}/speak", files=files, timeout=120)

        if resp.ok:
            r = resp.json()
            print(f"Response: {r.get('text', '?')}")
            print(f"({r.get('processing_time', 0):.1f}s processing)")
        else:
            print(f"Error: {resp.status_code}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    global recording, audio_chunks

    print("\n" + "=" * 40)
    print("  Push-to-Talk STS Client")
    print("  Hold [P] to talk, release to send")
    print("  Press [Q] to quit")
    print("=" * 40)

    # Check Pi connection
    try:
        r = requests.get(f"{PI_HOST}/health", timeout=3)
        if not r.ok:
            raise Exception()
        print("✓ Pi connected")
    except:
        print("✗ Cannot reach Pi")
        return

    # List mic
    devices = sd.query_devices()
    mic_name = devices[MIC_DEVICE]['name'] if MIC_DEVICE < len(devices) else "default"
    print(f"✓ Mic: {mic_name}")
    print("\nReady. Hold P to speak.\n")

    # Start audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
        device=MIC_DEVICE,
        callback=audio_callback
    )
    stream.start()

    try:
        from pynput import keyboard

        def on_press(key):
            global recording, audio_chunks
            try:
                if key.char == 'p' and not recording:
                    recording = True
                    audio_chunks = []
                    print("● Recording...", end='\r')
                elif key.char == 'q':
                    return False  # Stop listener
            except AttributeError:
                pass

        def on_release(key):
            global recording
            try:
                if key.char == 'p' and recording:
                    recording = False
                    print("○ Processing...  ")
                    send_audio()
                    print("\nHold P to speak.")
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    except ImportError:
        print("pynput not available, using fallback...")
        print("Type 'r' + Enter to record 5 seconds, 'q' to quit")
        while True:
            cmd = input("> ").strip().lower()
            if cmd == 'q':
                break
            elif cmd == 'r':
                print("Recording 5 seconds...")
                audio_chunks = []
                recording = True
                import time
                time.sleep(5)
                recording = False
                send_audio()
    finally:
        stream.stop()
        stream.close()

    print("Bye!")

if __name__ == '__main__':
    main()
