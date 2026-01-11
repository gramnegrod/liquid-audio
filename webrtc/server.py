"""
LFM2.5-Audio WebRTC Server
Real-time speech-to-speech streaming via WebRTC + FastRTC

Target: <1.2s end-to-end latency
Architecture: WebRTC (UDP) → LFM2.5-Audio interleaved generation → Audio streaming
"""

import asyncio
import logging
import time
from typing import AsyncGenerator
import numpy as np

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastrtc import Stream, ReplyOnPause
import torch
import os

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model config
MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"
SAMPLE_RATE = 16000  # LFM2.5-Audio expects 16kHz

# Global model + processor (loaded once)
_model = None
_processor = None
_chat_state = None


def load_models():
    """Load LFM2.5-Audio model and processor (one-time)"""
    global _model, _processor

    if _model is not None:
        logger.info("Models already loaded")
        return

    logger.info(f"Loading {MODEL_ID}...")

    try:
        # Determine target device
        if torch.cuda.is_available():
            target_device = "cuda"
            logger.info("✅ Using CUDA GPU")
        elif torch.backends.mps.is_available():
            target_device = "mps"
            logger.info("✅ Using Apple Metal Performance Shaders (MPS)")
        else:
            target_device = "cpu"
            logger.info("⚠️  Using CPU (this will be slow)")

        # Load on CPU first (safetensors requirement), then move to target device
        logger.info("Loading processor on CPU...")
        _processor = LFM2AudioProcessor.from_pretrained(MODEL_ID, device="cpu").eval()

        logger.info("Loading model on CPU...")
        _model = LFM2AudioModel.from_pretrained(MODEL_ID, device="cpu").eval()

        # Move to target device if not CPU
        if target_device != "cpu":
            logger.info(f"Moving models to {target_device.upper()}...")
            _processor = _processor.to(target_device)
            _model = _model.to(target_device)

        logger.info("✅ Models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}", exc_info=True)
        raise


def initialize_chat():
    """Create new ChatState for conversation"""
    global _chat_state, _processor
    _chat_state = ChatState(_processor)
    logger.info("✅ Chat state initialized")


async def speech_to_speech_streaming(audio_data: bytes) -> AsyncGenerator[bytes, None]:
    """
    Process audio via LFM2.5-Audio interleaved generation
    Streams audio chunks as they're generated

    Args:
        audio_data: Raw audio bytes from WebRTC (16kHz PCM)

    Yields:
        Audio chunks in real-time (16-bit PCM)
    """
    global _model, _processor, _chat_state

    if _model is None or _processor is None:
        logger.error("Models not loaded")
        return

    try:
        # Parse incoming audio
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        logger.info(f"📥 Received {len(audio_array)} samples ({len(audio_array)/SAMPLE_RATE:.2f}s)")

        # Add to chat state
        start_time = time.perf_counter()
        _chat_state.add_audio(audio_array, SAMPLE_RATE)
        logger.info(f"⏱️ Audio preprocessing: {(time.perf_counter()-start_time)*1000:.1f}ms")

        # Generate response via interleaved generation
        # This yields text + audio tokens mixed, enabling streaming
        audio_buffer = []
        text_output = ""
        ttfa = None  # Time to first audio

        inference_start = time.perf_counter()

        with torch.no_grad():
            for token_idx, token in enumerate(_model.generate_interleaved(
                _chat_state,
                max_new_tokens=512,
                audio_temperature=1.0,
                audio_top_k=4
            )):
                # Check token type
                if token.numel() == 1:
                    # Text token
                    text_token = _processor.text.decode(token)
                    text_output += text_token
                    logger.info(f"📝 Text: {text_token}", end="")

                else:
                    # Audio token (8 codebook indices from Mimi)
                    audio_buffer.append(token)

                    # Every 12 audio tokens, decode and stream a chunk
                    # (12 audio tokens ≈ 48ms of audio at ~250 codes/sec)
                    if len(audio_buffer) >= 12:
                        # Decode audio chunk
                        chunk_tensor = torch.stack(audio_buffer, dim=1).unsqueeze(0)

                        with torch.no_grad():
                            wav_chunk = _processor.detokenize_audio(chunk_tensor)

                        # Convert to 16-bit PCM bytes for WebRTC
                        pcm_int16 = (wav_chunk.numpy() * 32767).astype(np.int16)
                        audio_bytes = pcm_int16.tobytes()

                        # Record time to first audio
                        if ttfa is None:
                            ttfa = time.perf_counter() - inference_start
                            logger.info(f"🎵 TTFA: {ttfa*1000:.1f}ms")

                        # Yield to browser immediately (enables streaming)
                        yield audio_bytes
                        logger.info(f"📤 Yielded {len(audio_bytes)} bytes")

                        audio_buffer = []

        # Flush remaining audio tokens
        if audio_buffer:
            chunk_tensor = torch.stack(audio_buffer, dim=1).unsqueeze(0)
            with torch.no_grad():
                wav_chunk = _processor.detokenize_audio(chunk_tensor)
            pcm_int16 = (wav_chunk.numpy() * 32767).astype(np.int16)
            audio_bytes = pcm_int16.tobytes()
            yield audio_bytes
            logger.info(f"📤 Flushed final {len(audio_bytes)} bytes")

        total_time = time.perf_counter() - inference_start
        logger.info(f"✅ Generation complete: {total_time:.2f}s, TTFA: {ttfa*1000:.1f}ms if applicable")
        logger.info(f"📄 Full response: {text_output}")

    except Exception as e:
        logger.error(f"❌ Error in speech_to_speech_streaming: {e}", exc_info=True)
        raise


# Create FastAPI app
app = FastAPI()

# Create FastRTC Stream with LFM2.5-Audio handler
# ReplyOnPause means: listen for speech, detect silence, then respond
stream = Stream(
    ReplyOnPause(speech_to_speech_streaming),
    modality="audio"
)

# Mount WebRTC on FastAPI
stream.mount(app, path="/rtc")


@app.on_event("startup")
async def startup():
    """Load models on server startup"""
    load_models()
    initialize_chat()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve WebRTC client UI"""
    client_html_path = os.path.join(os.path.dirname(__file__), "client.html")
    if os.path.exists(client_html_path):
        with open(client_html_path, "r") as f:
            return f.read()
    else:
        return """
        <html>
            <body style="font-family: sans-serif; padding: 20px;">
                <h1>🎤 LFM2.5-Audio WebRTC Server</h1>
                <p><strong>Status:</strong> ✅ Running</p>
                <p><strong>Model:</strong> LiquidAI/LFM2.5-Audio-1.5B</p>
                <p><strong>WebRTC Endpoint:</strong> /rtc</p>
                <p>Client UI not found. Check server logs.</p>
            </body>
        </html>
        """


@app.post("/api/generate")
async def generate_response(audio: UploadFile):
    """HTTP fallback endpoint for audio processing (for client without WebRTC)"""
    global _model, _processor, _chat_state

    if _model is None or _processor is None:
        logger.error("Models not loaded")
        return {"error": "Models not loaded"}

    try:
        import io
        from pydub import AudioSegment
        import librosa

        # Read audio from upload
        audio_bytes = await audio.read()

        # Decode WebM/Opus audio using pydub
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
        except:
            # Fallback: try auto-detect format
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))

        # Convert to numpy array
        samples = np.array(audio_segment.get_array_of_samples())
        sr = audio_segment.frame_rate

        # Handle stereo
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = samples.mean(axis=1)

        # Convert to float32 in range [-1, 1]
        audio_array = samples.astype(np.float32) / 32768.0

        # Resample to 16kHz if needed
        if sr != SAMPLE_RATE:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Reshape to 2D (channels, samples) as required by add_audio
        audio_array = audio_array.reshape(1, -1)

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_array).float()

        logger.info(f"📥 Received {audio_tensor.shape[1]} samples ({audio_tensor.shape[1]/SAMPLE_RATE:.2f}s), sr: {sr}Hz, shape: {audio_tensor.shape}")

        # Add to chat state and prepare for generation
        _chat_state.add_audio(audio_tensor, SAMPLE_RATE)
        _chat_state.end_turn()
        _chat_state.new_turn("assistant")

        # Generate response via interleaved generation
        audio_tokens = []
        text_output = ""
        ttfa = None
        inference_start = time.perf_counter()

        with torch.no_grad():
            # generate_interleaved expects **chat_state unpacking
            for token_idx, token in enumerate(_model.generate_interleaved(
                **_chat_state,
                max_new_tokens=512,
                audio_temperature=1.0,
                audio_top_k=4
            )):
                # Check token type
                if token.numel() == 1:
                    # Text token
                    text_token = _processor.text.decode(token)
                    text_output += text_token

                    if ttfa is None:
                        ttfa = time.perf_counter() - inference_start
                        logger.info(f"🎵 TTFA: {ttfa*1000:.1f}ms")
                else:
                    # Audio token (8 codebook indices from Mimi)
                    audio_tokens.append(token)

        # Decode all audio tokens
        wav_output = None
        wav_bytes = None
        if audio_tokens:
            logger.info(f"📊 Total audio tokens: {len(audio_tokens)}")
            audio_codes = torch.stack(audio_tokens, dim=1)
            with torch.no_grad():
                # Decode audio codes to waveform
                wav_output = _processor.decode(audio_codes)

            # Convert to WAV format for browser playback
            import scipy.io.wavfile as wavfile
            wav_np = wav_output.squeeze().numpy() if wav_output.dim() > 1 else wav_output.numpy()
            wav_int16 = (wav_np * 32767).astype(np.int16)

            # Create WAV in memory
            wav_buffer = io.BytesIO()
            wavfile.write(wav_buffer, SAMPLE_RATE, wav_int16)
            wav_bytes = wav_buffer.getvalue()

            logger.info(f"📊 Generated {len(wav_int16)} audio samples ({len(wav_int16)/SAMPLE_RATE:.2f}s)")

        total_time = time.perf_counter() - inference_start
        audio_duration = len(audio_tokens) / 50.0 if audio_tokens else 0  # ~50 tokens/sec

        logger.info(f"✅ Generation complete: {total_time:.2f}s, TTFA: {ttfa*1000:.1f}ms if applicable")
        logger.info(f"📄 Full response: {text_output}")

        response = {
            "success": True,
            "text": text_output,
            "ttfa_ms": int(ttfa * 1000) if ttfa else 0,
            "total_ms": int(total_time * 1000),
            "audio_duration_s": audio_duration
        }

        # If audio exists, encode as base64 for browser
        if wav_bytes:
            import base64
            response["audio_base64"] = base64.b64encode(wav_bytes).decode('utf-8')

        return response

    except Exception as e:
        logger.error(f"❌ Error in generate_response: {e}", exc_info=True)
        return {"error": str(e)}


@app.get("/metrics")
async def get_metrics():
    """Return latency metrics"""
    return {
        "model": MODEL_ID,
        "mode": "interleaved_generation",
        "target_latency_ms": 1200,
        "note": "See WebRTC client for real-time metrics"
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting LFM2.5-Audio WebRTC server...")
    logger.info(f"📍 WebRTC endpoint: http://localhost:8000/rtc")
    logger.info(f"📍 Health check: http://localhost:8000/")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
