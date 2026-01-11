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

from fastapi import FastAPI
from fastrtc import Stream, ReplyOnPause
import torch

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
        _processor = LFM2AudioProcessor.from_pretrained(MODEL_ID)
        _model = LFM2AudioModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,  # Use BF16 for M4 efficiency
            device_map="auto"  # Use MPS on macOS
        )
        _model.eval()
        logger.info("✅ Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
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


@app.get("/")
async def root():
    """Health check"""
    return {"status": "ok", "model": MODEL_ID}


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
