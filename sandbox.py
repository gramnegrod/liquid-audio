"""
LFM2.5-Audio Parameter Sandbox - Backend
Web interface for systematic parameter exploration and latency measurement
"""

import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit, disconnect

# ============================================================================
# IMPORTS & SETUP
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Model settings
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEXT_TEMPERATURE = None
DEFAULT_TEXT_TOP_K = None
DEFAULT_AUDIO_TEMPERATURE = 1.0
DEFAULT_AUDIO_TOP_K = 4
DEFAULT_MODE = "interleaved"
DEFAULT_SYSTEM_PROMPT = "Respond with interleaved text and audio."

# Server settings
DEFAULT_DEVICE = "mps"
DEFAULT_PORT = 7860

# Audio constraints
MIN_AUDIO_DURATION_S = 0.1
MAX_AUDIO_DURATION_S = 60.0
MIN_NEW_TOKENS = 8
MAX_NEW_TOKENS = 1024
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 1
MAX_TOP_K_TEXT = 100
MAX_TOP_K_AUDIO = 50

# System prompts for different tasks
SYSTEM_PROMPTS = {
    "interleaved": "Respond with interleaved text and audio.",
    "brief": "Respond briefly with audio.",
    "asr": "Perform ASR.",
    "tts_uk_female": "Perform TTS. Use the UK female voice.",
    "tts_us_male": "Perform TTS. Use the US male voice.",
    "tts_us_female": "Perform TTS. Use the US female voice.",
    "tts_uk_male": "Perform TTS. Use the UK male voice.",
}

# Parameter presets
PRESETS = {
    "low_latency": {
        "max_new_tokens": 320,
        "text_temperature": 0.5,
        "text_top_k": 1,
        "audio_temperature": 0.5,
        "audio_top_k": 1,
    },
    "balanced": {
        "max_new_tokens": 512,
        "text_temperature": 1.0,
        "text_top_k": 4,
        "audio_temperature": 1.0,
        "audio_top_k": 4,
    },
    "high_quality": {
        "max_new_tokens": 512,
        "text_temperature": 1.0,
        "text_top_k": 10,
        "audio_temperature": 1.0,
        "audio_top_k": 10,
    },
    "creative": {
        "max_new_tokens": 512,
        "text_temperature": 1.5,
        "text_top_k": 40,
        "audio_temperature": 1.5,
        "audio_top_k": 40,
    },
}

# CSV logging
EXPERIMENT_LOG_PATH = "experiment_log.csv"

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Flask app
app = Flask(__name__, template_folder="templates", static_folder="templates")

# WebSocket support for streaming
socketio = SocketIO(app, cors_allowed_origins="*")

# Model and processor (loaded on startup)
lfm2_audio = None
proc = None
DEVICE = None
LFMModality = None
ChatState = None

# Generation parameters
current_settings = {
    "mode": DEFAULT_MODE,
    "max_new_tokens": DEFAULT_MAX_TOKENS,
    "text_temperature": DEFAULT_TEXT_TEMPERATURE,
    "text_top_k": DEFAULT_TEXT_TOP_K,
    "audio_temperature": DEFAULT_AUDIO_TEMPERATURE,
    "audio_top_k": DEFAULT_AUDIO_TOP_K,
}

# System prompt
current_system_prompt = DEFAULT_SYSTEM_PROMPT

# Conversation manager (initialized after model load)
conversation_manager = None

# Metrics logger
metrics_logger = None


# ============================================================================
# CLASSES
# ============================================================================


class ConversationManager:
    """Manages multi-turn conversation state."""

    def __init__(self, processor):
        """Initialize conversation with ChatState.

        Args:
            processor: LFM2AudioProcessor instance for ChatState.
        """
        from liquid_audio import ChatState

        self.processor = processor
        self.chat_state = ChatState(processor)
        self.turn_count = 0
        logger.info("ConversationManager initialized")

    def reset(self) -> None:
        """Clear history and start fresh conversation."""
        from liquid_audio import ChatState

        self.chat_state = ChatState(self.processor)
        self.turn_count = 0
        logger.info("Conversation reset")

    def get_state(self) -> "ChatState":
        """Return current ChatState.

        Returns:
            Current ChatState object.
        """
        return self.chat_state

    def append_response(
        self, text_tokens: list, audio_tokens: list, modality_flags: list
    ) -> None:
        """Append assistant response to conversation history.

        Args:
            text_tokens: List of text token tensors.
            audio_tokens: List of audio token tensors.
            modality_flags: List of modality flags (0=text, 1=audio).
        """
        if not text_tokens or not audio_tokens:
            logger.warning("Empty tokens in append_response")
            return

        try:
            self.chat_state.append(
                text=torch.stack(text_tokens, 1),
                audio_out=torch.stack(audio_tokens, 1),
                modality_flag=torch.tensor(modality_flags, device=DEVICE),
            )
            self.chat_state.end_turn()
            self.turn_count += 1
            logger.info(f"Appended response to chat state (turn {self.turn_count})")
        except Exception as e:
            logger.error(f"Error appending response: {e}", exc_info=True)

    def new_system_turn(self, prompt: str) -> None:
        """Start new system turn with prompt.

        Args:
            prompt: System prompt text.
        """
        try:
            self.chat_state.new_turn("system")
            # Add system prompt as text (ChatState.add_text handles tokenization)
            self.chat_state.add_text(prompt)
            self.chat_state.end_turn()
            logger.info("System prompt appended to chat state")
        except Exception as e:
            logger.error(f"Error adding system turn: {e}", exc_info=True)

    def new_user_turn(self) -> None:
        """Start new user turn."""
        try:
            self.chat_state.new_turn("user")
            logger.debug("New user turn started")
        except Exception as e:
            logger.error(f"Error starting user turn: {e}", exc_info=True)

    def end_user_turn(self) -> None:
        """End current user turn."""
        try:
            self.chat_state.end_turn()
            logger.debug("User turn ended")
        except Exception as e:
            logger.error(f"Error ending user turn: {e}", exc_info=True)


class MetricsLogger:
    """CSV logger for experiment metrics."""

    def __init__(self, filepath: str):
        """Initialize metrics logger.

        Args:
            filepath: Path to experiment_log.csv.
        """
        self.filepath = filepath
        self.fieldnames = [
            "timestamp",
            "mode",
            "max_tokens",
            "text_temp",
            "text_top_k",
            "audio_temp",
            "audio_top_k",
            "ttft_ms",
            "total_ms",
            "audio_sec",
            "quality_rating",
        ]
        self.ensure_headers()
        logger.info(f"MetricsLogger initialized: {filepath}")

    def ensure_headers(self) -> None:
        """Create CSV file with headers if it doesn't exist."""
        if Path(self.filepath).exists():
            return

        try:
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            logger.info(f"Created {self.filepath} with headers")
        except Exception as e:
            logger.error(f"Error creating CSV: {e}", exc_info=True)

    def log_generation(self, metrics: dict) -> None:
        """Append one row to CSV.

        Args:
            metrics: Dict with keys matching fieldnames.
        """
        try:
            with open(self.filepath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                row = {
                    "timestamp": metrics.get("timestamp", datetime.now().isoformat()),
                    "mode": metrics.get("mode", "unknown"),
                    "max_tokens": metrics.get("max_new_tokens", 0),
                    "text_temp": metrics.get("text_temperature", 0),
                    "text_top_k": metrics.get("text_top_k", 0),
                    "audio_temp": metrics.get("audio_temperature", 0),
                    "audio_top_k": metrics.get("audio_top_k", 0),
                    "ttft_ms": metrics.get("ttft_ms", None),
                    "total_ms": metrics.get("total_ms", None),
                    "audio_sec": metrics.get("audio_duration_s", None),
                    "quality_rating": metrics.get("quality_rating", None),
                }
                writer.writerow(row)
            logger.debug("Logged generation metrics to CSV")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}", exc_info=True)

    def get_csv_data(self) -> str:
        """Return all CSV data as string.

        Returns:
            CSV file contents as string.
        """
        try:
            with open(self.filepath, "r") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("CSV file not found")
            return ""
        except Exception as e:
            logger.error(f"Error reading CSV: {e}", exc_info=True)
            return ""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def validate_settings(data: dict) -> list:
    """Validate generation parameters.

    Args:
        data: Dict of parameters to validate.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    if "max_new_tokens" in data:
        if not isinstance(data["max_new_tokens"], int):
            errors.append("max_new_tokens must be integer")
        elif not (MIN_NEW_TOKENS <= data["max_new_tokens"] <= MAX_NEW_TOKENS):
            errors.append(
                f"max_new_tokens must be {MIN_NEW_TOKENS}-{MAX_NEW_TOKENS}"
            )

    if "text_temperature" in data:
        val = data["text_temperature"]
        if val is not None and not isinstance(val, (int, float)):
            errors.append("text_temperature must be number or null")
        elif isinstance(val, (int, float)) and not (
            MIN_TEMPERATURE <= val <= MAX_TEMPERATURE
        ):
            errors.append(
                f"text_temperature must be {MIN_TEMPERATURE}-{MAX_TEMPERATURE} or null"
            )

    if "text_top_k" in data:
        val = data["text_top_k"]
        if val is not None and not isinstance(val, int):
            errors.append("text_top_k must be integer or null")
        elif isinstance(val, int) and not (MIN_TOP_K <= val <= MAX_TOP_K_TEXT):
            errors.append(f"text_top_k must be {MIN_TOP_K}-{MAX_TOP_K_TEXT} or null")

    if "audio_temperature" in data:
        val = data["audio_temperature"]
        if val is not None and not isinstance(val, (int, float)):
            errors.append("audio_temperature must be number or null")
        elif isinstance(val, (int, float)) and not (
            MIN_TEMPERATURE <= val <= MAX_TEMPERATURE
        ):
            errors.append(
                f"audio_temperature must be {MIN_TEMPERATURE}-{MAX_TEMPERATURE} or null"
            )

    if "audio_top_k" in data:
        val = data["audio_top_k"]
        if val is not None and not isinstance(val, int):
            errors.append("audio_top_k must be integer or null")
        elif isinstance(val, int) and not (MIN_TOP_K <= val <= MAX_TOP_K_AUDIO):
            errors.append(f"audio_top_k must be {MIN_TOP_K}-{MAX_TOP_K_AUDIO} or null")

    if "mode" in data:
        if data["mode"] not in ("interleaved", "sequential"):
            errors.append("mode must be 'interleaved' or 'sequential'")

    return errors


# ============================================================================
# AUDIO PROCESSING & GENERATION
# ============================================================================


def process_audio_input(sample_rate: int, audio_data: np.ndarray) -> tuple:
    """Convert audio to model input tensor.

    Args:
        sample_rate: Sample rate in Hz.
        audio_data: Audio samples as numpy array (int16 or float32).

    Returns:
        Tuple of (audio_float, duration_s).

    Raises:
        ValueError: If audio is too short or too long.
    """
    # Handle 2D arrays (squeeze)
    if audio_data.ndim == 2:
        audio_data = audio_data.squeeze()

    # Convert int16 to float32 if needed
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32767.0
    else:
        audio_float = audio_data.astype(np.float32)

    # Resample if needed
    if sample_rate != 16000:
        import librosa

        audio_float = librosa.resample(
            audio_float, orig_sr=sample_rate, target_sr=16000
        )

    duration = len(audio_float) / 16000.0

    # Validate duration
    if duration < MIN_AUDIO_DURATION_S:
        raise ValueError(f"Audio too short: {duration:.2f}s (min {MIN_AUDIO_DURATION_S}s)")
    if duration > MAX_AUDIO_DURATION_S:
        raise ValueError(f"Audio too long: {duration:.2f}s (max {MAX_AUDIO_DURATION_S}s)")

    return audio_float, duration


def generate_response_streaming(audio_float: np.ndarray):
    """Generate response using streaming events for WebSocket delivery.

    Yields events as generation proceeds:
        {"event": "ttft", "ttft_ms": float}
        {"event": "token", "text": str, "type": "text"}
        {"event": "audio_chunk", "pcm_int16": hex_str, "sample_rate": 24000, "num_samples": int}
        {"event": "final_metrics", "total_ms": float, "audio_duration_s": float, "text": str}
        {"event": "error", "message": str}

    Args:
        audio_float: Audio tensor (float32, 16kHz).
    """
    t_start = time.perf_counter()
    ttft_ms = None
    accumulated_text = ""
    audio_duration_s = 0.0

    try:
        # Add system prompt turn
        conversation_manager.new_system_turn(current_system_prompt)

        # Prepare audio tensor
        audio_tensor = torch.tensor(audio_float, dtype=torch.float32).unsqueeze(0)

        # Add user audio turn
        conversation_manager.new_user_turn()
        conversation_manager.chat_state.add_audio(audio_tensor, 16000)
        conversation_manager.chat_state.end_turn()
        conversation_manager.chat_state.new_turn("assistant")

        # Prepare generation parameters
        gen_params = {
            "max_new_tokens": current_settings["max_new_tokens"],
        }

        # Add optional parameters only if not None
        if current_settings["text_temperature"] is not None:
            gen_params["text_temperature"] = current_settings["text_temperature"]
        if current_settings["text_top_k"] is not None:
            gen_params["text_top_k"] = current_settings["text_top_k"]
        if current_settings["audio_temperature"] is not None:
            gen_params["audio_temperature"] = current_settings["audio_temperature"]
        if current_settings["audio_top_k"] is not None:
            gen_params["audio_top_k"] = current_settings["audio_top_k"]

        # Select generation method - FORCE INTERLEAVED for speech-to-speech
        mode = current_settings.get("mode", "interleaved")
        if mode != "interleaved":
            logger.warning(f"Sequential mode requested but forcing Interleaved for speech-to-speech conversation")
            mode = "interleaved"

        # For multi-turn, create fresh ChatState to avoid modality_flag issues
        if conversation_manager.turn_count > 0:
            logger.info("Creating fresh ChatState for turn > 0")
            conversation_manager.chat_state = ChatState(proc)
            conversation_manager.new_system_turn(current_system_prompt)
            conversation_manager.new_user_turn()
            conversation_manager.chat_state.add_audio(audio_tensor, 16000)
            conversation_manager.chat_state.end_turn()
            conversation_manager.chat_state.new_turn("assistant")

        # Always use interleaved for speech-to-speech conversation
        generator = lfm2_audio.generate_interleaved(
            **conversation_manager.chat_state, **gen_params
        )

        logger.info(f"Generating {mode} mode with params: {gen_params}")

        # Collect tokens and emit events
        out_text = []
        out_audio = []
        audio_chunk_buffer = []  # Buffer for audio chunk decoding

        with torch.no_grad():
            for t in generator:
                if t.numel() == 1:  # Text token
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t_start) * 1000
                        logger.info(f"[TTFT: {ttft_ms:.0f}ms]")
                        yield {"event": "ttft", "ttft_ms": ttft_ms}

                    out_text.append(t)
                    # Emit individual text token
                    try:
                        token_text = proc.text.decode([t.item()])
                        accumulated_text += token_text
                        yield {"event": "token", "text": token_text, "type": "text"}
                    except Exception as e:
                        logger.warning(f"Failed to decode text token: {e}")

                elif t.numel() == 8:  # Audio token
                    out_audio.append(t)
                    audio_chunk_buffer.append(t)

                    # Emit audio chunk every 12 tokens (interleaved pattern)
                    if len(audio_chunk_buffer) >= 12:
                        try:
                            chunk_codes = torch.stack(audio_chunk_buffer, dim=1).unsqueeze(0)
                            with torch.no_grad():
                                chunk_wav = proc.decode(chunk_codes)

                            # Convert to int16 PCM
                            pcm_int16 = (chunk_wav.squeeze().cpu().numpy() * 32767).astype(np.int16)
                            # Encode as hex for JSON transmission
                            pcm_hex = pcm_int16.tobytes().hex()

                            num_samples = len(pcm_int16)
                            audio_duration_s += num_samples / 24000.0

                            yield {
                                "event": "audio_chunk",
                                "pcm_int16": pcm_hex,
                                "sample_rate": 24000,
                                "num_samples": num_samples,
                            }
                            audio_chunk_buffer = []
                        except Exception as e:
                            logger.warning(f"Failed to decode audio chunk: {e}")

        # Emit final audio chunk if any remaining tokens
        if audio_chunk_buffer:
            try:
                chunk_codes = torch.stack(audio_chunk_buffer, dim=1).unsqueeze(0)
                with torch.no_grad():
                    chunk_wav = proc.decode(chunk_codes)

                pcm_int16 = (chunk_wav.squeeze().cpu().numpy() * 32767).astype(np.int16)
                pcm_hex = pcm_int16.tobytes().hex()

                num_samples = len(pcm_int16)
                audio_duration_s += num_samples / 24000.0

                yield {
                    "event": "audio_chunk",
                    "pcm_int16": pcm_hex,
                    "sample_rate": 24000,
                    "num_samples": num_samples,
                }
            except Exception as e:
                logger.warning(f"Failed to decode final audio chunk: {e}")

        logger.info(f"Generated: {len(out_text)} text tokens, {len(out_audio)} audio tokens")

        # Update chat state with response
        if out_text and out_audio:
            out_modality = [0 if isinstance(t, torch.Tensor) and t.numel() == 1 else 1 for t in out_text + out_audio]
            conversation_manager.append_response(out_text, out_audio, out_modality)

        # Calculate total latency
        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(f"Total latency: {total_ms:.0f}ms")

        # Emit final metrics
        yield {
            "event": "final_metrics",
            "total_ms": total_ms,
            "audio_duration_s": audio_duration_s,
            "text": accumulated_text,
        }

    except Exception as e:
        logger.error(f"Streaming generation error: {e}", exc_info=True)
        yield {"event": "error", "message": str(e)}


def generate_response(audio_float: np.ndarray) -> dict:
    """Generate response using current model and settings.

    Args:
        audio_float: Audio tensor (float32, 16kHz).

    Returns:
        Dict with keys: ttft_ms, total_ms, audio_duration_s, audio_array.
    """
    metrics = {
        "ttft_ms": None,
        "total_ms": None,
        "audio_duration_s": None,
        "audio_array": None,
        "text": "",
    }

    t_start = time.perf_counter()

    try:
        # Add system prompt turn
        conversation_manager.new_system_turn(current_system_prompt)

        # Prepare audio tensor
        audio_tensor = torch.tensor(audio_float, dtype=torch.float32).unsqueeze(0)

        # Add user audio turn
        conversation_manager.new_user_turn()
        conversation_manager.chat_state.add_audio(audio_tensor, 16000)
        conversation_manager.chat_state.end_turn()
        conversation_manager.chat_state.new_turn("assistant")

        # Prepare generation parameters
        gen_params = {
            "max_new_tokens": current_settings["max_new_tokens"],
        }

        # Add optional parameters only if not None
        if current_settings["text_temperature"] is not None:
            gen_params["text_temperature"] = current_settings["text_temperature"]
        if current_settings["text_top_k"] is not None:
            gen_params["text_top_k"] = current_settings["text_top_k"]
        if current_settings["audio_temperature"] is not None:
            gen_params["audio_temperature"] = current_settings["audio_temperature"]
        if current_settings["audio_top_k"] is not None:
            gen_params["audio_top_k"] = current_settings["audio_top_k"]

        # Select generation method - FORCE INTERLEAVED for speech-to-speech
        # Sequential mode is for ASR/TTS only, not conversational chat
        mode = current_settings.get("mode", "interleaved")
        if mode != "interleaved":
            logger.warning(f"Sequential mode requested but forcing Interleaved for speech-to-speech conversation")
            mode = "interleaved"

        # For multi-turn, create fresh ChatState to avoid modality_flag issues from accumulated audio_in
        # This keeps the first turn's response context but starts fresh for each generation
        if conversation_manager.turn_count > 0:
            logger.info("Creating fresh ChatState for turn > 0")
            conversation_manager.chat_state = ChatState(proc)
            conversation_manager.new_system_turn(current_system_prompt)
            # Re-add the current user audio
            conversation_manager.new_user_turn()
            conversation_manager.chat_state.add_audio(audio_tensor, 16000)
            conversation_manager.chat_state.end_turn()
            conversation_manager.chat_state.new_turn("assistant")

        # Always use interleaved for speech-to-speech conversation
        generator = lfm2_audio.generate_interleaved(
            **conversation_manager.chat_state, **gen_params
        )

        # Collect tokens
        out_text = []
        out_audio = []
        out_modality = []

        logger.info(
            f"Generating {mode} mode with params: {gen_params}"
        )

        with torch.no_grad():
            for t in generator:
                if t.numel() == 1:  # Text token
                    if metrics["ttft_ms"] is None:
                        metrics["ttft_ms"] = (
                            time.perf_counter() - t_start
                        ) * 1000
                        logger.info(f"[TTFT: {metrics['ttft_ms']:.0f}ms]")
                    out_text.append(t)
                    out_modality.append(0)  # 0 = text
                elif t.numel() == 8:  # Audio token
                    out_audio.append(t)
                    out_modality.append(1)  # 1 = audio

        logger.info(
            f"Generated: {len(out_text)} text tokens, {len(out_audio)} audio tokens"
        )

        # Decode text tokens back to string
        if out_text:
            # Flatten and concatenate all text tokens into a single list
            text_ids = [t.item() if t.numel() == 1 else t.flatten()[0].item() for t in out_text]
            metrics["text"] = proc.text.decode(text_ids)
            logger.info(f"Decoded text: {metrics['text']}")

        # Decode audio - filter special tokens (2048 = end marker)
        valid_audio = [t for t in out_audio if not (t == 2048).any()]
        logger.info(
            f"Audio tokens: {len(out_audio)} total, {len(valid_audio)} valid"
        )

        if valid_audio:
            # Stack and decode
            audio_codes = torch.stack(valid_audio, dim=1).unsqueeze(0)
            with torch.no_grad():
                wav_tensor = proc.decode(audio_codes)

            # Convert to int16 numpy array
            audio_array = (
                wav_tensor.squeeze().cpu().numpy() * 32767
            ).astype(np.int16)
            audio_array = audio_array.reshape(1, -1)

            metrics["audio_array"] = audio_array
            metrics["audio_duration_s"] = len(audio_array[0]) / 24000.0

            logger.info(f"Audio: {metrics['audio_duration_s']:.2f}s generated")
        else:
            logger.warning("No valid audio tokens generated")

        # Update chat state with response
        if out_text and valid_audio:
            conversation_manager.append_response(
                out_text, out_audio, out_modality
            )

        # Calculate total latency
        metrics["total_ms"] = (time.perf_counter() - t_start) * 1000

        logger.info(f"Total latency: {metrics['total_ms']:.0f}ms")

        return metrics

    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise


# ============================================================================
# ROUTE HANDLERS
# ============================================================================


@app.route("/")
def index():
    """Serve main HTML page."""
    return render_template("index.html")


@app.route("/simple")
def simple():
    """Serve simplified chat interface."""
    return render_template("simple.html")


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update generation parameters.

    Request JSON:
        {
            "mode": "interleaved",
            "max_new_tokens": 512,
            "text_temperature": 1.0,
            "text_top_k": 4,
            "audio_temperature": 1.0,
            "audio_top_k": 4
        }

    Returns:
        {"status": "ok", "settings": {...}} or {"status": "error", "errors": [...]}
    """
    global current_settings

    try:
        data = request.json or {}
        errors = validate_settings(data)

        if errors:
            logger.warning(f"Invalid settings: {errors}")
            return jsonify({"status": "error", "errors": errors}), 400

        current_settings.update(data)
        logger.info(f"Updated settings: {data}")
        return jsonify({"status": "ok", "settings": current_settings})

    except Exception as e:
        logger.error(f"Error updating settings: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/system-prompt", methods=["POST"])
def set_system_prompt():
    """Update system prompt.

    Request JSON:
        {"prompt": "Custom system prompt..."}

    Returns:
        {"status": "ok", "prompt": "..."}
    """
    global current_system_prompt

    try:
        data = request.json or {}
        prompt = data.get("prompt", DEFAULT_SYSTEM_PROMPT)

        if not isinstance(prompt, str) or not prompt.strip():
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "prompt must be non-empty string",
                    }
                ),
                400,
            )

        current_system_prompt = prompt
        logger.info(f"System prompt updated: {prompt[:50]}...")
        return jsonify({"status": "ok", "prompt": current_system_prompt})

    except Exception as e:
        logger.error(f"Error setting system prompt: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/preset/<name>", methods=["POST"])
def apply_preset(name):
    """Apply parameter preset.

    Args:
        name: Preset name (low_latency, balanced, high_quality, creative)

    Returns:
        {"status": "ok", "settings": {...}} or error
    """
    global current_settings

    try:
        if name not in PRESETS:
            available = ", ".join(PRESETS.keys())
            logger.warning(f"Unknown preset: {name}")
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Unknown preset. Available: {available}",
                    }
                ),
                400,
            )

        current_settings.update(PRESETS[name])
        logger.info(f"Applied preset: {name}")
        return jsonify({"status": "ok", "settings": current_settings})

    except Exception as e:
        logger.error(f"Error applying preset: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset_conversation():
    """Reset conversation history."""
    try:
        conversation_manager.reset()
        logger.info("Conversation reset via API")
        return jsonify({"status": "ok", "message": "Conversation reset"})
    except Exception as e:
        logger.error(f"Error resetting conversation: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate response from audio input.

    Request: multipart/form-data with:
        - audio: WAV file (int16 PCM)
        - sample_rate: int (Hz)

    Returns:
        {"status": "ok", "ttft_ms": ..., "total_ms": ..., "audio_duration_s": ..., "audio_base64": ...}
    """
    try:
        # Get audio file and sample rate
        if "audio" not in request.files:
            return (
                jsonify({"status": "error", "message": "Missing audio file"}),
                400,
            )

        audio_file = request.files["audio"]
        requested_sr = request.form.get("sample_rate", 48000, type=int)

        # Save temp file and load with librosa (handles WebM, WAV, etc.)
        import tempfile
        import librosa

        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
            audio_file.save(tmp.name)
            audio_float, sr = librosa.load(tmp.name, sr=None, mono=True)
            os.unlink(tmp.name)

        sample_rate = sr or requested_sr
        logger.info(f"Received audio: {len(audio_float)} samples @ {sample_rate}Hz")

        # CRITICAL: Resample to 16kHz (model requirement)
        # This is required because WebM often loads at 48kHz but model expects 16kHz
        TARGET_SR = 16000
        if sample_rate != TARGET_SR:
            logger.info(f"Resampling {sample_rate}Hz → {TARGET_SR}Hz")
            audio_float = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=TARGET_SR)
            sample_rate = TARGET_SR
            logger.info(f"Resampled: {len(audio_float)} samples @ {sample_rate}Hz")

        # Process audio (librosa already returns float32 mono)
        duration = len(audio_float) / sample_rate
        logger.info(f"Processed: {duration:.2f}s audio")

        # Truncate to max 10 seconds (model expects short utterances)
        max_duration = 10.0
        if duration > max_duration:
            max_samples = int(max_duration * sample_rate)
            audio_float = audio_float[:max_samples]
            logger.info(f"Truncated to {max_duration}s ({max_samples} samples)")

        # Generate response
        metrics = generate_response(audio_float)

        # Prepare response
        import base64

        response = {
            "status": "ok",
            "ttft_ms": metrics["ttft_ms"],
            "total_ms": metrics["total_ms"],
            "audio_duration_s": metrics["audio_duration_s"],
            "text": metrics["text"],
            "settings": current_settings,
            "mode": current_settings["mode"],
        }

        # Add audio if generated
        if metrics["audio_array"] is not None:
            # Convert to base64 for JSON response
            audio_bytes = metrics["audio_array"].tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            response["audio_base64"] = audio_b64
            response["audio_shape"] = list(metrics["audio_array"].shape)

        # Log to CSV
        metrics_logger.log_generation(
            {
                **response,
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info("Response generated and logged")
        return jsonify(response)

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/log", methods=["GET"])
def get_log():
    """Download experiment log as CSV."""
    try:
        csv_data = metrics_logger.get_csv_data()
        return (
            csv_data,
            200,
            {
                "Content-Type": "text/csv",
                "Content-Disposition": 'attachment; filename="experiment_log.csv"',
            },
        )
    except Exception as e:
        logger.error(f"Error getting log: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/rate", methods=["POST"])
def rate_generation():
    """Save quality rating for last generation.

    Request JSON:
        {"rating": 5}

    Returns:
        {"status": "ok"}
    """
    try:
        data = request.json or {}
        rating = data.get("rating")

        if not isinstance(rating, int) or not (1 <= rating <= 5):
            return (
                jsonify({"status": "error", "message": "rating must be 1-5"}),
                400,
            )

        logger.info(f"Quality rating: {rating}/5")
        return jsonify({"status": "ok"})

    except Exception as e:
        logger.error(f"Error rating generation: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================================
# WEBSOCKET HANDLERS (Streaming)
# ============================================================================


@socketio.on("generate")
def ws_generate(data):
    """WebSocket streaming generation endpoint.

    Receives audio and emits events as generation proceeds.
    """
    try:
        logger.info("WebSocket /generate request received")

        # Get audio blob and sample rate
        audio_b64 = data.get("audio")
        sample_rate = data.get("sample_rate", 48000)

        if not audio_b64:
            emit("error", {"message": "Missing audio data"})
            return

        # Decode base64 audio
        import base64
        import tempfile
        import librosa

        audio_bytes = base64.b64decode(audio_b64)

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            audio_float, sr = librosa.load(tmp.name, sr=None, mono=True)
            os.unlink(tmp.name)

        sample_rate = sr or sample_rate
        logger.info(f"Received audio: {len(audio_float)} samples @ {sample_rate}Hz")

        # CRITICAL: Resample to 16kHz (model requirement)
        TARGET_SR = 16000
        if sample_rate != TARGET_SR:
            logger.info(f"Resampling {sample_rate}Hz → {TARGET_SR}Hz")
            audio_float = librosa.resample(
                audio_float, orig_sr=sample_rate, target_sr=TARGET_SR
            )
            sample_rate = TARGET_SR

        # Truncate to max 10 seconds
        max_duration = 10.0
        duration = len(audio_float) / sample_rate
        if duration > max_duration:
            max_samples = int(max_duration * sample_rate)
            audio_float = audio_float[:max_samples]
            logger.info(f"Truncated to {max_duration}s ({max_samples} samples)")

        # Stream generation events
        for event in generate_response_streaming(audio_float):
            emit(event["event"], {k: v for k, v in event.items() if k != "event"})

        logger.info("WebSocket generation completed")

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        emit("error", {"message": str(e)})


@socketio.on("connect")
def handle_connect():
    """Handle WebSocket connection."""
    logger.info("WebSocket client connected")


@socketio.on("disconnect")
def handle_disconnect():
    """Handle WebSocket disconnection."""
    logger.info("WebSocket client disconnected")


# ============================================================================
# INITIALIZATION & STARTUP
# ============================================================================


def load_models():
    """Load LFM2.5-Audio model on startup."""
    global lfm2_audio, proc, DEVICE, LFMModality, ChatState, conversation_manager, metrics_logger

    try:
        logger.info("=" * 70)
        logger.info("Loading LFM2.5-Audio models...")
        logger.info("=" * 70)

        # Import model loading
        from liquid_audio import ChatState as _ChatState
        from liquid_audio.utils import LFMModality as _LFMModality
        from liquid_audio.demo.model import DEVICE as model_device
        from liquid_audio.demo.model import lfm2_audio as model
        from liquid_audio.demo.model import proc as processor

        DEVICE = model_device
        lfm2_audio = model
        proc = processor
        LFMModality = _LFMModality
        ChatState = _ChatState

        logger.info(f"✓ Models loaded on {DEVICE.upper()}")
        logger.info(f"✓ Processor: {proc}")
        logger.info(f"✓ Model: {lfm2_audio}")

        # Initialize conversation manager
        conversation_manager = ConversationManager(proc)

        # Initialize metrics logger
        metrics_logger = MetricsLogger(EXPERIMENT_LOG_PATH)

        logger.info("=" * 70)
        logger.info("Initialization complete!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        raise


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    # Load models before starting server
    load_models()

    # Print startup info
    logger.info("")
    logger.info("Starting LFM2.5-Audio Parameter Sandbox...")
    logger.info(f"Open browser: http://127.0.0.1:{DEFAULT_PORT}")
    logger.info("Press Ctrl+C to stop")
    logger.info("")

    # Run Flask app with WebSocket support
    socketio.run(app, host="127.0.0.1", port=DEFAULT_PORT, debug=False, allow_unsafe_werkzeug=True)
