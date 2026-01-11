"""
LFM2.5-Audio WebRTC Implementation
Real-time speech-to-speech streaming with <1.2s latency
"""

__version__ = "0.1.0"
__author__ = "LFM2.5-Audio WebRTC Project"

from .server import app, load_models, initialize_chat

__all__ = ["app", "load_models", "initialize_chat"]
