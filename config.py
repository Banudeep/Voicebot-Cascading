"""
Configuration file for Voice Agent
Stores API keys, settings, and thresholds
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_SERVICE_TOKEN")

# Azure OpenAI Settings
USE_AZURE = os.getenv("USE_AZURE", "false").lower() == "true"
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# Strip quotes and whitespace from deployment name (common issue in Cloud Run)
_deployment_raw = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_DEPLOYMENT = _deployment_raw.strip('"\'') if _deployment_raw else None
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")

# Azure Speech Services Settings
USE_AZURE_SPEECH = os.getenv("USE_AZURE_SPEECH", "false").lower() == "true"
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_SPEECH_ENDPOINT = os.getenv("AZURE_SPEECH_ENDPOINT")  # Optional, auto-constructed if not provided
AZURE_SPEECH_LANGUAGE = os.getenv("AZURE_SPEECH_LANGUAGE", "en-US")
AZURE_SPEECH_VOICE = os.getenv("AZURE_SPEECH_VOICE", "en-US-JennyNeural")  # TTS voice

# OpenAI Settings
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # Use deployment name for Azure
OPENAI_TEMPERATURE = 0.7
OPENAI_MAX_TOKENS = 500

# STT Provider Selection
STT_PROVIDER = os.getenv("STT_PROVIDER", "deepgram").lower()  # "deepgram" or "azure"

# Deepgram STT Settings
STT_MODEL = "flux-general-en"  # Changed to match working STT.py
STT_LANGUAGE = "en-US"
STT_ENCODING = "linear16"
STT_SAMPLE_RATE = 16000  # 16kHz is optimal for Deepgram speech recognition
STT_CHANNELS = 1
STT_CONFIDENCE_THRESHOLD = 0.5  # Lowered threshold for better recognition

# Azure Speech STT Settings
AZURE_STT_SAMPLE_RATE = 16000  # 16kHz for Azure Speech
AZURE_STT_CHANNELS = 1
AZURE_STT_FORMAT = "PCM"  # PCM format

# TTS Provider Selection
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "deepgram").lower()  # "deepgram" or "azure"

# Deepgram TTS Settings
TTS_MODEL = "aura-asteria-en"
TTS_ENCODING = "linear16"
TTS_SAMPLE_RATE = 16000  # Match STT sample rate
TTS_CONTAINER = "none"

# Azure Speech TTS Settings
AZURE_TTS_SAMPLE_RATE = 24000  # 24kHz for Azure TTS (standard)
AZURE_TTS_FORMAT = "audio-16khz-128kbitrate-mono-mp3"  # or "raw-16khz-16bit-mono-pcm"

# Audio Settings
CHUNK_SIZE = 8192
AUDIO_FORMAT = "int16"
INPUT_DEVICE = None  # None = default mic
OUTPUT_DEVICE = None  # None = default speaker

# Voice Activity Detection
VAD_THRESHOLD = 0.005  # Voice activity threshold (0-1) - Lowered for better sensitivity
SILENCE_DURATION = 1.5  # Seconds of silence to end speech
MIN_SPEECH_DURATION = 0.3  # Minimum speech duration in seconds
END_OF_SPEECH_TIMEOUT = 0.8  # Wait this many seconds after last transcript before processing (reduced for faster response)

# Conversation Settings
SYSTEM_PROMPT = """You are a helpful voice assistant. Keep your responses concise and natural 
for voice conversation, ideally 1-3 sentences unless more detail is specifically requested. 
Speak in a friendly, conversational tone."""

MAX_CONVERSATION_HISTORY = 20  # Keep last N messages
ENABLE_TOOLS = True  # Enable MCP tools

# MCP Server Settings
MCP_SERVER_PATH = "../Chat-agent/server.py"

# Performance Settings
STT_BUFFER_SIZE = 4096
TTS_BUFFER_SIZE = 8192
AUDIO_QUEUE_MAXSIZE = 500  # Increased from 100 to handle more audio

# Recording Settings
ENABLE_RECORDINGS = os.getenv("ENABLE_RECORDINGS", "false").lower() == "true"

# Debug Settings
DEBUG = True
VERBOSE = False  # Disable verbose to reduce spam (VAD working now)

# Validate API keys
def validate_config():
    """Validate that required API keys are set"""
    errors = []
    
    if USE_AZURE:
        # Azure OpenAI mode
        if not AZURE_OPENAI_ENDPOINT:
            errors.append("AZURE_OPENAI_ENDPOINT not set")
        if not AZURE_OPENAI_API_KEY:
            errors.append("AZURE_OPENAI_API_KEY not set")
        if not AZURE_OPENAI_DEPLOYMENT:
            errors.append("AZURE_OPENAI_DEPLOYMENT not set")
    else:
        # Standard OpenAI mode
        if not OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY not set")
    
    # Validate STT provider
    if STT_PROVIDER == "deepgram":
        if not DEEPGRAM_API_KEY:
            errors.append("DEEPGRAM_API_KEY not set (required for Deepgram STT)")
    elif STT_PROVIDER == "azure":
        if not AZURE_SPEECH_KEY:
            errors.append("AZURE_SPEECH_KEY not set (required for Azure Speech STT)")
        if not AZURE_SPEECH_REGION:
            errors.append("AZURE_SPEECH_REGION not set (required for Azure Speech STT)")
    else:
        errors.append(f"Invalid STT_PROVIDER: {STT_PROVIDER} (must be 'deepgram' or 'azure')")
    
    # Validate TTS provider
    if TTS_PROVIDER == "deepgram":
        if not DEEPGRAM_API_KEY:
            errors.append("DEEPGRAM_API_KEY not set (required for Deepgram TTS)")
    elif TTS_PROVIDER == "azure":
        if not AZURE_SPEECH_KEY:
            errors.append("AZURE_SPEECH_KEY not set (required for Azure Speech TTS)")
        if not AZURE_SPEECH_REGION:
            errors.append("AZURE_SPEECH_REGION not set (required for Azure Speech TTS)")
    else:
        errors.append(f"Invalid TTS_PROVIDER: {TTS_PROVIDER} (must be 'deepgram' or 'azure')")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    if VERBOSE:
        print("✓ Configuration validated")

if __name__ == "__main__":
    validate_config()
    print("✓ All required API keys are configured")

