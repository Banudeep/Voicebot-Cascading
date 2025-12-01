"""
Text-to-Speech Streaming Module
Supports both Deepgram and Azure Speech Services
"""
import asyncio
import aiohttp
import config

# Conditional imports based on provider
if config.TTS_PROVIDER == "azure":
    import azure.cognitiveservices.speech as speechsdk
    import io

class TTSStream:
    """Streaming text-to-speech processor"""
    
    def __init__(self):
        self.provider = config.TTS_PROVIDER
        self.audio_queue = asyncio.Queue()
        
        if self.provider == "deepgram":
            self.api_key = config.DEEPGRAM_API_KEY
        elif self.provider == "azure":
            self._init_azure()
        else:
            raise ValueError(f"Unsupported TTS provider: {self.provider}")
    
    def _init_azure(self):
        """Initialize Azure Speech TTS"""
        if not config.AZURE_SPEECH_KEY or not config.AZURE_SPEECH_REGION:
            raise ValueError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set for Azure Speech")
        
        # Create speech config
        if config.AZURE_SPEECH_ENDPOINT:
            self.speech_config = speechsdk.SpeechConfig(
                endpoint=config.AZURE_SPEECH_ENDPOINT,
                subscription=config.AZURE_SPEECH_KEY
            )
        else:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=config.AZURE_SPEECH_KEY,
                region=config.AZURE_SPEECH_REGION
            )
        
        # Set voice
        self.speech_config.speech_synthesis_voice_name = config.AZURE_SPEECH_VOICE
        
        # No audio_config needed - we'll use None to get audio data directly
        # When audio_config=None, the synthesizer returns audio data instead of playing
    
    async def synthesize(self, text: str) -> bytes:
        """Convert text to speech audio"""
        if self.provider == "deepgram":
            return await self._synthesize_deepgram(text)
        elif self.provider == "azure":
            return await self._synthesize_azure(text)
    
    async def _synthesize_deepgram(self, text: str) -> bytes:
        """Convert text to speech using Deepgram"""
        try:
            if config.VERBOSE:
                print(f"🔊 Synthesizing with Deepgram: {text[:50]}...")
            
            url = (
                f"https://api.deepgram.com/v1/speak"
                f"?model={config.TTS_MODEL}"
                f"&encoding={config.TTS_ENCODING}"
                f"&sample_rate={config.TTS_SAMPLE_RATE}"
                f"&container={config.TTS_CONTAINER}"
            )
            
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {"text": text}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=30) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"❌ Deepgram API error {response.status}: {error_text}")
                        return b""
                    
                    audio_data = bytearray()
                    async for chunk in response.content.iter_any():
                        audio_data.extend(chunk)
                    
                    audio_bytes = bytes(audio_data)
                    
                    if config.DEBUG:
                        print(f"✓ Deepgram TTS generated {len(audio_bytes)} bytes")
                    
                    return audio_bytes
            
        except Exception as e:
            print(f"❌ Deepgram TTS error: {e}")
            import traceback
            traceback.print_exc()
            return b""
    
    async def _synthesize_azure(self, text: str) -> bytes:
        """Convert text to speech using Azure Speech"""
        try:
            if config.VERBOSE:
                print(f"🔊 Synthesizing with Azure Speech: {text[:50]}...")
            
            # Create synthesizer (no audio config = returns audio data)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=None  # None = return audio data instead of playing
            )
            
            # Run synthesis in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: synthesizer.speak_text_async(text).get()
            )
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Get audio data from result
                audio_bytes = bytes(result.audio_data)
                
                if config.DEBUG:
                    print(f"✓ Azure TTS generated {len(audio_bytes)} bytes")
                
                return audio_bytes
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = speechsdk.CancellationDetails(result)
                print(f"❌ Azure TTS canceled: {cancellation.reason}")
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    print(f"   Error details: {cancellation.error_details}")
                return b""
            else:
                print(f"❌ Azure TTS failed: {result.reason}")
                return b""
            
        except Exception as e:
            print(f"❌ Azure TTS error: {e}")
            import traceback
            traceback.print_exc()
            return b""
    
    async def synthesize_stream(self, text: str, chunk_callback):
        """Stream audio synthesis with callback for each chunk"""
        if self.provider == "deepgram":
            await self._synthesize_stream_deepgram(text, chunk_callback)
        elif self.provider == "azure":
            await self._synthesize_stream_azure(text, chunk_callback)
    
    async def _synthesize_stream_deepgram(self, text: str, chunk_callback):
        """Stream Deepgram TTS"""
        try:
            url = (
                f"https://api.deepgram.com/v1/speak"
                f"?model={config.TTS_MODEL}"
                f"&encoding={config.TTS_ENCODING}"
                f"&sample_rate={config.TTS_SAMPLE_RATE}"
                f"&container={config.TTS_CONTAINER}"
            )
            
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {"text": text}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=30) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"❌ Deepgram API error {response.status}: {error_text}")
                        return
                    
                    total_bytes = 0
                    async for chunk in response.content.iter_any():
                        if chunk:
                            total_bytes += len(chunk)
                            await chunk_callback(chunk)
                    
                    if config.DEBUG:
                        print(f"✓ Streamed {total_bytes} bytes from Deepgram")
                
        except Exception as e:
            print(f"❌ Deepgram TTS streaming error: {e}")
            import traceback
            traceback.print_exc()
    
    async def _synthesize_stream_azure(self, text: str, chunk_callback):
        """Stream Azure TTS"""
        try:
            # Azure Speech doesn't have native streaming like Deepgram
            # So we'll synthesize and then stream the chunks
            audio_bytes = await self._synthesize_azure(text)
            
            if audio_bytes:
                # Stream in chunks
                chunk_size = 4096
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    await chunk_callback(chunk)
            
        except Exception as e:
            print(f"❌ Azure TTS streaming error: {e}")
            import traceback
            traceback.print_exc()


async def test_tts():
    """Test TTS synthesis"""
    print(f"Testing {config.TTS_PROVIDER.upper()} TTS stream...")
    
    tts = TTSStream()
    
    audio = await tts.synthesize("Hello! This is a test of the text to speech system.")
    
    if audio:
        print(f"✓ Generated {len(audio)} bytes of audio")
        
        # Optionally save to file
        extension = "raw" if config.TTS_PROVIDER == "deepgram" else "wav"
        filename = f"test_tts_{config.TTS_PROVIDER}.{extension}"
        with open(filename, "wb") as f:
            f.write(audio)
        print(f"✓ Saved to {filename}")
    
    print("✓ TTS test complete")


if __name__ == "__main__":
    asyncio.run(test_tts())
