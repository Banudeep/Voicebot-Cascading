"""
Speech-to-Text Streaming Module
Supports both Deepgram and Azure Speech Services
"""
import asyncio
import config

# Conditional imports based on provider
if config.STT_PROVIDER == "deepgram":
    from deepgram import AsyncDeepgramClient
    from deepgram.core.events import EventType
elif config.STT_PROVIDER == "azure":
    import azure.cognitiveservices.speech as speechsdk
    from azure.cognitiveservices.speech.audio import AudioStreamFormat, PushAudioInputStream

class STTStream:
    """Streaming speech-to-text processor"""
    
    def __init__(self):
        self.provider = config.STT_PROVIDER
        self.transcript_queue = asyncio.Queue()
        self.is_connected = False
        self._connection_task = None
        self._last_transcript = None
        self._event_loop = None  # Store event loop for Azure callbacks
        
        if self.provider == "deepgram":
            self._init_deepgram()
        elif self.provider == "azure":
            self._init_azure()
        else:
            raise ValueError(f"Unsupported STT provider: {self.provider}")
    
    def _init_deepgram(self):
        """Initialize Deepgram client"""
        self.client = AsyncDeepgramClient()
        self.connection = None
        self._warned_disconnected = False
    
    def _init_azure(self):
        """Initialize Azure Speech client"""
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
        
        self.speech_config.speech_recognition_language = config.AZURE_SPEECH_LANGUAGE
        
        # Audio input stream
        self.audio_format = AudioStreamFormat(
            samples_per_second=config.AZURE_STT_SAMPLE_RATE,
            bits_per_sample=16,
            channels=config.AZURE_STT_CHANNELS
        )
        self.push_stream = PushAudioInputStream(stream_format=self.audio_format)
        self.audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)
        
        # Recognizer
        self.recognizer = None
        self._transcript_buffer = []
        
    async def connect(self):
        """Connect to STT service"""
        if self.provider == "deepgram":
            return await self._connect_deepgram()
        elif self.provider == "azure":
            return await self._connect_azure()
    
    async def _connect_deepgram(self):
        """Connect to Deepgram STT service"""
        try:
            print("🔌 Starting Deepgram STT connection...")
            self._connection_task = asyncio.create_task(self._maintain_deepgram_connection())
            
            # Wait for connection
            for i in range(50):  # 5 seconds
                await asyncio.sleep(0.1)
                if self.is_connected:
                    print("✓ Deepgram STT connected!")
                    return True
            
            print("⚠️ Deepgram STT connection timeout - continuing anyway")
            return True
            
        except Exception as e:
            print(f"❌ Deepgram STT connection error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _connect_azure(self):
        """Connect to Azure Speech STT service"""
        try:
            print("🔌 Starting Azure Speech STT connection...")
            print(f"   Language: {config.AZURE_SPEECH_LANGUAGE}")
            print(f"   Sample Rate: {config.AZURE_STT_SAMPLE_RATE}")
            
            # Store event loop for callbacks
            self._event_loop = asyncio.get_event_loop()
            
            # Create recognizer
            self.recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=self.audio_config
            )
            
            # Set up event handlers
            # IMPORTANT: Connect to 'recognizing' for interim results (faster) AND 'recognized' for final results
            self.recognizer.recognizing.connect(self._on_azure_recognizing)  # Interim results - fires quickly
            self.recognizer.recognized.connect(self._on_azure_recognized)  # Final results - fires after silence
            self.recognizer.session_started.connect(self._on_azure_session_started)
            self.recognizer.session_stopped.connect(self._on_azure_session_stopped)
            self.recognizer.canceled.connect(self._on_azure_canceled)
            
            # Configure recognition settings for faster response
            # Set property to return interim results more frequently
            self.speech_config.set_property(
                speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, 
                "500"  # 500ms silence timeout (faster than default)
            )
            
            # Start continuous recognition
            self.recognizer.start_continuous_recognition_async()
            
            self.is_connected = True
            print("✓ Azure Speech STT connected!")
            return True
            
        except Exception as e:
            print(f"❌ Azure Speech STT connection error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _maintain_deepgram_connection(self):
        """Maintain Deepgram connection"""
        try:
            print(f"📡 Connecting to Deepgram STT...")
            print(f"   Model: {config.STT_MODEL}")
            print(f"   Encoding: {config.STT_ENCODING}")
            print(f"   Sample Rate: {config.STT_SAMPLE_RATE}")
            
            async with self.client.listen.v2.connect(
                model=config.STT_MODEL,
                encoding=config.STT_ENCODING,
                sample_rate=str(config.STT_SAMPLE_RATE)
            ) as connection:
                
                print("✅ Deepgram STT context entered")
                self.connection = connection
                self.is_connected = True
                self._warned_disconnected = False
                
                # Set up event handlers
                connection.on(EventType.OPEN, self._on_open)
                connection.on(EventType.MESSAGE, self._on_message)
                connection.on(EventType.ERROR, self._on_error)
                connection.on(EventType.CLOSE, self._on_close)
                
                print("📡 Starting Deepgram listening...")
                await connection.start_listening()
                print("✅ Deepgram listening started")
                
                # Keep connection alive
                while self.is_connected:
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            print(f"❌ Deepgram STT connection lost: {e}")
            import traceback
            traceback.print_exc()
            self.is_connected = False
    
    def _on_open(self, *args):
        """Handle Deepgram connection opened"""
        print("🔗 Deepgram STT connection opened - Ready to receive audio")
    
    def _on_message(self, message):
        """Handle Deepgram transcription message"""
        try:
            transcript = None
            is_final = False
            confidence = 1.0
            
            # Extract transcript
            if hasattr(message, 'channel'):
                channel = message.channel
                if hasattr(channel, 'alternatives') and channel.alternatives:
                    alternative = channel.alternatives[0]
                    transcript = alternative.transcript
                    confidence = getattr(alternative, 'confidence', confidence)
                    is_final = getattr(message, 'is_final', getattr(message, 'speech_final', True))
            
            if not transcript and hasattr(message, 'transcript'):
                transcript = message.transcript
                is_final = True
                confidence = getattr(message, 'confidence', confidence)
            
            # Process final transcripts
            if transcript and transcript.strip() and is_final:
                if transcript == self._last_transcript:
                    return
                
                if confidence < getattr(config, "STT_CONFIDENCE_THRESHOLD", 0.0):
                    return
                
                self._last_transcript = transcript
                print(f"\n✅ 📝 TRANSCRIPT: {transcript}")
                
                asyncio.create_task(self.transcript_queue.put(transcript))
                
        except Exception as e:
            print(f"❌ Deepgram STT message error: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_error(self, error):
        """Handle Deepgram STT error"""
        print(f"❌ Deepgram STT error: {error}")
    
    def _on_close(self, *args):
        """Handle Deepgram connection closed"""
        self.is_connected = False
        if config.VERBOSE:
            print("🔌 Deepgram STT connection closed")
    
    def _on_azure_recognizing(self, evt):
        """Handle Azure Speech interim recognition results (faster, partial transcripts)"""
        try:
            if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
                transcript = evt.result.text.strip()
                if transcript:
                    # Send interim results immediately (like Deepgram does)
                    # These are partial/refined results that update as you speak
                    if self._event_loop and self._event_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self.transcript_queue.put(transcript),
                            self._event_loop
                        )
                    else:
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.run_coroutine_threadsafe(
                                    self.transcript_queue.put(transcript),
                                    loop
                                )
                        except RuntimeError:
                            pass
        except Exception as e:
            if config.DEBUG:
                print(f"⚠️ Azure Speech interim result error: {e}")
    
    def _on_azure_recognized(self, evt):
        """Handle Azure Speech final recognition result (after silence detected)"""
        try:
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                transcript = evt.result.text.strip()
                if transcript and transcript != self._last_transcript:
                    self._last_transcript = transcript
                    print(f"\n✅ 📝 TRANSCRIPT: {transcript}")
                    
                    # Azure callbacks run in a different thread, so we need to use
                    # run_coroutine_threadsafe to schedule the async operation
                    if self._event_loop and self._event_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self.transcript_queue.put(transcript),
                            self._event_loop
                        )
                    else:
                        # Fallback: try to get current loop
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.run_coroutine_threadsafe(
                                    self.transcript_queue.put(transcript),
                                    loop
                                )
                        except RuntimeError:
                            # No event loop available - this shouldn't happen but handle gracefully
                            print("⚠️ No event loop available for Azure Speech callback")
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                if config.VERBOSE:
                    print("⚠️ Azure Speech: No speech could be recognized")
        except Exception as e:
            print(f"❌ Azure Speech recognition error: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_azure_session_started(self, evt):
        """Handle Azure Speech session started"""
        if config.DEBUG:
            print("🔗 Azure Speech session started")
    
    def _on_azure_session_stopped(self, evt):
        """Handle Azure Speech session stopped"""
        if config.DEBUG:
            print("🔌 Azure Speech session stopped")
    
    def _on_azure_canceled(self, evt):
        """Handle Azure Speech cancellation"""
        if evt.reason == speechsdk.CancellationReason.Error:
            print(f"❌ Azure Speech error: {evt.error_details}")
        self.is_connected = False
    
    async def send_audio(self, audio_data: bytes):
        """Send audio data for transcription"""
        if self.provider == "deepgram":
            await self._send_audio_deepgram(audio_data)
        elif self.provider == "azure":
            await self._send_audio_azure(audio_data)
    
    async def _send_audio_deepgram(self, audio_data: bytes):
        """Send audio to Deepgram"""
        if self.connection and self.is_connected:
            try:
                await self.connection.send_media(audio_data)
            except Exception as e:
                print(f"⚠️ Error sending audio to Deepgram: {e}")
        else:
            if not hasattr(self, '_warned_disconnected') or not self._warned_disconnected:
                print(f"⚠️ Cannot send audio: connection={self.connection is not None}, connected={self.is_connected}")
                self._warned_disconnected = True
    
    async def _send_audio_azure(self, audio_data: bytes):
        """Send audio to Azure Speech"""
        if self.push_stream and self.is_connected:
            try:
                # Azure expects audio data as bytes
                self.push_stream.write(audio_data)
            except Exception as e:
                print(f"⚠️ Error sending audio to Azure Speech: {e}")
        else:
            if not hasattr(self, '_warned_disconnected') or not self._warned_disconnected:
                print(f"⚠️ Cannot send audio: Azure Speech not connected")
                self._warned_disconnected = True
    
    async def get_transcript(self) -> str:
        """Get next transcript from queue"""
        return await self.transcript_queue.get()
    
    async def close(self):
        """Close STT connection"""
        self.is_connected = False
        
        if self.provider == "deepgram":
            if self._connection_task:
                self._connection_task.cancel()
                try:
                    await self._connection_task
                except asyncio.CancelledError:
                    pass
        elif self.provider == "azure":
            if self.recognizer:
                try:
                    self.recognizer.stop_continuous_recognition_async()
                except:
                    pass
            if self.push_stream:
                try:
                    self.push_stream.close()
                except:
                    pass
        
        if config.DEBUG:
            print(f"🔌 {self.provider.upper()} STT disconnected")


async def test_stt():
    """Test STT streaming"""
    print(f"Testing {config.STT_PROVIDER.upper()} STT stream...")
    
    stt = STTStream()
    await stt.connect()
    
    print("Say something...")
    
    try:
        transcript = await asyncio.wait_for(
            stt.get_transcript(),
            timeout=10.0
        )
        print(f"✓ Received: {transcript}")
    except asyncio.TimeoutError:
        print("⚠️ No speech detected")
    
    await stt.close()
    print("✓ STT test complete")


if __name__ == "__main__":
    asyncio.run(test_stt())
