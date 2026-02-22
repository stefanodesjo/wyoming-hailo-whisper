"""Event handler for clients of the server."""
import argparse
import asyncio
import logging

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from wyoming_hailo_whisper.common.postprocessing import clean_transcription
from wyoming_hailo_whisper.common.preprocessing import improve_input_audio, preprocess

_LOGGER = logging.getLogger(__name__)


class HailoWhisperEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model,
        model_lock: asyncio.Lock,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        _LOGGER.debug(cli_args)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.audio = bytes()
        self.audio_converter = AudioChunkConverter(
            rate=16000,
            width=2,
            channels=1,
        )
        self._language = self.cli_args.language or "en"

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            if not self.audio:
                _LOGGER.debug("Receiving audio")

            chunk = AudioChunk.from_event(event)
            chunk = self.audio_converter.convert(chunk)
            self.audio += chunk.audio

            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped")
            text = ""
            sampled_audio = np.frombuffer(self.audio, dtype=np.int16).flatten().astype(np.float32) / 32768.0
            enhance = getattr(self.cli_args, 'enhance_audio', False)
            sampled_audio, start_time = improve_input_audio(sampled_audio, vad=True, enhance=enhance)

            if start_time is None:
                _LOGGER.info("No speech detected in audio")
                self.audio = bytes()
                await self.write_event(Transcript(text="").event())
                return False

            chunk_offset = start_time - 0.2
            if chunk_offset < 0:
                chunk_offset = 0

            use_cpu = getattr(self.cli_args, 'use_cpu', False)

            if use_cpu:
                # CPU mode: send trimmed raw audio directly
                offset_samples = int(chunk_offset * 16000)
                trimmed_audio = sampled_audio[offset_samples:]
                _LOGGER.info("CPU mode: sending %.2fs of audio", len(trimmed_audio) / 16000)

                async with self.model_lock:
                    self.model.send_data(trimmed_audio, language=self._language)
                    transcription = self.model.get_transcription()

                text = transcription.replace("[BLANK_AUDIO]", "").strip()
            else:
                # Hailo mode: generate mel spectrograms and process chunks
                chunk_length = self.model.get_model_input_audio_length()

                mel_spectrograms = preprocess(
                    sampled_audio,
                    True,
                    chunk_length=chunk_length,
                    chunk_offset=chunk_offset
                )

                async with self.model_lock:
                    transcription = ""
                    _LOGGER.info(f"Processing mel spectrograms: {len(mel_spectrograms)}")
                    for mel in mel_spectrograms:
                        _LOGGER.info("Processing mel spectrogram shape: %s, min=%.4f, max=%.4f",
                                     mel.shape, mel.min(), mel.max())
                        self.model.send_data(mel, language=self._language)
                        raw_transcription = self.model.get_transcription()
                        _LOGGER.info(raw_transcription)
                        transcription += clean_transcription(raw_transcription)

                text = transcription.replace("[BLANK_AUDIO]", "").strip()

            _LOGGER.info(text)

            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            # Reset
            self.audio = bytes()
            self._language = self.cli_args.language

            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
                _LOGGER.debug("Language set to %s", transcribe.language)
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
