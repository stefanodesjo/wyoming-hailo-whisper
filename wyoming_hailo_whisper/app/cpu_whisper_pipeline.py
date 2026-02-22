"""CPU-based Whisper pipeline using transformers for higher accuracy decoding."""

import logging
import numpy as np
import torch
from queue import Queue, Empty
from threading import Thread
from transformers import WhisperForConditionalGeneration, WhisperProcessor

_LOGGER = logging.getLogger(__name__)


class CpuWhisperPipeline:
    """
    A pipeline that runs Whisper entirely on CPU using the transformers library.
    Trades speed for accuracy: full float32 precision with beam search.
    """

    def __init__(self, variant="base", beam_size=5):
        self.variant = variant
        self.beam_size = beam_size

        model_name = f"openai/whisper-{variant}"
        _LOGGER.info("Loading CPU Whisper model: %s (beam_size=%d)", model_name, beam_size)

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()

        _LOGGER.info("CPU Whisper model loaded (%.0fM parameters)",
                     sum(p.numel() for p in self.model.parameters()) / 1e6)

        self.data_queue = Queue()
        self.results_queue = Queue()
        self.running = True
        self.thread = Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def _inference_loop(self):
        while self.running:
            try:
                audio, language = self.data_queue.get(timeout=1)
            except Empty:
                continue

            try:
                _LOGGER.info("CPU decode: audio length=%.2fs, language=%s",
                             len(audio) / 16000, language)

                inputs = self.processor(
                    audio, sampling_rate=16000, return_tensors="pt"
                )
                input_features = inputs.input_features
                attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long)

                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language, task="transcribe"
                )

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_features,
                        attention_mask=attention_mask,
                        forced_decoder_ids=forced_decoder_ids,
                        num_beams=self.beam_size,
                        max_new_tokens=224,
                    )

                transcription = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

                _LOGGER.info("CPU transcription: '%s'", transcription)
                self.results_queue.put(transcription)
            except Exception:
                _LOGGER.exception("Error during CPU inference")
                self.results_queue.put("")

    def get_model_input_audio_length(self):
        return 30  # transformers handles up to 30s natively

    def send_data(self, data, language="en"):
        self.data_queue.put((data, language))

    def get_transcription(self):
        return self.results_queue.get()

    def stop(self):
        self.running = False
        self.thread.join()
