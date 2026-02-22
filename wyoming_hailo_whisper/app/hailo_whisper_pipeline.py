import logging
import numpy as np
import os
from hailo_platform import (HEF, VDevice, HailoSchedulingAlgorithm, FormatType)
from transformers import AutoTokenizer
from queue import Queue, Empty
from threading import Thread
from wyoming_hailo_whisper.common.postprocessing import apply_repetition_penalty, suppress_special_tokens, WHISPER_EOT_TOKEN

_LOGGER = logging.getLogger(__name__)


class HailoWhisperPipeline:
    """
    A pipeline for running inference using Hailo's Whisper models.
    """

    def __init__(self, encoder_model_path: str, decoder_model_path: str, variant, host="arm64", multi_process_service=False, beam_size=1):
        """
        Initialize the pipeline.

        :param encoder_model_path: Path to the encoder model file.
        :param decoder_model_path: Path to the decoder model file.
        :param variant: Model variant (e.g., "tiny").
        """
        self.encoder_model_path = encoder_model_path
        self.decoder_model_path = decoder_model_path
        self.timeout_ms = 100000000
        self.variant = variant

        self.decoding_sequence_length = None  # set automatically based on HEF details
        self.host = host  # not used in this version
        self.multi_process_service = multi_process_service
        self.beam_size = beam_size

        # Token embedding (ensure float32 for Hailo compatibility)
        self.token_embedding_weight = self._load_token_embedding_weight().astype(np.float32)
        self.onnx_add_input = self._load_onnx_add_input().astype(np.float32)

        self.constant_output_0 = np.array([1])  # Unsqueeze axis
        _LOGGER.info("Token embedding weight shape: %s", self.token_embedding_weight.shape)
        _LOGGER.info("ONNX add input shape: %s", self.onnx_add_input.shape)
        self._load_tokenizer()

        encoder_hef = HEF(self.encoder_model_path)  # load HEF to get input length
        self.input_audio_length = int((encoder_hef.get_input_vstream_infos()[0].shape[1]) / 100)  # in seconds

        self.data_queue = Queue()
        self.results_queue = Queue()
        self.running = True
        self.thread = Thread(target=self._inference_loop)
        self.thread.start()

    def _load_token_embedding_weight(self):
        """
        Load token embedding weights.
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path,
                                 f"decoder_assets/{self.variant}/decoder_tokenization/token_embedding_weight_{self.variant}.npy")
        return np.load(file_path)

    def _load_onnx_add_input(self):
        """
        Load ONNX add input.
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path,
                                 f"decoder_assets/{self.variant}/decoder_tokenization/onnx_add_input_{self.variant}.npy")
        return np.load(file_path)

    def _load_tokenizer(self):
        """
        Load the tokenizer for the specified variant.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(f"openai/whisper-{self.variant}")

    def _tokenization(self, decoder_input_ids, add_embed=True):
        """
        Perform tokenization operations.

        :param decoder_input_ids: Input token IDs for the decoder.
        :param add_embed: Whether to add positional embedding bias.
        :return: Contiguous float32 array ready for Hailo set_buffer.
        """
        # embedding lookup
        gather_output = self.token_embedding_weight[decoder_input_ids]

        if add_embed:
            add_output = gather_output + self.onnx_add_input
            unsqueeze_output = np.expand_dims(add_output, axis=int(self.constant_output_0[0]))
            transpose_output = np.transpose(unsqueeze_output, (0, 2, 1, 3))
            return np.ascontiguousarray(transpose_output, dtype=np.float32)
        else:
            unsqueeze_output = np.expand_dims(gather_output, axis=0)
            return np.ascontiguousarray(unsqueeze_output, dtype=np.float32)

    def _inference_loop(self):
        """
        Main inference loop for processing input data and generating transcriptions.
        """
        try:
            self._run_inference()
        except Exception:
            _LOGGER.exception("Inference loop crashed")

    def _run_inference(self):
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        if self.multi_process_service:
            params.multi_process_service = True
            params.group_id = "SHARED"

        # get output info
        decoder_hef = HEF(self.decoder_model_path)
        sorted_output_names = decoder_hef.get_sorted_output_names()
        decoder_model_name = decoder_hef.get_network_group_names()[0]
        self.decoding_sequence_length = decoder_hef.get_output_vstream_infos()[0].shape[1]
        _LOGGER.info("Decoder sequence length: %d", self.decoding_sequence_length)
        _LOGGER.info("Encoder input audio length: %ds", self.input_audio_length)
        _LOGGER.info("Decoder output names: %s", sorted_output_names)

        with VDevice(params) as vdevice:
            encoder_infer_model = vdevice.create_infer_model(self.encoder_model_path)
            decoder_infer_model = vdevice.create_infer_model(self.decoder_model_path)
            encoder_infer_model.input().set_format_type(FormatType.FLOAT32)
            encoder_infer_model.output().set_format_type(FormatType.FLOAT32)
            decoder_infer_model.input(f"{decoder_model_name}/input_layer1").set_format_type(FormatType.FLOAT32)
            decoder_infer_model.input(f"{decoder_model_name}/input_layer2").set_format_type(FormatType.FLOAT32)

            # model's outputs will be concatenated on the host
            for output_name in sorted_output_names:
                decoder_infer_model.output(output_name).set_format_type(FormatType.FLOAT32)

            useful_outputs = []
            for output_name in sorted_output_names:
                if "conv" in output_name:
                    useful_outputs.append(output_name)
            if not useful_outputs:
                _LOGGER.warning("No 'conv' outputs found, using all outputs: %s", sorted_output_names)
                useful_outputs = sorted_output_names
            _LOGGER.info("Useful (conv) outputs: %s", useful_outputs)

            _LOGGER.info("Encoder input shape: %s", encoder_infer_model.input().shape)
            _LOGGER.info("Encoder output shape: %s", encoder_infer_model.output().shape)
            _LOGGER.info("Decoder input_layer1 shape: %s", decoder_infer_model.input(f"{decoder_model_name}/input_layer1").shape)
            _LOGGER.info("Decoder input_layer2 shape: %s", decoder_infer_model.input(f"{decoder_model_name}/input_layer2").shape)
            for oname in sorted_output_names:
                _LOGGER.info("Decoder output '%s' shape: %s", oname, decoder_infer_model.output(oname).shape)

            with encoder_infer_model.configure() as encoder_configured_infer_model:
                with decoder_infer_model.configure() as decoder_configured_infer_model:
                    encoder_bindings = encoder_configured_infer_model.create_bindings()
                    decoder_bindings = decoder_configured_infer_model.create_bindings()

                    while self.running:
                        try:
                            # Wait for new data with a timeout to allow clean exit
                            input_mel, language = self.data_queue.get(timeout=1)
                        except Empty:
                            continue

                        try:
                            input_mel = np.ascontiguousarray(input_mel, dtype=np.float32)
                            _LOGGER.info("Input mel shape: %s", input_mel.shape)
                            encoder_bindings.input().set_buffer(input_mel)
                            buffer = np.zeros(encoder_infer_model.output().shape).astype(np.float32)
                            encoder_bindings.output().set_buffer(buffer)

                            encoder_configured_infer_model.run([encoder_bindings], self.timeout_ms)
                            encoded_features = encoder_bindings.output().get_buffer()
                            _LOGGER.info("Encoded features shape: %s", encoded_features.shape)
                            _LOGGER.info("Encoded features stats: min=%.6f, max=%.6f, mean=%.6f, std=%.6f",
                                         encoded_features.min(), encoded_features.max(),
                                         encoded_features.mean(), encoded_features.std())

                            # Build forced Whisper prefix: SOT, language, transcribe, notimestamps
                            sot_token = 50258
                            language_token = self.tokenizer.convert_tokens_to_ids(f"<|{language}|>")
                            if language_token is None or language_token == self.tokenizer.unk_token_id:
                                _LOGGER.warning("Unknown language '%s', falling back to English", language)
                                language_token = 50259  # <|en|>
                            transcribe_token = 50359
                            notimestamps_token = 50363

                            prefix = [sot_token, language_token, transcribe_token, notimestamps_token]
                            _LOGGER.info("Forced prefix: %s (language=%s)", prefix, language)

                            # Helper: run one decoder step and return raw logits at position
                            def run_decoder_step(beam_ids, pos):
                                tok_emb = self._tokenization(beam_ids, add_embed=True)
                                decoder_bindings.input(f"{decoder_model_name}/input_layer1").set_buffer(encoded_features)
                                decoder_bindings.input(f"{decoder_model_name}/input_layer2").set_buffer(tok_emb)
                                bufs = [np.zeros(decoder_infer_model.output(n).shape, dtype=np.float32) for n in sorted_output_names]
                                for n, b in zip(sorted_output_names, bufs):
                                    decoder_bindings.output(n).set_buffer(b)
                                decoder_configured_infer_model.run([decoder_bindings], self.timeout_ms)
                                return np.concatenate(
                                    [decoder_bindings.output(n).get_buffer() for n in useful_outputs], axis=2
                                )[:, pos]

                            beam_size = self.beam_size
                            length_penalty_alpha = 0.6
                            first_decode_pos = len(prefix) - 1

                            # Initialize beams
                            initial_ids = np.zeros((1, self.decoding_sequence_length), dtype=np.int64)
                            for j, tok in enumerate(prefix):
                                initial_ids[0][j] = tok

                            active_beams = [{
                                'ids': initial_ids,
                                'tokens': list(prefix[1:]),
                                'content': [],
                                'score': 0.0,
                            }]
                            finished_beams = []
                            _LOGGER.info("Decoding with beam_size=%d", beam_size)

                            # Beam search decoding loop
                            for i in range(first_decode_pos, self.decoding_sequence_length - 1):
                                all_candidates = []

                                for beam in active_beams:
                                    raw_logits = run_decoder_step(beam['ids'], i)

                                    content_count = i - first_decode_pos
                                    logits = apply_repetition_penalty(raw_logits, beam['content'], penalty=1.5)
                                    logits = suppress_special_tokens(logits, allow_eot=content_count >= 1)

                                    # Log softmax for beam scoring
                                    max_l = np.max(logits)
                                    log_probs = logits - max_l - np.log(np.sum(np.exp(logits - max_l)))

                                    # Top candidates per beam
                                    top_k = beam_size * 2
                                    top_indices = np.argsort(log_probs)[-top_k:][::-1]

                                    if i == first_decode_pos and beam is active_beams[0]:
                                        top5 = top_indices[:5]
                                        top5_tokens = [self.tokenizer.decode([idx]) for idx in top5]
                                        _LOGGER.info("Step %d: top5=%s ids=%s scores=%.2f..%.2f",
                                                     i, top5_tokens, top5.tolist(),
                                                     float(log_probs[top5[0]]), float(log_probs[top5[-1]]))

                                    for idx_np in top_indices:
                                        idx = int(idx_np)
                                        new_ids = beam['ids'].copy()
                                        new_ids[0][i + 1] = idx
                                        new_beam = {
                                            'ids': new_ids,
                                            'tokens': beam['tokens'] + [idx],
                                            'content': beam['content'] + [idx],
                                            'score': beam['score'] + float(log_probs[idx]),
                                        }
                                        if idx == WHISPER_EOT_TOKEN:
                                            finished_beams.append(new_beam)
                                        else:
                                            all_candidates.append(new_beam)

                                # Keep top beam_size active beams by score
                                all_candidates.sort(key=lambda b: b['score'], reverse=True)
                                active_beams = all_candidates[:beam_size]

                                if not active_beams:
                                    break

                                # Early stop if enough finished beams collected
                                if len(finished_beams) >= beam_size:
                                    break

                            # Select best beam with length-normalized score
                            all_beams = finished_beams + active_beams
                            def beam_score(b):
                                length = max(len(b['content']), 1)
                                return b['score'] / (length ** length_penalty_alpha)

                            best = max(all_beams, key=beam_score)
                            generated_tokens = best['tokens']

                            _LOGGER.info("Beam search: %d finished, %d active, best_score=%.2f, length=%d",
                                         len(finished_beams), len(active_beams),
                                         beam_score(best), len(best['content']))
                            _LOGGER.info("Generated tokens: %s", generated_tokens)
                            transcription = self.tokenizer.decode(
                                generated_tokens, skip_special_tokens=True
                            )
                            _LOGGER.info("Transcription: '%s'", transcription)
                            self.results_queue.put(transcription)
                        except Exception:
                            _LOGGER.exception("Error during inference")
                            self.results_queue.put("")

    def get_model_input_audio_length(self):
        """
        Get the expected input audio length for the encoder.

        :return: Input audio length in seconds.
        """
        return self.input_audio_length

    def send_data(self, data, language="en"):
        """
        Send new data to the queue.

        :param data: Input data to process.
        :param language: Language code for transcription (e.g., "en", "sv").
        """
        self.data_queue.put((data, language))

    def get_transcription(self):
        """
        Retrieve the next transcription result.

        :return: Transcription result.
        """
        return self.results_queue.get()

    def stop(self):
        """
        Stop the processing loop.
        """
        self.running = False
        self.thread.join()

