#!/usr/bin/env bash

ARGS=(
    --uri "${WHISPER_URI:-tcp://0.0.0.0:10300}"
    --device "${WHISPER_DEVICE:-hailo8}"
    --variant "${WHISPER_VARIANT:-base}"
    --language "${WHISPER_LANGUAGE:-sv}"
    --beam-size "${WHISPER_BEAM_SIZE:-5}"
)

if [ "${WHISPER_USE_CPU:-false}" = "true" ]; then
    ARGS+=(--use-cpu)
fi

if [ "${WHISPER_ENHANCE_AUDIO:-true}" = "true" ]; then
    ARGS+=(--enhance-audio)
fi

if [ "${WHISPER_DEBUG:-true}" = "true" ]; then
    ARGS+=(--debug)
fi

exec python3 -m wyoming_hailo_whisper "${ARGS[@]}"
