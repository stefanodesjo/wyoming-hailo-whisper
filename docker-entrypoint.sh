#!/usr/bin/env bash

python3 -m wyoming_hailo_whisper \
    --uri 'tcp://0.0.0.0:10300' \
    --device 'hailo8' \
    --variant 'base' \
    --language 'sv' \
    --beam-size 3 \
    --debug
