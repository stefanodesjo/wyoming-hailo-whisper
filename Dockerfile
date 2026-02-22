FROM ubuntu:22.04

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
ARG DEBIAN_FRONTEND=noninteractive

# Install OS dependencies (rarely changes)
RUN apt-get update && apt-get install -y \
  python3.10 \
  python3-pip \
  python3-venv \
  ffmpeg \
  libportaudio2 \
  wget \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/wyoming_hailo_whisper
WORKDIR /home/wyoming_hailo_whisper

# Install HailoRT packages (rarely changes)
COPY ./hailo-packages/hailort_4.23.0_arm64.deb ./hailo_packages/
COPY ./hailo-packages/hailort-4.23.0-cp310-cp310-linux_aarch64.whl ./hailo_packages/
RUN dpkg --unpack hailo_packages/hailort_4.23.0_arm64.deb

# Install Python dependencies (cached unless requirements.txt changes)
COPY ./requirements.txt ./
COPY ./script/ ./script/
RUN script/setup
ENV PATH="/home/wyoming_hailo_whisper/.venv/bin:$PATH"
ENV HOME="/home/wyoming_hailo_whisper"
RUN pip install ./hailo_packages/hailort-4.23.0-cp310-cp310-linux_aarch64.whl \
  && rm -rf hailo_packages

# Download HEF model resources (cached unless download script changes)
COPY ./wyoming_hailo_whisper/app/download_resources.sh ./wyoming_hailo_whisper/app/
WORKDIR /home/wyoming_hailo_whisper/wyoming_hailo_whisper/app
RUN ./download_resources.sh
WORKDIR /home/wyoming_hailo_whisper

# Copy source code last (changes most frequently)
COPY ./wyoming_hailo_whisper/ ./wyoming_hailo_whisper/
COPY ./setup.py ./setup.cfg ./
RUN script/package

# Entrypoint and permissions
COPY ./docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh \
  && groupadd -r hailo && useradd -r -g hailo -d /home/wyoming_hailo_whisper hailo \
  && chown -R hailo:hailo /home/wyoming_hailo_whisper

USER hailo
EXPOSE 10300
ENTRYPOINT ["./docker-entrypoint.sh"]
