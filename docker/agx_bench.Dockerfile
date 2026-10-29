# Component benchmarks on a Jetson with JetPack 6 (GitHub issues #4, #6; README "Jetson AGX Orin"). Built on the
# device by scripts/agx_components.sh, on top of an image that is already there: by default the uSpaces image
# (bauersan/jetson-ndi-yolo: DeepStream 7.1 for Jetson, TensorRT, g++, GStreamer with the hardware decoder), else
# NVIDIA's L4T JetPack image. Only what's missing is added, inside the image: trtexec (pinned to the installed
# TensorRT), g++, the GStreamer tools. Nothing is installed on the host.
ARG BASE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE}
ENV DEBIAN_FRONTEND=noninteractive
RUN missing=""; \
    { command -v trtexec || test -x /usr/src/tensorrt/bin/trtexec; } >/dev/null || \
      missing="$missing libnvinfer-bin=$(dpkg-query -W -f='${Version}' libnvinfer10)"; \
    command -v g++ >/dev/null || missing="$missing g++"; \
    command -v gst-launch-1.0 >/dev/null || missing="$missing gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad"; \
    if [ -n "$missing" ]; then apt-get update && apt-get install -y --no-install-recommends $missing && \
      rm -rf /var/lib/apt/lists/*; fi
WORKDIR /ws
