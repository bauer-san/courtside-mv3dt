#!/usr/bin/env bash
# Phase 3: DeepStream 9.1 MV3DT environment.
#   - clone NVIDIA/DeepStream at v9.1.0.2 into external/ (LFS only for the MV3DT app)
#   - fetch Kafka 4.2.0 from archive.apache.org (dlcdn no longer hosts it), SHA-512 verified
#   - run NVIDIA's setup_prerequisites.sh (container pull, datasets, models, parsers, Kafka, venv)
# Host prerequisites (sudo, once): NVIDIA Container Toolkit (scripts/install_nvidia_container_toolkit.sh),
#   net-tools, and /etc/mosquitto/conf.d/mv3dt.conf with "set_tcp_nodelay true".
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$ROOT/external"
DS_TAG=v9.1.0.2
KAFKA=kafka_2.13-4.2.0
APP="$EXT/DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view"
mkdir -p "$EXT"

if [[ ! -d "$EXT/DeepStream/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVIDIA/DeepStream.git "$EXT/DeepStream"
fi
git -C "$EXT/DeepStream" -c advice.detachedHead=false checkout -q "$DS_TAG"
git -C "$EXT/DeepStream" lfs pull --include "src/apps/reference_apps/deepstream-tracker-3d-multi-view/**"

if [[ ! -d "$EXT/$KAFKA" ]]; then
  url="https://archive.apache.org/dist/kafka/4.2.0/$KAFKA.tgz"
  wget -q -O "$EXT/$KAFKA.tgz" "$url"
  expected=$(curl -s "$url.sha512" | sed 's/^.*://' | tr -d ' \n' | tr A-F a-f)
  actual=$(sha512sum "$EXT/$KAFKA.tgz" | cut -d' ' -f1)
  [[ "$expected" == "$actual" ]] || { echo "Kafka SHA-512 mismatch"; exit 1; }
  tar -xzf "$EXT/$KAFKA.tgz" -C "$EXT" && rm "$EXT/$KAFKA.tgz"
fi

cd "$APP"
BASE_DIR="$EXT" ./scripts/setup_prerequisites.sh
