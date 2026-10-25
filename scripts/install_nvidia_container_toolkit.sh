#!/usr/bin/env bash
# One-time host setup (needs sudo; run manually). Approved in Phase 0 as option A:
# install NVIDIA Container Toolkit and register the nvidia runtime with Docker.
# Creates /etc/docker/daemon.json and restarts Docker (stops running containers).
set -euo pipefail

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker run --rm --gpus all ubuntu nvidia-smi
docker rmi ubuntu >/dev/null 2>&1 || true
echo "OK: GPU visible inside Docker"
