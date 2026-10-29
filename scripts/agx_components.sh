#!/usr/bin/env bash
# On the Jetson AGX Orin (JetPack 6; GitHub issues #4, #6): build the benchmark image and run the component
# measurements inside one container (scripts/agx_components_inside.sh). Nothing is installed on the host; everything
# is read from and written to this repository directory (unpack scripts/jetson_bundle.sh components here first).
# DeepStream 9.1 (MV3DT) needs JetPack 7.2, so on JetPack 6 the pipeline is measured by its parts:
#   detector (PeopleNet INT8: GPU, DLA 0, DLA 1, both DLAs at once), jersey models, court tracker replay, decoding.
#   ./scripts/agx_components.sh            # base image: bauersan/jetson-ndi-yolo if present on the device, else
#                                          # nvcr.io/nvidia/l4t-jetpack with the tag of /etc/nv_tegra_release
#   BASE=<image> ./scripts/agx_components.sh   # another base image (L4T_TAG=r36.4.0 for the JetPack image tag)
#   IMAGE=nvcr.io/nvidia/deepstream:9.1-triton-multiarch ./scripts/agx_components.sh   # dry run on the PC (no DLA)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -z "${IMAGE:-}" ]]; then
  [[ -f /etc/nv_tegra_release ]] || { echo "not a Jetson (no /etc/nv_tegra_release); set IMAGE to test elsewhere"; exit 1; }
  rel=$(sed -n 's/^# R\([0-9]*\) (release), REVISION: \([0-9]*\).*/\1 \2/p' /etc/nv_tegra_release)
  read -r major rev <<< "$rel"
  [[ "$major" == 36 ]] || echo "warning: L4T R$major (expected R36 = JetPack 6); set L4T_TAG if the image tag is wrong"
  if [[ -z "${BASE:-}" ]]; then
    BASE=nvcr.io/nvidia/l4t-jetpack:${L4T_TAG:-r$major.$rev.0}
    docker image inspect bauersan/jetson-ndi-yolo:latest >/dev/null 2>&1 && BASE=bauersan/jetson-ndi-yolo:latest
  fi
  echo "base image: $BASE"
  IMAGE=gym-mv3dt-agx-bench:latest
  docker build -q -t "$IMAGE" --build-arg BASE="$BASE" -f docker/agx_bench.Dockerfile docker >/dev/null
fi
MOUNTS=(-v "$ROOT:/ws")
[[ -f /etc/nv_tegra_release ]] && MOUNTS+=(-v /etc/nv_tegra_release:/etc/nv_tegra_release:ro)
docker run --rm --runtime=nvidia "${MOUNTS[@]}" -w /ws -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  --entrypoint bash "$IMAGE" scripts/agx_components_inside.sh
