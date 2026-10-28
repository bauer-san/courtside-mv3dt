"""INT8 calibration cache for the PeopleNet 2.6.3 detector (run inside the DeepStream container).

TensorRT entropy calibration (IInt8EntropyCalibrator2) on frames preprocessed exactly as nvinfer does for this model
(config_pgie_peoplenet.txt: RGB, plain resize to 960 x 544, x / 255): experiments/int8_calib/calib_frames.npy,
400 frames spread over the whole video, all four cameras (no labels involved). Writes the calibration cache that
nvinfer reads through int8-calib-file (network-mode=1).

  pip install --target experiments/int8_calib/pylib tensorrt-cu13-bindings==<container TensorRT version>  (in the container)
  docker run --rm --gpus all -v $PWD:/ws -w /ws -e PYTHONPATH=/ws/experiments/int8_calib/pylib \\
      --entrypoint python3 nvcr.io/nvidia/deepstream:9.1-triton-multiarch \\
      tools/int8_calibrate.py <model.onnx> experiments/int8_calib/calib_frames.npy <out.cache>

YOLOX (GitHub issue #10) takes BGR 0-255 pixels, letterboxed to 640 x 640: the same 400 frames converted by
tools/yolox_person.py calib-frames into experiments/int8_calib/calib_frames_yolox.npy, then this tool with --raw
(no x / 255). The input shape is taken from the frames file.
"""

import ctypes
import sys

import numpy as np
try:
    import tensorrt as trt
except ImportError:          # tensorrt-cu13-bindings alone installs the module as tensorrt_bindings
    import tensorrt_bindings as trt

BATCH = 8
_cudart = ctypes.CDLL("libcudart.so")       # the container's CUDA runtime: device buffer for calibration batches


class DeviceBuffer:
    def __init__(self, nbytes: int):
        self.ptr = ctypes.c_void_p()
        if _cudart.cudaMalloc(ctypes.byref(self.ptr), ctypes.c_size_t(nbytes)) != 0:
            raise RuntimeError("cudaMalloc failed")

    def copy_from(self, x: np.ndarray) -> None:
        if _cudart.cudaMemcpy(self.ptr, x.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(x.nbytes), 1) != 0:
            raise RuntimeError("cudaMemcpy failed")      # 1 = cudaMemcpyHostToDevice


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, frames: np.ndarray, cache: str, scale: float = 1 / 255.0):
        super().__init__()
        self.frames, self.cache, self.i, self.scale = frames, cache, 0, scale
        self.buf = DeviceBuffer(BATCH * frames[0].size * 4)

    def get_batch_size(self):
        return BATCH

    def get_batch(self, names):
        if self.i + BATCH > len(self.frames):
            return None
        x = np.ascontiguousarray(self.frames[self.i:self.i + BATCH].astype(np.float32) * self.scale)
        self.i += BATCH
        self.buf.copy_from(x)
        return [self.buf.ptr.value]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache, "wb") as f:
            f.write(cache)


def main() -> None:
    onnx, frames_path, out = sys.argv[1:4]
    raw = "--raw" in sys.argv[4:]
    frames = np.load(frames_path, mmap_mode="r")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    with open(onnx, "rb") as f:
        if not parser.parse(f.read()):
            raise SystemExit([parser.get_error(i) for i in range(parser.num_errors)])
    inp = net.get_input(0)
    print("input", inp.name, inp.shape)
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.INT8)
    profile = builder.create_optimization_profile()
    shape = (BATCH, *frames.shape[1:])
    profile.set_shape(inp.name, shape, shape, shape)
    cfg.add_optimization_profile(profile)
    cfg.set_calibration_profile(profile)
    cfg.int8_calibrator = Calibrator(frames, out, 1.0 if raw else 1 / 255.0)
    engine = builder.build_serialized_network(net, cfg)
    print("calibration cache written" if engine is not None else "build failed", out)


if __name__ == "__main__":
    main()
