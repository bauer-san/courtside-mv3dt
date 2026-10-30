# Build inside the DeepStream container (see scripts/build_court_app.sh)
NVDS_VERSION?=9.1
CUDA_VER?=13.2
DS:=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)
CUDA:=/usr/local/cuda-$(CUDA_VER)
INC:=-I$(DS)/sources/includes -I$(DS)/sources/gst-plugins/gst-nvdspreprocess/include -I$(CUDA)/include \
     $(shell pkg-config --cflags gstreamer-1.0)
LIB:=libnvds_upright_preprocess.so

all: $(LIB)

upright_warp.o: upright_warp.cu upright_warp.h
	$(CUDA)/bin/nvcc -O2 -Xcompiler -fPIC -c -o $@ upright_warp.cu -I$(CUDA)/include

$(LIB): upright_preprocess.cpp upright_warp.o upright_meta.h upright_warp.h
	g++ -Wall -O2 -std=c++17 -shared -fPIC -o $@ upright_preprocess.cpp upright_warp.o $(INC) -L$(CUDA)/lib64 -lcudart

clean:
	rm -f $(LIB) *.o
