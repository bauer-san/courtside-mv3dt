# Build inside the DeepStream container (see scripts/build_court_app.sh)
NVDS_VERSION?=9.1
CUDA_VER?=13.2
CXX:=g++
CFLAGS:=-Wall -std=c++17 -shared -fPIC -I/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/includes \
        -I/usr/local/cuda-$(CUDA_VER)/include
LIB:=libnvdsinfer_jersey.so

all: $(LIB)

$(LIB): nvdsinfer_jersey_parsers.cpp
	$(CXX) -o $@ $^ $(CFLAGS)

clean:
	rm -f $(LIB)
