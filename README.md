ndarray Documentation
=====================

Summary
-------

ndarray is a C++ template library for n-dimensional arrays on CPU and GPU using NVIDIA CUDA™. It is extracted from the [CUV library][cuv].

Features
--------

### Supported Platforms ###

  - This library was only tested on Ubuntu Karmic, Lucid and Maverick. It uses
    mostly standard components and should run without major
    modification on any current linux system.

### Supported GPUs ###

  - By default, code is generated for the lowest compute architecture. We
    recommend you change this to match your hardware. Using ccmake you can set
    the build variable "CUDA_ARCHITECTURE" for example to -arch=compute_20
  - All GT 9800 and GTX 280 and above
  - GT 9200 without convolutions. It might need some minor modifications to
    make the rest work. If you want to use that card and have problems, just
    get in contact.
  - On 8800GTS, random numbers and convolutions wont work.


Installation
------------

### Dependencies ###

To build the C++ lib, you will need:

  - cmake (and cmake-curses-gui for easy configuration)
  - libboost-dev >= 1.37
  - NVIDIA CUDA (tm), including SDK. We support versions 3.X, 4.X and 5.X
  - [thrust library][thrust] - included in CUDA since 4.0


### Building a debug version ###

```bash
mkdir -p build/debug
cd build/debug
cmake -DCMAKE_BUILD_TYPE=Release ../../
ccmake .             # adjust paths to your system (cuda, thrust, ...)!
make -j
ctest                # run tests to see if it went well
sudo make install
```

### Building a release version ###

```bash
mkdir -p build/release
cd build/release
cmake -DCMAKE_BUILD_TYPE=Release ../../
ccmake .             # adjust paths to your system (cuda, thrust, ...)!
make -j
ctest                # run tests to see if it went well
sudo make install
```

Usage
-----

### Example ###

```c++
#include <cuv/ndarray.hpp>

int main(void) {

	// allocate a 10×20 array of ints in row-major order on host (CPU)
	cuv::ndarray<int, cuv::host_memory_space> a_host(10, 20);

	assert(a_host.ndim() == 2);        // a_host is a two-dimensional array
	assert(a_host.size() == 10 * 20);

	// initialize the array
	int x = 0;
	for(int i=0; i < a_host.shape(0); i++) { // shape(0) == 10
		for(int j=0; j < a_host.shape(1); j++) { // shape(1) == 20
			a_host(i, j) = x++;
		}
	}

	// reshape to a 20×10 array
	a_host.reshape(20, 10);
	assert(a_host.shape(0) == 20);
	assert(a_host.shape(1) == 10);

	// copy the array to the GPU
	cuv::ndarray<int, cuv::dev_memory_space> a_device = a_host;

	// get the pointer to global device memory
	int* device_ptr = a_device.ptr();

	return 0;
}
```

[thrust]: http://code.google.com/p/thrust/
[cuv]: https://github.com/deeplearningais/CUV
