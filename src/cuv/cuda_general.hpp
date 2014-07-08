#if 0
#######################################################################################
# The MIT License

# Copyright (c) 2013       Benedikt Waldvogel, University of Bonn <mail@bwaldvogel.de>
# Copyright (c) 2012-2014  Hannes Schulz, University of Bonn  <schulz@ais.uni-bonn.de>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######################################################################################
#endif
#ifndef __CUV_GENERAL_HPP__
#define __CUV_GENERAL_HPP__

#include <cuda_runtime_api.h>
#include <stdexcept>

#ifndef CUDA_TEST_DEVICE
#  define CUDA_TEST_DEVICE 0
#endif

namespace cuv {

/** check whether cuda thinks there was an error and fail with msg, if this is the case
 * @ingroup tools
 */
static inline void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// use this macro to make sure no error occurs when cuda functions are called
#ifdef NDEBUG
#  define cuvSafeCall(X)  \
      if(strcmp(#X,"cudaThreadSynchronize()")!=0){ X; cuv::checkCudaError(#X); }
#else
#  define cuvSafeCall(X) X; cuv::checkCudaError(#X);
#endif

}

#endif
