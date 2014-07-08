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
#include "reference.hpp"

#include <thrust/device_ptr.h>

namespace cuv {
namespace detail {

template<class value_type>
void entry_set(value_type* ptr, size_t idx, value_type val, host_memory_space) {
    ptr[idx] = val;
}

template<class value_type>
value_type entry_get(const value_type* ptr, size_t idx, host_memory_space) {
    return ptr[idx];
}

template<class value_type>
void entry_set(value_type* ptr, size_t idx, value_type val, dev_memory_space) {
    thrust::device_ptr<value_type> dev_ptr(ptr);
    dev_ptr[idx] = val;
}

template<class value_type>
value_type entry_get(const value_type* ptr, size_t idx, dev_memory_space) {
    const thrust::device_ptr<const value_type> dev_ptr(ptr);
    return static_cast<value_type>(*(dev_ptr + idx));
}

}
}

template<class T, class M>
std::ostream& operator<<(std::ostream& os, const cuv::reference<T, M>& reference) {
    os << static_cast<T>(reference);
    return os;
}


#define CUV_REFERENCE_INST(TYPE) \
    template void cuv::detail::entry_set(TYPE*, size_t, TYPE, cuv::host_memory_space); \
    template void cuv::detail::entry_set(TYPE*, size_t, TYPE, cuv::dev_memory_space); \
    template TYPE cuv::detail::entry_get(const TYPE*, size_t, cuv::host_memory_space); \
    template TYPE cuv::detail::entry_get(const TYPE*, size_t, cuv::dev_memory_space); \
    template std::ostream& operator<<(std::ostream& os, const cuv::reference<TYPE, cuv::host_memory_space>& reference); \
    template std::ostream& operator<<(std::ostream& os, const cuv::reference<TYPE, cuv::dev_memory_space>& reference);

CUV_REFERENCE_INST(signed char);
CUV_REFERENCE_INST(unsigned char);
CUV_REFERENCE_INST(short);
CUV_REFERENCE_INST(unsigned short);
CUV_REFERENCE_INST(int);
CUV_REFERENCE_INST(unsigned int);
CUV_REFERENCE_INST(float);
CUV_REFERENCE_INST(double);
