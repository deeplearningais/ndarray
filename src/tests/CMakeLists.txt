#######################################################################################
# The MIT License

# Copyright (c) 2014       Hannes Schulz, University of Bonn  <schulz@ais.uni-bonn.de>
# Copyright (c) 2013       Benedikt Waldvogel, University of Bonn <mail@bwaldvogel.de>
# Copyright (c) 2008-2009  Sebastian Nowozin                       <nowozin@gmail.com>

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
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/cuv)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/)

FIND_PACKAGE(TBB      REQUIRED)

CUDA_INCLUDE_DIRECTORIES( ${TBB_INCLUDE_DIRS}  )
INCLUDE_DIRECTORIES(      ${TBB_INCLUDE_DIRS}  )

SET (TEST_LINK_LIBS ${Boost_LIBRARIES} ${TBB_LIBRARIES} ndarray${LIB_SUFFIX})

CUDA_ADD_EXECUTABLE(allocators_test allocators_test.cpp)
TARGET_LINK_LIBRARIES(allocators_test ${TEST_LINK_LIBS})

CUDA_ADD_EXECUTABLE(ndarray_test ndarray_test.cpp)
TARGET_LINK_LIBRARIES(ndarray_test ${TEST_LINK_LIBS})

ADD_TEST(allocators_test "${CMAKE_BINARY_DIR}/src/tests/allocators_test")
ADD_TEST(ndarray_test "${CMAKE_BINARY_DIR}/src/tests/ndarray_test")
