cmake_minimum_required( VERSION 2.6 FATAL_ERROR )

FIND_PACKAGE(CUDA)

if ( NOT CUDA_ARCHITECTURE )
	SET( CUDA_ARCHITECTURE -gencode;arch=compute_13,code=sm_13;-gencode;arch=compute_20,code=sm_20;-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35 )
endif()

# ---------- Find Boost Headers/Libraries -----------------------
SET (Boost_FIND_REQUIRED TRUE)
SET (Boost_FIND_QUIETLY TRUE)
SET (Boost_USE_MULTITHREADED TRUE)
SET (Boost_USE_STATIC_LIBS FALSE)
SET (Boost_ADDITIONAL_VERSIONS "1.39" "1.39.0" "1.40" "1.42" "1.43" "1.44" "1.45" "1.46" "1.47" "1.48")
FIND_PACKAGE( Boost 1.37 COMPONENTS unit_test_framework serialization REQUIRED )
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

FIND_PATH(THRUST_PATH thrust/device_vector.h /usr/include /usr/local/include ${CUDA_INCLUDE_DIRS} "$ENV{THRUST_ROOT}")
IF(NOT THRUST_PATH)
	MESSAGE(FATAL_ERROR "Could not find the thrust library. Please install in standard locations or set THRUST_ROOT environment variable.")
ENDIF(NOT THRUST_PATH)

SET(CUDA_ARCHITECTURE "" CACHE STRING "The CUDA architecture to compile for, i.e. -arch=sm_20")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CUDA_ARCHITECTURE}")
MESSAGE(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

CUDA_INCLUDE_DIRECTORIES( ${THRUST_PATH}                                )
INCLUDE_DIRECTORIES(      ${THRUST_PATH}                                )

add_subdirectory(cuv)
add_subdirectory(tests)

INSTALL(DIRECTORY cuv${LIB_SUFFIX}/
	DESTINATION "include/cuv"
	FILES_MATCHING PATTERN "*.hpp"
)
