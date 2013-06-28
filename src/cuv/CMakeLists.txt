CUDA_ADD_LIBRARY("ndarray${LIB_SUFFIX}" SHARED allocators.cu memory.cu reference.cu)
TARGET_LINK_LIBRARIES("ndarray${LIB_SUFFIX}" ndarray${LIB_SUFFIX})

install(TARGETS "ndarray${LIB_SUFFIX}"
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static )
