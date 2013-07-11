CUDA_ADD_LIBRARY("ndarray${LIB_SUFFIX}" SHARED allocators.cu memory.cu reference.cu)
TARGET_LINK_LIBRARIES("ndarray${LIB_SUFFIX}" ndarray${LIB_SUFFIX})

install(TARGETS "ndarray${LIB_SUFFIX}"
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static )

INSTALL(FILES ndarray.hpp tags.hpp allocators.hpp cuda_general.hpp memory.hpp meta_programming.hpp reference.hpp
	DESTINATION "include/cuv"
)