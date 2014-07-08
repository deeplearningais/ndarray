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
#ifndef __CUV_ALLOCATORS_HPP__
#define __CUV_ALLOCATORS_HPP__

#include <assert.h>
#include <boost/shared_ptr.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <map>
#include <string>

#ifdef DEBUG_POOLING
#include <iostream>
#define CUV_LOG_DEBUG(X) std::cout << X << std::endl;
#else
#define CUV_LOG_DEBUG(X)
#endif

#include "tags.hpp"
#include "meta_programming.hpp"

namespace cuv {

class allocator {

public:

    virtual ~allocator() {
    }

    virtual void alloc(void** ptr, size_t memsize, size_t valueSize, host_memory_space) = 0;

    virtual void alloc(void** ptr, size_t memsize, size_t valueSize, dev_memory_space) = 0;

    virtual void alloc2d(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize,
            host_memory_space) = 0;

    virtual void alloc2d(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize,
            dev_memory_space) = 0;

    virtual void dealloc(void** ptr, host_memory_space) = 0;

    virtual void dealloc(void** ptr, dev_memory_space) = 0;

};

/**
 * Allocator allows allocation, deallocation and copying depending on memory_space_type
 *
 * \ingroup tools
 */
class default_allocator: public allocator {

public:

    virtual ~default_allocator() {
    }

    virtual void alloc(void** ptr, size_t memsize, size_t valueSize, host_memory_space);

    virtual void alloc(void** ptr, size_t memsize, size_t valueSize, dev_memory_space);

    virtual void alloc2d(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize,
            host_memory_space);

    virtual void alloc2d(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize,
            dev_memory_space);

    virtual void dealloc(void** ptr, host_memory_space);

    virtual void dealloc(void** ptr, dev_memory_space);

};

/**
 * @brief allocator that uses cudaMallocHost for allocations in host_memory_space
 */
class cuda_allocator: public default_allocator {

public:

    virtual ~cuda_allocator() {
    }

    virtual void alloc(void** ptr, size_t memsize, size_t valueSize, host_memory_space);

    virtual void alloc(void** ptr, size_t memsize, size_t valueSize, dev_memory_space);

    virtual void alloc2d(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize,
            host_memory_space);

    virtual void alloc2d(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize,
            dev_memory_space);

    virtual void dealloc(void** ptr, host_memory_space);

    virtual void dealloc(void** ptr, dev_memory_space);

}
;

/**
 * @brief allocator that naively pools device and host memory
 */
class pooled_cuda_allocator: public allocator {
private:

    std::string m_name;

    boost::recursive_mutex m_dev_pool_mutex;
    boost::recursive_mutex m_host_pool_mutex;

    // maps pointers to flag: true means memory is available. false means: currently in use
    std::map<void*, bool> m_dev_pool;
    std::map<void*, size_t> m_dev_pool_sizes;

    std::map<void*, bool> m_host_pool;
    std::map<void*, size_t> m_host_pool_sizes;

    default_allocator default_alloc;
    cuda_allocator cuda_alloc;

    pooled_cuda_allocator(const pooled_cuda_allocator& o);
    pooled_cuda_allocator& operator=(const pooled_cuda_allocator& o);

    // for logging
    std::string memtype(host_memory_space) const {
        return "host space";
    }

    // for logging
    std::string memtype(dev_memory_space) const {
        return "dev space";
    }

    template<class memory_space>
    boost::recursive_mutex& get_pool_mutex(memory_space m) const;

    template<class memory_space>
    std::map<void*, bool>& get_pool(memory_space m);

    template<class memory_space>
    const std::map<void*, bool>& get_pool(memory_space m) const;

    template<class memory_space>
    std::map<void*, size_t>& get_pool_sizes(memory_space m);

    template<class memory_space>
    const std::map<void*, size_t>& get_pool_sizes(memory_space m) const;

    template<class memory_space>
    void collect_garbage(memory_space m);

    template<class memory_space>
    void alloc_pooled(void** ptr, size_t memsize, size_t valueSize, memory_space m);

    template<class memory_space>
    void delete_pool(memory_space);

    template<class memory_space>
    void do_dealloc(void** ptr, memory_space m);

public:
    static const size_t MIN_SIZE_HOST = 8192;
    static const size_t MIN_SIZE_DEV = 1;
    explicit pooled_cuda_allocator(const std::string& _name = "");

    virtual ~pooled_cuda_allocator();

    virtual void garbage_collection() {
        collect_garbage(host_memory_space());
        collect_garbage(dev_memory_space());
    }

    virtual void alloc(void** ptr, size_t memsize, size_t valueSize, host_memory_space);

    virtual void alloc(void** ptr, size_t memsize, size_t valueSize, dev_memory_space);

    virtual void alloc2d(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize,
            host_memory_space);

    virtual void alloc2d(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize,
            dev_memory_space);

    virtual void dealloc(void** ptr, host_memory_space);

    virtual void dealloc(void** ptr, dev_memory_space);

    template<class memory_space>
    size_t pool_free_count(memory_space m) const;

    template<class memory_space>
    size_t pool_size(memory_space m) const;

    template<class memory_space>
    size_t pool_count(memory_space m) const;

    size_t pool_free_count() const;

    size_t pool_size() const;

    size_t pool_count() const;

};

}

#endif
