#define BOOST_TEST_MODULE example
#include <boost/test/included/unit_test.hpp>

#include "ndarray.hpp"

using namespace cuv;

BOOST_AUTO_TEST_SUITE(ndarray_test)

/**
 * @test
 * @brief create ndarray
 */
BOOST_AUTO_TEST_CASE( create_ndarray ) {
    // column_major
    ndarray<float, host_memory_space, column_major> m(extents[2][3][4]);
    BOOST_CHECK_EQUAL(24, m.size());
    BOOST_CHECK_EQUAL(2ul, m.shape(0));
    BOOST_CHECK_EQUAL(3ul, m.shape(1));
    BOOST_CHECK_EQUAL(4ul, m.shape(2));

    BOOST_CHECK_EQUAL(0ul, m.index_of(extents[0][0][0]));
    // column major test
    BOOST_CHECK_EQUAL(1ul, m.index_of(extents[1][0][0]));
    BOOST_CHECK_EQUAL(2ul, m.index_of(extents[0][1][0]));

    // row_major
    ndarray<float, host_memory_space, row_major> n(extents[2][3][4]);
    BOOST_CHECK_EQUAL(24, m.size());
    BOOST_CHECK_EQUAL(2ul, n.shape(0));
    BOOST_CHECK_EQUAL(3ul, n.shape(1));
    BOOST_CHECK_EQUAL(4ul, n.shape(2));

    BOOST_CHECK_EQUAL(0ul, n.index_of(extents[0][0][0]));
    // row major test
    BOOST_CHECK_EQUAL(1ul, n.index_of(extents[0][0][1]));
    BOOST_CHECK_EQUAL(2ul, n.index_of(extents[0][0][2]));
    BOOST_CHECK_EQUAL(4ul, n.index_of(extents[0][1][0]));
}

BOOST_AUTO_TEST_CASE( ndarray_data_access ) {
    ndarray<float, host_memory_space, column_major> m(extents[2][3][4]);
    ndarray<float, host_memory_space, row_major> n(extents[2][3][4]);

    ndarray<float, host_memory_space, column_major> o(extents[2][3][4]);
    ndarray<float, host_memory_space, row_major> p(extents[2][3][4]);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                m(i, j, k) = i * j + k;
                n(i, j, k) = i * j + k;

                o(i, j, k) = i * j + k;
                p(i, j, k) = i * j + k;
            }
        }
    }
    BOOST_CHECK_EQUAL(1*2+3, m(1,2,3));
    BOOST_CHECK_EQUAL(1*2+3, n(1,2,3));
    BOOST_CHECK_EQUAL(1*2+3, o(1,2,3));
    BOOST_CHECK_EQUAL(1*2+3, p(1,2,3));

    BOOST_CHECK_EQUAL(1*2+3-1, --p(1,2,3));
    BOOST_CHECK_EQUAL(1*2+3, p(1,2,3)+=1);
}

BOOST_AUTO_TEST_CASE( ndarray_assignment ) {
    ndarray<float, host_memory_space, column_major> m(extents[2][3][4]);
    ndarray<float, host_memory_space, column_major> n(extents[2][3][4]);

    ndarray<float, host_memory_space, column_major> o(extents[2][3][4]);

    for (int i = 0; i < 2 * 3 * 4; ++i)
        m[i] = i;
    n = m;
    o = m;

    ndarray<float, host_memory_space, column_major> s(n);
    ndarray<float, dev_memory_space, column_major> t(n);

    for (int i = 0; i < 2 * 3 * 4; ++i) {
        BOOST_CHECK_EQUAL(m[i], i);
        BOOST_CHECK_EQUAL(n[i], i);
        BOOST_CHECK_EQUAL(o[i], i);
        BOOST_CHECK_EQUAL(s[i], i);
        BOOST_CHECK_EQUAL(t[i], i);
    }

}

BOOST_AUTO_TEST_CASE( ndarray_zero_copy_assignment ) {
    ndarray<float, host_memory_space> x(extents[4][5][6]);
    for (int i = 0; i < 4 * 5 * 6; i++) {
        x[i] = i;
    }

    ndarray<float, host_memory_space> y = x;

    for (int i = 0; i < 4 * 5 * 6; i++) {
        BOOST_CHECK_EQUAL(x[i], y[i]);
        y[i] = i + 1; // change the copy results in change of original!
        BOOST_CHECK_EQUAL(x[i], y[i]);
    }
}

BOOST_AUTO_TEST_CASE( ndarray_copy ) {
    boost::shared_ptr<allocator> allocator(new pooled_cuda_allocator("ndarray_copy"));
    ndarray<float, host_memory_space> x(extents[4][5][6], allocator);
    for (int i = 0; i < 4 * 5 * 6; i++) {
        x[i] = i;
    }

    ndarray<float, host_memory_space> y = x.copy();
    BOOST_CHECK_NE(x.ptr(), y.ptr());

    for (int i = 0; i < 4; i++) {
        BOOST_CHECK_NE(x[indices[i][index_range()][index_range()]].ptr(),
                y[indices[i][index_range()][index_range()]].ptr());
    }

    ndarray<float, host_memory_space> y2(x.copy());
    BOOST_CHECK_NE(x.ptr(), y2.ptr());

    for (int i = 0; i < 4; i++) {
        BOOST_CHECK_NE(x[indices[i][index_range()][index_range()]].ptr(),
                y2[indices[i][index_range()][index_range()]].ptr());
    }

    for (int i = 0; i < 4 * 5 * 6; i++) {
        BOOST_CHECK_EQUAL(x[i], y[i]);
        y[i]++; // change must not change original!
        BOOST_CHECK_NE(x[i], y[i]);
    }
}

BOOST_AUTO_TEST_CASE( ndarray_out_of_scope_view ) {
    // sub-ndarray views should persist when original ndarray falls out of scope
    ndarray<float, host_memory_space> y;
    {
        ndarray<float, host_memory_space> x(extents[4][5][6]);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 5; ++j)
                for (int k = 0; k < 6; ++k)
                    x(i, j, k) = i + j + k;
        y = x[indices[index_range(1, 3)][index_range()][index_range()]];
    }
    for (int i = 1; i < 3; ++i)
        for (int j = 0; j < 5; ++j)
            for (int k = 0; k < 6; ++k) {
                BOOST_CHECK_EQUAL(y(i-1,j,k), i+j+k);
            }
}

BOOST_AUTO_TEST_CASE( ndarray_slice1col ) {
    ndarray<float, host_memory_space> y;
    ndarray<float, host_memory_space> x(extents[4][5][6]);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 6; ++k) {
                x(i, j, k) = i + j + k;
            }
        }
    }

    // accessing strided memory
    y = x[indices[index_range(0,1)][index_range()][index_range()]];
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 6; ++k) {
                BOOST_CHECK_EQUAL(y(i,j,k), i+j+k);
            }
        }
    }
    x[indices[index_range(0,1)][index_range()][index_range()]] = y.copy();
}

BOOST_AUTO_TEST_CASE( ndarray_slice1row ) {
    ndarray<float, host_memory_space> y;
    ndarray<float, host_memory_space> x(extents[4][5][6]);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 6; ++k) {
                x(i, j, k) = i + j + k;
            }
        }
    }

    // accessing strided memory
    y = x[indices[index_range()][index_range()][index_range(0,1)]];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 1; ++k) {
                BOOST_CHECK_EQUAL(y(i,j,k), i+j+k);
            }
        }
    }
    x[indices[index_range()][index_range()][index_range(0,1)]] = y.copy();
}

BOOST_AUTO_TEST_CASE( ndarray_memcpy2d ) {
    ndarray<float, host_memory_space> y;
    ndarray<float, host_memory_space> x(extents[4][5][6]);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 6; ++k) {
                x(i, j, k) = i + j + k;
            }
        }
    }

    // accessing strided memory
    y = x[indices[index_range()][index_range()][index_range(0, 1)]];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 1; ++k) {
                BOOST_CHECK_EQUAL(y(i,j,k), i+j+k);
            }
        }
    }

    // copying strided memory
    y = y.copy(); // y in R^(4,5,1)
    for (size_t k = 0; k < y.size(); k++) { // avoid fill() dependency in this file (speed up compiling...)
        y[k] = 0.f;
    }

    ndarray_view<float, host_memory_space> m(x, indices[index_range()][index_range()][index_range(0, 1)]);
    m = y;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 1; ++k) {
                if (k != 0) {
                    BOOST_CHECK_EQUAL(x(i,j,k), i+j+k);
                } else {
                    BOOST_CHECK_EQUAL(x(i,j,k), 0.f);
                }
            }
        }
    }
}

template<class V, class M>
void test_resize() {

    // resize with default allocator
    ndarray<V, M, row_major> a(100, 100);
    V* p0 = a.ptr();
    a.resize(100, 100);
    BOOST_CHECK_EQUAL(p0, a.ptr());
    // no size change. pointer must not change

    boost::shared_ptr<pooled_cuda_allocator> allocator(new pooled_cuda_allocator("test_resize"));
    {
        ndarray<V, M, row_major> a(200, 300, allocator);

        BOOST_CHECK_EQUAL(a.shape(0), 200);
        BOOST_CHECK_EQUAL(a.shape(1), 300);

        BOOST_CHECK_EQUAL(allocator->pool_count(M()), 1);
        BOOST_CHECK_EQUAL(allocator->pool_free_count(M()), 0);
        BOOST_CHECK_EQUAL(allocator->pool_size(M()), 200 * 300 * sizeof(V));

        a.resize(100, 100);

        // make sure the memory is freed before new memory is allocated

        BOOST_CHECK_EQUAL(allocator->pool_count(M()), 1);
        BOOST_CHECK_EQUAL(allocator->pool_free_count(M()), 0);
        BOOST_CHECK_EQUAL(allocator->pool_size(M()), 200 * 300 * sizeof(V));

        BOOST_CHECK_EQUAL(a.shape(0), 100);
        BOOST_CHECK_EQUAL(a.shape(1), 100);
    }

    BOOST_CHECK_EQUAL(allocator->pool_count(M()), 1);
    BOOST_CHECK_EQUAL(allocator->pool_free_count(M()), 1);
}

template<class V, class M1, class M2>
void test_pushpull_2d() {
    static const int h = 123, w = 247;
    ndarray<V, M1, row_major> t1;
    ndarray<V, M2, row_major> t2(extents[h][w]);

    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            t2(i, j) = (float) drand48();
        }
    t1 = t2;
    BOOST_CHECK(equal_shape(t1,t2));
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            BOOST_CHECK_EQUAL( (V) t1(i,j), (V) t2(i,j));
        }
    }
}

template<class V, class M1, class M2>
void test_pushpull_3d() {
    static const int d = 3, h = 123, w = 247;
    ndarray<V, M1, row_major> t1;
    ndarray<V, M2, row_major> t2(extents[d][h][w]);

    // ***************************************
    // assignment 2D --> 1D
    // ***************************************
    for (int k = 0; k < d; k++)
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++) {
                t2(k, i, j) = (float) drand48();
            }
    t1 = t2;
    BOOST_CHECK(equal_shape(t1,t2));
    for (int k = 0; k < d; ++k) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                BOOST_CHECK_EQUAL( (V) t1(k,i,j), (V) t2(k,i,j));
            }
        }
    }
}

template<class V, class M>
void test_lowdim_views() {
    static const int d = 3, h = 123, w = 247;
    ndarray<V, M, row_major> t1d(extents[d][h][w]);
    ndarray<V, M, row_major> t2d(extents[d][h][w]);

    for (int k = 0; k < d; k++) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                t2d(k, i, j) = (float) drand48();
            }
        }
    }

    // ***************************************
    // 2D View on 3D ndarray
    // ***************************************
    for (int k = 0; k < d; ++k) {
        ndarray_view<V, M, row_major> view(indices[k][index_range(0, h)][index_range(0, w)], t2d);
        BOOST_CHECK_EQUAL( view.ndim(), 2);
        BOOST_CHECK_EQUAL( view.shape(0), h);
        BOOST_CHECK_EQUAL( view.shape(1), w);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                BOOST_CHECK_EQUAL( (V) view(i,j), (V) t2d(k,i,j));
            }
        }

        // alternative spec
        ndarray_view<V, M, row_major> view_(indices[k][index_range()][index_range() < cuv::index(w)], t2d);
        BOOST_CHECK_EQUAL( view_.ndim(), 2);
        BOOST_CHECK_EQUAL( view_.shape(0), h);
        BOOST_CHECK_EQUAL( view_.shape(1), w);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                BOOST_CHECK_EQUAL( (V) view_(i,j), (V) t2d(k,i,j));
            }
        }
    }

    // ***************************************
    // 1D View on 3D ndarray
    // ***************************************
    for (int k = 0; k < d; ++k) {
        for (int i = 0; i < h; ++i) {
            ndarray_view<V, M, row_major> view(indices[k][i][index_range(0, w)], t2d);
            for (int j = 0; j < w; j++) {
                BOOST_REQUIRE_EQUAL( (V) view(j), (V) t2d(k,i,j));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE( lowdim_views ) {
    test_lowdim_views<float, host_memory_space>();
    test_lowdim_views<float, dev_memory_space>();
}

BOOST_AUTO_TEST_CASE( ndarray_wrapping ) {
    {
        std::vector<float> v_orig(10, 0.f);
        ndarray<float, host_memory_space> v(extents[10], &v_orig[0]);
        ndarray<float, host_memory_space> w(extents[10]);
        for (unsigned int i = 0; i < 10; i++)
            w[i] = 1.f;

        // overwrite the wrapped memory (needs copying)
        v = w;
    }
    {
        std::vector<float> v_orig(10, 0.f);
        ndarray<float, host_memory_space> v(extents[10], &v_orig[0]);
        ndarray<float, dev_memory_space> w(extents[10]);
        for (unsigned int i = 0; i < 10; i++)
            w[i] = 1.f;

        // overwrite the wrapped memory (needs copying)
        v = w;
    }
}

BOOST_AUTO_TEST_CASE( pushpull_nd ) {
    // same memory space, linear container
    test_pushpull_2d<float, host_memory_space, host_memory_space>();
    test_pushpull_2d<float, dev_memory_space, dev_memory_space>();

    // same memory space, 2d container
    test_pushpull_2d<float, host_memory_space, host_memory_space>();
    test_pushpull_2d<float, dev_memory_space, dev_memory_space>();

    // same memory space, 2d vs. 1d
    test_pushpull_2d<float, host_memory_space, host_memory_space>();
    test_pushpull_2d<float, dev_memory_space, dev_memory_space>();
    test_pushpull_2d<float, host_memory_space, host_memory_space>();
    test_pushpull_2d<float, dev_memory_space, dev_memory_space>();
}

BOOST_AUTO_TEST_CASE( ndarray_resize ) {
    test_resize<float, host_memory_space>();
    test_resize<float, dev_memory_space>();
}

BOOST_AUTO_TEST_CASE( create_lm )
{
    unsigned int N = 54;
    {
        linear_memory<float, host_memory_space> v(N);
        BOOST_CHECK_EQUAL(v.size(), N);
        BOOST_CHECK_NE(v.ptr(), (float*)NULL);
        v.dealloc();
        BOOST_CHECK_EQUAL(v.ptr(), (float*)NULL);
    }
    {
        linear_memory<float, dev_memory_space> v(N);
        BOOST_CHECK_EQUAL(v.size(), N);
        BOOST_CHECK_NE(v.ptr(), (float*)NULL);
        v.dealloc();
        BOOST_CHECK_EQUAL(v.ptr(), (float*)NULL);
    }

}

BOOST_AUTO_TEST_CASE( readwrite_lm )
{
    unsigned int N = 54;
    {
        linear_memory<float, host_memory_space> v(N);
        v[1] = 0;
        BOOST_CHECK_EQUAL(v[1], 0);
        v[1] = 1;
        BOOST_CHECK_EQUAL(v[1], 1);
    }
    {
        linear_memory<float, dev_memory_space> v(N);
        v[1] = 0;
        BOOST_CHECK_EQUAL(v[1], 0);
        v[1] = 1;
        BOOST_CHECK_EQUAL(v[1], 1);
    }

}

BOOST_AUTO_TEST_CASE( create_pm )
{
    unsigned int N = 54, M = 97;
    {
        pitched_memory<float, host_memory_space> v(N, M);
        BOOST_CHECK_EQUAL(v.size(), N*M);
        BOOST_CHECK_EQUAL(v.rows(), N);
        BOOST_CHECK_EQUAL(v.cols(), M);
        BOOST_CHECK_GE(v.pitch(), M);
        BOOST_CHECK_NE(v.ptr(), (float*)NULL);
        v.dealloc();
        BOOST_CHECK_EQUAL(v.ptr(), (float*)NULL);
    }
    {
        pitched_memory<float, dev_memory_space> v(N, M);
        BOOST_CHECK_GE(v.size(), N*M);
        BOOST_CHECK_EQUAL(v.rows(), N);
        BOOST_CHECK_EQUAL(v.cols(), M);
        BOOST_CHECK_GE(v.pitch(), M);
        BOOST_CHECK_NE(v.ptr(), (float*)NULL);
        v.dealloc();
        BOOST_CHECK_EQUAL(v.ptr(), (float*)NULL);
    }

}

BOOST_AUTO_TEST_CASE( readwrite_pm )
{
    unsigned int N = 54, M = 97;
    {
        pitched_memory<float, host_memory_space> v(N, M);
        v[1] = 0;
        BOOST_CHECK_EQUAL(v[1], 0);
        v[1] = 1;
        BOOST_CHECK_EQUAL(v[1], 1);
    }
    {
        pitched_memory<float, dev_memory_space> v(N, M);
        v[1] = 0;
        BOOST_CHECK_EQUAL(v[1], 0);
        v[1] = 1;
        BOOST_CHECK_EQUAL(v[1], 1);
    }

    {
        pitched_memory<float, host_memory_space> v(N, M);
        v(3, 4) = 0;
        BOOST_CHECK_EQUAL(v(3,4), 0);
        v(3, 4) = 1;
        BOOST_CHECK_EQUAL(v(3,4), 1);
    }
    {
        pitched_memory<float, dev_memory_space> v(N, M);
        v(3, 4) = 0;
        BOOST_CHECK_EQUAL(v(3,4), 0);
        v(3, 4) = 1;
        BOOST_CHECK_EQUAL(v(3,4), 1);
    }

}

/**
 * @test
 * @brief create dense matrix.
 */BOOST_AUTO_TEST_CASE( create_linear )
{
    unsigned int N = 16, M = 32;
    {
        ndarray<float, dev_memory_space, row_major> m(extents[N][M]);
        BOOST_CHECK_EQUAL(m.size(), N*M);
        BOOST_CHECK_EQUAL(m.shape(0), N);
        BOOST_CHECK_EQUAL(m.shape(1), M);
        BOOST_CHECK_EQUAL(m.stride(0), M);
        BOOST_CHECK_EQUAL(m.stride(1), 1);
    }

    {
        ndarray<float, host_memory_space, row_major> m(extents[N][M]);
        BOOST_CHECK_EQUAL(m.size(), N*M);
        BOOST_CHECK_EQUAL(m.shape(0), N);
        BOOST_CHECK_EQUAL(m.shape(1), M);
        BOOST_CHECK_EQUAL(m.stride(0), M);
        BOOST_CHECK_EQUAL(m.stride(1), 1);
    }

    {
        ndarray<float, dev_memory_space, column_major> m(extents[N][M]);
        BOOST_CHECK_EQUAL(m.size(), N*M);
        BOOST_CHECK_EQUAL(m.shape(0), N);
        BOOST_CHECK_EQUAL(m.shape(1), M);
        BOOST_CHECK_EQUAL(m.stride(0), 1);
        BOOST_CHECK_EQUAL(m.stride(1), N);
    }

    {
        ndarray<float, host_memory_space, column_major> m(extents[N][M]);
        BOOST_CHECK_EQUAL(m.size(), N*M);
        BOOST_CHECK_EQUAL(m.shape(0), N);
        BOOST_CHECK_EQUAL(m.shape(1), M);
        BOOST_CHECK_EQUAL(m.stride(0), 1);
        BOOST_CHECK_EQUAL(m.stride(1), N);
    }
}

/**
 * @test
 * @brief create pitched matrix.
 */BOOST_AUTO_TEST_CASE( create_pitched )
{
    unsigned int N = 16, M = 32;
    {
        ndarray<float, dev_memory_space, row_major> m(extents[N][M], pitched_memory_tag());
        BOOST_CHECK_EQUAL(m.size(), N*M);
        BOOST_CHECK_EQUAL(m.shape(0), N);
        BOOST_CHECK_EQUAL(m.shape(1), M);
        BOOST_CHECK_GE(m.stride(0), M);
        BOOST_CHECK_EQUAL(m.stride(1), 1);
    }

    {
        ndarray<float, host_memory_space, row_major> m(extents[N][M], pitched_memory_tag());
        BOOST_CHECK_EQUAL(m.size(), N*M);
        BOOST_CHECK_EQUAL(m.shape(0), N);
        BOOST_CHECK_EQUAL(m.shape(1), M);
        BOOST_CHECK_GE(m.stride(0), M);
        BOOST_CHECK_EQUAL(m.stride(1), 1);
    }

    {
        ndarray<float, dev_memory_space, column_major> m(extents[N][M], pitched_memory_tag());
        BOOST_CHECK_EQUAL(m.size(), N*M);
        BOOST_CHECK_EQUAL(m.shape(0), N);
        BOOST_CHECK_EQUAL(m.shape(1), M);
        BOOST_CHECK_EQUAL(m.stride(0), 1);
        BOOST_CHECK_GE(m.stride(1), N);
    }

    {
        ndarray<float, host_memory_space, column_major> m(extents[N][M], pitched_memory_tag());
        BOOST_CHECK_EQUAL(m.size(), N*M);
        BOOST_CHECK_EQUAL(m.shape(0), N);
        BOOST_CHECK_EQUAL(m.shape(1), M);
        BOOST_CHECK_EQUAL(m.stride(0), 1);
        BOOST_CHECK_GE(m.stride(1), N);
    }
}

/**
 * @test
 * @brief setting and getting for device and host vectors.
 */BOOST_AUTO_TEST_CASE( set_vector_elements )
{
    static const unsigned int N = 145;
    static const unsigned int M = 97;
    ndarray<float, host_memory_space> v(extents[N][M]);                     // linear memory
    ndarray<float, dev_memory_space> w(extents[N][M], pitched_memory_tag()); // pitched memory
    for (unsigned int i = 0; i < N; i++) {
        v[i] = (float) i / N;
        w[i] = (float) i / N;
    }
    //convert(w,v);
    for (unsigned int i = 0; i < N; i++) {
        BOOST_CHECK_EQUAL(v[i], (float) i/N);
        BOOST_CHECK_EQUAL(w[i], (float) i/N);
    }
}

BOOST_AUTO_TEST_CASE( assign_func )
{
    static const unsigned int N = 145;
    static const unsigned int M = 97;
    ndarray<float, host_memory_space> v(extents[N][M]);
    ndarray<float, host_memory_space> w(extents[N][M]);
    v[5] = 5;
    w[5] = 0;
    w.assign(v);
    BOOST_CHECK_NE(w.ptr(), v.ptr());
    BOOST_CHECK_EQUAL(v[5], 5);
    BOOST_CHECK_EQUAL(w[5], 5);
}

BOOST_AUTO_TEST_CASE( stream_values )
{
    ndarray<float, host_memory_space> v(3, 2);
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = i;
    }
    std::ostringstream o;
    for (size_t i = 0; i < v.size(); i++) {
        o << v[i];
    }
    BOOST_CHECK_EQUAL(o.str(), "012345");
}
BOOST_AUTO_TEST_SUITE_END()
