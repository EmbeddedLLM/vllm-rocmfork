#include <vector>
#include <iostream>

template <int _size, typename Th, typename Td>
void compare_value(const Th &h_out, const Th &h_val) {
    if constexpr (_size == 0) {
        assert(abs(h_out - h_val) < PRECISION);
        return;
    }
    for (int i = 0; i < _size; ++i) {
        auto comp = abs(h_out[i] - h_val[i]);
        // std::cout << "h_out:" << h_out[i] << " - h_val:" << h_val[i] << " = " << comp << ", " << PRECISION << std::endl;
        assert(comp < PRECISION);
    }
}

template <int M, int N, typename T>
void print_dumb_vector(const std::vector<T> &vec) {
    assert(vec.size() == M * N);
    for (int i = 0; i < M; ++i) {
        int head = i * N;
        for (int j = 0; j < N; ++j) {
            std::cout << vec[head + j] << ", ";
        }
        std::cout << std::endl;
    }
}

template <int M, int N, int K, 
          typename ThA, typename ThB, typename ThC, 
          typename TdA, typename TdB, typename TdC,
          typename KernelT, typename CheckT, typename CompT>
void launch_mfma_kernels(
        const std::vector<ThA> &hA, const std::vector<ThB> &hB, const std::vector<ThC> &hC,
        KernelT kernel_func, CheckT test_func, CompT compare_func) {
    assert(hA.size() == M * K);
    assert(hB.size() == K * N);
    assert(hC.size() == M * N);
    constexpr int bytesA = M * K * sizeof(ThA);
    constexpr int bytesB = K * N * sizeof(ThB);
    constexpr int bytesC = M * N * sizeof(ThC);

    std::vector<ThC> hD(M * N);

    TdA *dA;
    TdB *dB;
    TdC *dC;
    TdC *dD;
    HIP_CHECK(hipMalloc(&dA, bytesA));
    HIP_CHECK(hipMalloc(&dB, bytesB));
    HIP_CHECK(hipMalloc(&dC, bytesC));
    HIP_CHECK(hipMalloc(&dD, bytesC));

    HIP_CHECK(hipMemcpy(dA, hA.data(), bytesA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), bytesB, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, hC.data(), bytesC, hipMemcpyHostToDevice));

    // const int blocksPerGrid = (size + TEST_BLOCK_SIZE - 1) / TEST_BLOCK_SIZE;
    // const int warps_per_block = (TEST_BLOCK_SIZE + TEST_WARP_SIZE - 1) / TEST_WARP_SIZE;
    // dim3 threadsPerBlock_dim3(TEST_WARP_SIZE, warps_per_block);
    // dim3 blocksPerGrid_dim3(blocksPerGrid);
    kernel_func(dD, dA, dB, dC);

    HIP_CHECK(hipMemcpy(hD.data(), dD, bytesC, hipMemcpyDeviceToHost));

    auto hD_val = test_func(hA, hB, hC);

    compare_func(hD.data(), hD_val.data());

    std::cout << "Kernel test passed!" << std::endl;

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    HIP_CHECK(hipFree(dD));
}

// mfma test
// D = AB + C

// 16x4-thread blocks
template <int M, int N, int K, typename TdA, typename TdB, typename TdC>
__global__ void single_block_gemm_cuda(TdC *dD, TdA *dA, TdB *dB, TdC *dC) {
    using float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    float4 dmn;
    for (int i = 0; i < 4; ++i) {
        const int idx = threadIdx.x + i * N + threadIdx.y * 4 * N;
        dmn[i] = dC[idx];
    }
    int mk = K * threadIdx.x + threadIdx.y;
    int kn = N * threadIdx.y + threadIdx.x;
    float amk = dA[mk];
    float bkn = dB[kn];
    dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
    for (int i = 0; i < 4; ++i) {
        const int idx = threadIdx.x + i * N + threadIdx.y * 4 * N;
        dD[idx] = dmn[i];
    }
}


template <int M, int N, int K, typename TA, typename TB, typename TC>
std::vector<TC> validate_gemm(const std::vector<TA> &A, const std::vector<TB> &B) {
    assert(A.size() == M * K);
    assert(B.size() == K * N);
    std::vector<TC> C(M*N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int index = i*N + j;
            C[index] = 0;
            for (int k = 0; k < K; ++k) {
                C[index] += A[i*K + k] * B[k*N + j];
            }
        }
    }
    return C;
}

template <int M, int N, int K, typename TA, typename TB, typename TC>
std::vector<TC> validate_mfma(const std::vector<TA> &A, const std::vector<TB> &B, const std::vector<TC> &C) {
    assert(A.size() == M * K);
    assert(B.size() == K * N);
    assert(C.size() == M * N);
    std::vector<TC> D = validate_gemm<M, N, K, TA, TB, TC>(A, B);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int index = i*N + j;
            D[index] += C[index];
        }
    }
    return D;
}

template <int M, int N, int K, typename TA, typename TB, typename TC>
void test_mfma(void) {
    std::vector<TA> A(M*K, 0);
    std::vector<TB> B(K*N, 0);
    std::vector<TC> C(M*N, 100);
    
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < M; ++i) {
            A[i*K + k] = i*K + k;
        }
        for (int j = 0; j < N; ++j) {
            B[k*N + j] = k*N + j;
        }
    }

    std::vector<TC> D = validate_mfma<M, N, K>(A, B, C);

    std::cout << "A:-------------------------" << std::endl;
    print_dumb_vector<M, K>(A);
    std::cout << "B:-------------------------" << std::endl;
    print_dumb_vector<K, N>(B);
    std::cout << "C:-------------------------" << std::endl;
    print_dumb_vector<M, N>(C);
    std::cout << "D:-------------------------" << std::endl;
    print_dumb_vector<M, N>(D);
}

template <typename ThA, typename ThB, typename ThC, typename TdA, typename TdB, typename TdC>
void test_mfma_16x16x4(void) {
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 4;
    std::vector<ThA> A(M*K, 0);
    std::vector<ThB> B(K*N, 0);
    std::vector<ThC> C(M*N, 100);
    
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < M; ++i) {
            A[i*K + k] = i*K + k;
        }
        for (int j = 0; j < N; ++j) {
            B[k*N + j] = k*N + j;
        }
    }

    launch_mfma_kernels<16, 16, 4, ThA, ThB, ThC, TdA, TdB, TdC>(
        A, B, C,
        [] (TdC *dD, TdA *dA, TdB *dB, TdC *dC) -> void {
            dim3 blocks(1);
            dim3 threads(16,4);
            hipLaunchKernelGGL((single_block_gemm_cuda<16, 16, 4>), blocks, threads, 0, 0, dD, dA, dB, dC);
        },
        [] (const std::vector<ThA> &hA, const std::vector<ThB> &hB, const std::vector<ThC> &hC) -> std::vector<ThC> {
            return validate_mfma<16, 16, 4>(hA, hB, hC);
        },
        [] (const ThC *hD, const ThC *hD_val) {
            compare_value<M*N, const ThC*, const TdC*>(hD, hD_val);
        }
    );
    std::cout << "mfma_16x16x4 passed" << std::endl;
}
