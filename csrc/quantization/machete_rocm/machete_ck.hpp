#pragma once
#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>


#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <torch/torch.h>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_b_preshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"

            template <ck::index_t... Is>
            using S = ck::Sequence<Is...>;

            using F16  = ck::half_t;
            using BF16 = ck::bhalf_t;
            using FP8  = ck::f8_t;
            using F32  = float;

            using Row = ck::tensor_layout::gemm::RowMajor;
            using Col = ck::tensor_layout::gemm::ColumnMajor;

            using A0DataType       = FP8;
            using B0DataType       = FP8;
            using AccDataType      = F32;
            using CShuffleDataType = F32;
            using D0DataType       = F32;
            using D1DataType       = F32;
            using DsDataType       = ck::Tuple<D0DataType, D1DataType>;
            using EDataType        = F16;

            using A0Layout = Row;
            using B0Layout = Col;
            using D0Layout = Row;
            using D1Layout = Col;
            using DsLayout = ck::Tuple<D0Layout, D1Layout>;
            using ELayout  = Row;

            struct MultiplyMultiply
            {
                template <typename E, typename C, typename D0, typename D1>
                __host__ __device__ constexpr void
                operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

                template <>
                __host__ __device__ constexpr void operator()<F16, float, float, float>(F16& e,
                                                                                        const float& c,
                                                                                        const float& d0,
                                                                                        const float& d1) const
                {
                    const float x0_f = c * d0 * d1;

                    e = ck::type_convert<F16>(x0_f);
                }

                template <>
                __host__ __device__ constexpr void operator()<BF16, float, float, float>(BF16& e,
                                                                                        const float& c,
                                                                                        const float& d0,
                                                                                        const float& d1) const
                {
                    const float x0_f = c * d0 * d1;

                    e = ck::type_convert<BF16>(x0_f);
                }

                template <>
                __host__ __device__ constexpr void operator()<ck::half_t, int, float, float>(
                    ck::half_t& e, const int& c, const float& d0, const float& d1) const
                {
                    const float x0_f =
                        ck::type_convert<float>(c) * ck::type_convert<float>(d0) * ck::type_convert<float>(d1);

                    e = ck::type_convert<ck::half_t>(x0_f);
                }

                template <>
                __host__ __device__ constexpr void operator()<ck::bhalf_t, int, float, float>(
                    ck::bhalf_t& e, const int& c, const float& d0, const float& d1) const
                {
                    const float x0_f =
                        ck::type_convert<float>(c) * ck::type_convert<float>(d0) * ck::type_convert<float>(d1);

                    e = ck::type_convert<ck::bhalf_t>(x0_f);
                }
            };

            using PassThrough = ck::tensor_operation::element_wise::PassThrough;

            using AElementOp   = PassThrough;
            using BElementOp   = PassThrough;
            using CDEElementOp = MultiplyMultiply;

            template <
                ck::index_t BLOCK_SIZE,
                ck::index_t MBLOCK,
                ck::index_t NBLOCK,
                ck::index_t KBLOCK,
                ck::index_t WAVE_TILE_M,
                ck::index_t WAVE_TILE_N,
                ck::index_t WAVE_MAP_M,
                ck::index_t WAVE_MAP_N,
                typename ABLOCK_TRANSFER,
                typename BBLOCK_TRANSFER,
                typename CBLOCK_TRANSFER,
                typename CBLOCK_SPV,
                ck::index_t CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
                ck::index_t CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
                ck::BlockGemmPipelineScheduler LOOP_SCHED,
                ck::BlockGemmPipelineVersion PIPELINE_VERSION,
                ck::tensor_operation::device::GemmSpecialization GEMM_SPEC = ck::tensor_operation::device::GemmSpecialization::Default,
                ck::index_t AReadVecLength = 16,
                ck::index_t BReadVecLength = 16>
            using DeviceGemmHelper = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle
            <   Row,       // ALayout
                Col,       // BLayout
                DsLayout,  // DsLayout
                ELayout,   // ELayout
                A0DataType, // AData Type
                B0DataType, // BData Type
                DsDataType, // DsData Type
                EDataType,  // EData Type
                AccDataType, // AccData Type
                CShuffleDataType, // CShuffle Type
                AElementOp,  // A Elementwise Operator
                BElementOp,  // B Elementwise Operator
                CDEElementOp, // CDE Elementwise Operator
                GEMM_SPEC,   // GEMM Specialization
                BLOCK_SIZE,  // Block Size
                MBLOCK,      // MPer Block
                NBLOCK,      // NPer Block
                KBLOCK,      // KPer Block
                16,          // AK1
                16,          // BK1
                WAVE_TILE_M, // Wave Tile : MPer XDL
                WAVE_TILE_N, // Wave Tile : NPer XDL
                WAVE_MAP_M,  // Wave Map : MXdl Per Wave
                WAVE_MAP_N,  // Wave Map : NXdl Per Wave
                ABLOCK_TRANSFER,  // ABlockTransfer ThreadCluster Lengths_K0_M_K1
                S<1, 0, 2>,  // ABlockTransfer ThreadCluster ArrangeOrder
                S<1, 0, 2>,  // ABlockTransfer SrcAccessOrder
                2,           // ABlockTransfer SrcVectorDim
                AReadVecLength, // ABlockTransfer SrcScalar PerVector == VmemReadVec[0]
                16,          // ABlockTransfer DstScalar PerVector_K1
                0,           // ABlockLds AddExtraM
                BBLOCK_TRANSFER, // BBlockTransfer ThreadCluster Lengths_K0_N_K1
                S<1, 0, 2>,  // BBlockTransfer ThreadCluster ArrangeOrder
                S<1, 0, 2>,  // BBlockTransfer SrcAccessOrder
                2,           // BBlockTransfer SrcVectorDim
                BReadVecLength, // BBlockTransfer SrcScalar PerVector == VmemReadVec[1]
                16,          // BBlockTransfer DstScalar PerVector_K1
                0,           // BBlockLds AddExtraN
                CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,  // CShuffle MXd1PerWave PerShuffle
                CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,  // CShuffle NXd1PerWave PerShuffle
                CBLOCK_TRANSFER, // CBlockTransferClusterLengths MBlock_MWaveMPerXd1 NBlock_NWaveNPerXd1
                CBLOCK_SPV,  // CDEShuffleBlockTransferScalarPerVectors OR CBlockTransfer ScalarPerVector NWaveNPerXd1 S<C, D0, D1>
                LOOP_SCHED,  // Loop Scheduler
                PIPELINE_VERSION, // Pipeline Version
                FP8>;        // Data Type


        template <typename DeviceGemmInstance>
        at::Tensor f8f8bf16_pshuffleb_rowwise_impl(
            at::Tensor XQ,
            at::Tensor WQ,
            at::Tensor x_scale,
            at::Tensor w_scale,
            at::Tensor Y,
            ck::index_t KBatch = 1) {
            // Get input information.
            int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
            int N = WQ.size(0);
            int K = WQ.size(1);

            ck::index_t StrideA = K;
            ck::index_t StrideB = K;
            ck::index_t StrideE = N;
            // ck::index_t KBatch = 1;

            // Create gemm launcher and arguments.
            // do GEMM
            auto device_op = DeviceGemmInstance{};

            auto a_element_op = AElementOp{};
            auto b_element_op = BElementOp{};
            auto cde_element_op = CDEElementOp{};

            constexpr ck::index_t NumDTensor = DsDataType::Size();

            constexpr auto I0 = ck::Number<0>{};


            // int NPerXdl = device_op.GetPreShuffleParameters();

            auto invoker = device_op.MakeInvoker();
            auto argument =
                device_op.MakeArgument(
                reinterpret_cast<A0DataType*>(XQ.data_ptr()),
                reinterpret_cast<B0DataType*>(WQ.data_ptr()),
                std::array<const void*, NumDTensor>{
                    reinterpret_cast<D0DataType*>(w_scale.data_ptr()),
                    reinterpret_cast<D1DataType*>(x_scale.data_ptr())},
                reinterpret_cast<EDataType*>(Y.data_ptr()),
                M,
                N,
                K,
                StrideA,
                StrideB,
                std::array<ck::index_t, NumDTensor>{I0, I0},
                StrideE,
                KBatch,
                a_element_op,
                b_element_op,
                cde_element_op);

            auto stream = at::cuda::getCurrentHIPStream().stream();
            invoker.Run(argument, StreamConfig{stream, false});

            if(!device_op.IsSupportedArgument(argument))
            {
                throw std::runtime_error(
                    "wrong! device_gemm with the specified compilation parameters does "
                    "not support this GEMM problem");
            }

            // std::cout << device_op.GetTypeString() << std::endl;

            return Y;
        }
