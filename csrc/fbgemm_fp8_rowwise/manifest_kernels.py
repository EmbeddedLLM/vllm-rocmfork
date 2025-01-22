from dataclasses import dataclass
import os
import argparse
from typing import List
import shutil

SPLIT_K_CHUNKS = [128, 256, 512, 1024, 2048]

TARGET_DIR = "kernel/"

TEMPLATE_FUNC_NAME1 = "f8f8bf16_rowwise_32x32x16_[[[__CONFIGS_DIM__]]]"
TEMPLATE_FUNC_NAME2 = "f8f8bf16_rowwise_16x16x32_[[[__CONFIGS_DIM__]]]"

TEMPLATE_KERNEL = """
#include <cuda_runtime.h>
#include "../fp8_gemm_common_hip.cuh"
#include "../fp8_gemm_common_splitk_hip.cuh"

constexpr uint32_t BLOCKS_X = [[[BLOCKS_X]]];
constexpr uint32_t BLOCKS_Y = [[[BLOCKS_Y]]];
constexpr uint32_t BLOCKS_Z = [[[BLOCKS_Z]]];
constexpr uint32_t MBLOCKS_X = [[[MBLOCKS_X]]];
constexpr uint32_t MBLOCKS_Y = [[[MBLOCKS_Y]]];

at::Tensor [[[TEMPLATE_FUNC_NAME1]]](
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias, // Not implemented
    bool use_fast_accum, // Not implemented
    std::optional<at::ScalarType> out_dtype
) {
    const at::ScalarType _out_dtype = (out_dtype.has_value()) ? out_dtype.value() : at::kBFloat16;
    // Invoke f8f8bf16 rowwise without preallocated output.
    return custom_fp8_32x32x16::f8f8bf16_rowwise_wrapper(
        [_out_dtype](at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale, at::Tensor w_scale, at::Tensor Y, int M, int N, int K) -> void {
            TORCH_CHECK(K % (custom_fp8_32x32x16::BLOCK_K * BLOCKS_Z) == 0, "K must be divisible by 16x");
            LAUNCH_KERNEL_OUTTYPE_32x32x16(_out_dtype, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, M, N, K)
        },
        XQ, WQ, x_scale, w_scale, use_fast_accum, _out_dtype
    );
}

at::Tensor [[[TEMPLATE_FUNC_NAME2]]](
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias, // Not implemented
    bool use_fast_accum, // Not implemented
    std::optional<at::ScalarType> out_dtype
) {
    const at::ScalarType _out_dtype = (out_dtype.has_value()) ? out_dtype.value() : at::kBFloat16;
    // Invoke f8f8bf16 rowwise without preallocated output.
    return custom_fp8_16x16x32::f8f8bf16_rowwise_wrapper(
        [_out_dtype](at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale, at::Tensor w_scale, at::Tensor Y, int M, int N, int K) -> void {
            TORCH_CHECK(K % (custom_fp8_16x16x32::BLOCK_K * BLOCKS_Z) == 0, "K must be divisible by 32x");
            LAUNCH_KERNEL_OUTTYPE_16x16x32(_out_dtype, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, M, N, K)
        },
        XQ, WQ, x_scale, w_scale, use_fast_accum, _out_dtype
    );
}

at::Tensor [[[TEMPLATE_FUNC_NAME2]]]_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias, // Not implemented
    bool use_fast_accum, // Not implemented
    std::optional<at::ScalarType> out_dtype
) {
    const at::ScalarType _out_dtype = (out_dtype.has_value()) ? out_dtype.value() : at::kBFloat16;
    // Invoke f8f8bf16 rowwise without preallocated output.
    return custom_fp8_16x16x32::f8f8bf16_rowwise_wrapper_sk(
        [_out_dtype](at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale, at::Tensor w_scale, at::Tensor Y, int M, int N, int K) -> void {
            TORCH_CHECK(K % (custom_fp8_16x16x32::BLOCK_K * BLOCKS_Z) == 0, "K must be divisible by 32x");
            TORCH_CHECK(custom_fp8_16x16x32::BLOCK_K * BLOCKS_Z <= 128, "K chunk size is too small to split workload along K");
            LAUNCH_KERNEL_OUTTYPE_16x16x32_SK(_out_dtype, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, 128, M, N, K)
        },
        XQ, WQ, x_scale, w_scale, use_fast_accum, _out_dtype
    );
}

at::Tensor [[[TEMPLATE_FUNC_NAME2]]]_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias, // Not implemented
    bool use_fast_accum, // Not implemented
    std::optional<at::ScalarType> out_dtype
) {
    const at::ScalarType _out_dtype = (out_dtype.has_value()) ? out_dtype.value() : at::kBFloat16;
    // Invoke f8f8bf16 rowwise without preallocated output.
    return custom_fp8_16x16x32::f8f8bf16_rowwise_wrapper_sk(
        [_out_dtype](at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale, at::Tensor w_scale, at::Tensor Y, int M, int N, int K) -> void {
            TORCH_CHECK(K % (custom_fp8_16x16x32::BLOCK_K * BLOCKS_Z) == 0, "K must be divisible by 32x");
            TORCH_CHECK(custom_fp8_16x16x32::BLOCK_K * BLOCKS_Z <= 256, "K chunk size is too small to split workload along K");
            LAUNCH_KERNEL_OUTTYPE_16x16x32_SK(_out_dtype, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, 256, M, N, K)
        },
        XQ, WQ, x_scale, w_scale, use_fast_accum, _out_dtype
    );
}

at::Tensor [[[TEMPLATE_FUNC_NAME2]]]_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias, // Not implemented
    bool use_fast_accum, // Not implemented
    std::optional<at::ScalarType> out_dtype
) {
    const at::ScalarType _out_dtype = (out_dtype.has_value()) ? out_dtype.value() : at::kBFloat16;
    // Invoke f8f8bf16 rowwise without preallocated output.
    return custom_fp8_16x16x32::f8f8bf16_rowwise_wrapper_sk(
        [_out_dtype](at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale, at::Tensor w_scale, at::Tensor Y, int M, int N, int K) -> void {
            TORCH_CHECK(K % (custom_fp8_16x16x32::BLOCK_K * BLOCKS_Z) == 0, "K must be divisible by 32x");
            TORCH_CHECK(custom_fp8_16x16x32::BLOCK_K * BLOCKS_Z <= 512, "K chunk size is too small to split workload along K");
            LAUNCH_KERNEL_OUTTYPE_16x16x32_SK(_out_dtype, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, 512, M, N, K)
        },
        XQ, WQ, x_scale, w_scale, use_fast_accum, _out_dtype
    );
}

at::Tensor [[[TEMPLATE_FUNC_NAME2]]]_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias, // Not implemented
    bool use_fast_accum, // Not implemented
    std::optional<at::ScalarType> out_dtype
) {
    const at::ScalarType _out_dtype = (out_dtype.has_value()) ? out_dtype.value() : at::kBFloat16;
    // Invoke f8f8bf16 rowwise without preallocated output.
    return custom_fp8_16x16x32::f8f8bf16_rowwise_wrapper_sk(
        [_out_dtype](at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale, at::Tensor w_scale, at::Tensor Y, int M, int N, int K) -> void {
            TORCH_CHECK(K % (custom_fp8_16x16x32::BLOCK_K * BLOCKS_Z) == 0, "K must be divisible by 32x");
            TORCH_CHECK(custom_fp8_16x16x32::BLOCK_K * BLOCKS_Z <= 1024, "K chunk size is too small to split workload along K");
            LAUNCH_KERNEL_OUTTYPE_16x16x32_SK(_out_dtype, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, 1024, M, N, K)
        },
        XQ, WQ, x_scale, w_scale, use_fast_accum, _out_dtype
    );
}

at::Tensor [[[TEMPLATE_FUNC_NAME2]]]_sk2048(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias, // Not implemented
    bool use_fast_accum, // Not implemented
    std::optional<at::ScalarType> out_dtype
) {
    const at::ScalarType _out_dtype = (out_dtype.has_value()) ? out_dtype.value() : at::kBFloat16;
    // Invoke f8f8bf16 rowwise without preallocated output.
    return custom_fp8_16x16x32::f8f8bf16_rowwise_wrapper_sk(
        [_out_dtype](at::Tensor XQ, at::Tensor WQ, at::Tensor x_scale, at::Tensor w_scale, at::Tensor Y, int M, int N, int K) -> void {
            TORCH_CHECK(K % (custom_fp8_16x16x32::BLOCK_K * BLOCKS_Z) == 0, "K must be divisible by 32x");
            TORCH_CHECK(custom_fp8_16x16x32::BLOCK_K * BLOCKS_Z <= 2048, "K chunk size is too small to split workload along K");
            LAUNCH_KERNEL_OUTTYPE_16x16x32_SK(_out_dtype, BLOCKS_X, BLOCKS_Y, BLOCKS_Z, MBLOCKS_X, MBLOCKS_Y, 2048, M, N, K)
        },
        XQ, WQ, x_scale, w_scale, use_fast_accum, _out_dtype
    );
}
"""

# TEMPLATE_BINDING = """
# #include "core/registration.h"
# #include "ops.h"

# TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, fp8gemm_ops) {
#     // fp8 GEMM ops
#     [[[MANIFESTS]]]
# }

# REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
# """

TEMPLATE_BINDING = """
#pragma once

#define BENCHMARK_KERNELS_DEF_IMPL \\ [[[MANIFESTS]]]
"""

TEMPLATE_BINDING_MANIFEST = """
    fp8gemm_ops.def("[[[TEMPLATE_FUNC_NAME1]]](Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &[[[TEMPLATE_FUNC_NAME1]]]); \\
    fp8gemm_ops.impl("[[[TEMPLATE_FUNC_NAME1]]]", torch::kCUDA, &[[[TEMPLATE_FUNC_NAME1]]]); \\
    fp8gemm_ops.def("[[[TEMPLATE_FUNC_NAME2]]](Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &[[[TEMPLATE_FUNC_NAME2]]]); \\
    fp8gemm_ops.impl("[[[TEMPLATE_FUNC_NAME2]]]", torch::kCUDA, &[[[TEMPLATE_FUNC_NAME2]]]); \\
    fp8gemm_ops.def("[[[TEMPLATE_FUNC_NAME2]]]_sk128(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &[[[TEMPLATE_FUNC_NAME2]]]_sk128); \\
    fp8gemm_ops.impl("[[[TEMPLATE_FUNC_NAME2]]]_sk128", torch::kCUDA, &[[[TEMPLATE_FUNC_NAME2]]]_sk128); \\
    fp8gemm_ops.def("[[[TEMPLATE_FUNC_NAME2]]]_sk256(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &[[[TEMPLATE_FUNC_NAME2]]]_sk256); \\
    fp8gemm_ops.impl("[[[TEMPLATE_FUNC_NAME2]]]_sk256", torch::kCUDA, &[[[TEMPLATE_FUNC_NAME2]]]_sk256); \\
    fp8gemm_ops.def("[[[TEMPLATE_FUNC_NAME2]]]_sk512(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &[[[TEMPLATE_FUNC_NAME2]]]_sk512); \\
    fp8gemm_ops.impl("[[[TEMPLATE_FUNC_NAME2]]]_sk512", torch::kCUDA, &[[[TEMPLATE_FUNC_NAME2]]]_sk512); \\
    fp8gemm_ops.def("[[[TEMPLATE_FUNC_NAME2]]]_sk1024(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &[[[TEMPLATE_FUNC_NAME2]]]_sk1024); \\
    fp8gemm_ops.impl("[[[TEMPLATE_FUNC_NAME2]]]_sk1024", torch::kCUDA, &[[[TEMPLATE_FUNC_NAME2]]]_sk1024); \\
    fp8gemm_ops.def("[[[TEMPLATE_FUNC_NAME2]]]_sk2048(Tensor XQ, Tensor WQ, Tensor x_scale, Tensor w_scale, Tensor? bias, bool use_, ScalarType? out_dtype) -> (Tensor)", &[[[TEMPLATE_FUNC_NAME2]]]_sk2048); \\
    fp8gemm_ops.impl("[[[TEMPLATE_FUNC_NAME2]]]_sk2048", torch::kCUDA, &[[[TEMPLATE_FUNC_NAME2]]]_sk2048); \\
"""

TEMPLATE_HEADER = """
#pragma once

#include <torch/all.h>

[[[MANIFESTS]]]
"""

TEMPLATE_HEADER_MANIFEST = """
at::Tensor [[[TEMPLATE_FUNC_NAME1]]](
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor [[[TEMPLATE_FUNC_NAME2]]](
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor [[[TEMPLATE_FUNC_NAME2]]]_sk128(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor [[[TEMPLATE_FUNC_NAME2]]]_sk256(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor [[[TEMPLATE_FUNC_NAME2]]]_sk512(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor [[[TEMPLATE_FUNC_NAME2]]]_sk1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

at::Tensor [[[TEMPLATE_FUNC_NAME2]]]_sk2048(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::ScalarType> out_dtype);

"""

TEMPLATE_PYTHON_BINDINGS = """
import torch

from vllm import _custom_ops as ops
import vllm._fp8gemm_C  # noqa: F401

FP8_GEMM_KERNELS =[
[[[TEMPLATE_FUNC_NAMES]]]
]
"""

@dataclass
class Config:
    BLOCKS_X: int
    BLOCKS_Y: int
    BLOCKS_Z: int
    MBLOCKS_X: int
    MBLOCKS_Y: int

    def generate_suffix(self) -> str:
        return "{}{}{}{}{}".format(self.BLOCKS_X, self.BLOCKS_Y, self.BLOCKS_Z, self.MBLOCKS_X, self.MBLOCKS_Y)

    def generate_func_names(self, func1_template=TEMPLATE_FUNC_NAME1, func2_template=TEMPLATE_FUNC_NAME2):
        return func1_template.replace("[[[__CONFIGS_DIM__]]]", self.generate_suffix()), func2_template.replace("[[[__CONFIGS_DIM__]]]", self.generate_suffix())
    
    def generate_kernel_string(self, func1_template=TEMPLATE_FUNC_NAME1, func2_template=TEMPLATE_FUNC_NAME2, kernel_template=TEMPLATE_KERNEL):
        func1, func2 = self.generate_func_names(func1_template, func2_template)
        return kernel_template.replace(
            "[[[TEMPLATE_FUNC_NAME1]]]", func1).replace(
            "[[[TEMPLATE_FUNC_NAME2]]]", func2).replace(
            "[[[BLOCKS_X]]]",  str(self.BLOCKS_X)).replace(
            "[[[BLOCKS_Y]]]",  str(self.BLOCKS_Y)).replace(
            "[[[BLOCKS_Z]]]",  str(self.BLOCKS_Z)).replace(
            "[[[MBLOCKS_X]]]", str(self.MBLOCKS_X)).replace(
            "[[[MBLOCKS_Y]]]", str(self.MBLOCKS_Y))
    
    def get_filename(self):
        return "gemm_kernel_{}.cu".format(self.generate_suffix())
    
    def generate_kernel_file(self, dir, func1_template=TEMPLATE_FUNC_NAME1, func2_template=TEMPLATE_FUNC_NAME2, kernel_template=TEMPLATE_KERNEL):
        filename = self.get_filename()
        if dir is not None:
            filename = os.path.join(dir, filename)
        with open(filename, "w") as f:
            f.write(self.generate_kernel_string(func1_template, func2_template, kernel_template))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--show-manifests", action="store_true", help="Show only the manifests without modifying anything")
    parser.add_argument("-f", "--print-functions", action="store_true", help="Show only the functions and then exit")
    parser.add_argument("-s", "--silence", action="store_true", help="Don't print too much")
    args = parser.parse_args()

    configs: List[Config] = []

    for mblocks_x in [1, 2, 4, 8, 16]:
        for mblocks_y in [1, 2, 4, 8, 16]:
            for blocks_x in [1, 2, 4, 8, 16, 32]:
                for blocks_y in [1, 2, 4, 8, 16, 32]:
                    for blocks_z in [1, 2]:
                        if mblocks_x * mblocks_y > 16:
                            continue
                        if mblocks_x * mblocks_y * blocks_x * blocks_y * blocks_z > 64:
                            continue
                        config = Config(
                            BLOCKS_X=blocks_x, 
                            BLOCKS_Y=blocks_y,
                            BLOCKS_Z=blocks_z,
                            MBLOCKS_X=mblocks_x,
                            MBLOCKS_Y=mblocks_y
                        )
                        if config.generate_suffix() in [
                            "111614", "11881", "111641", "11818",
                            "114116", "114161", "113212", "122116",
                            "113221", "121612", "116114", "116212",
                            "116411",
                            "12418", "14218", "12814", "141116",
                            "14414", "14812", "123211", "18118",
                            "18214", "141611", "18412", "161141",
                            "18811", "161221", "161411", "132112",
                            "132211", "161411", "132112", "132211",
                            "212161", "21481", "21841", "211621",
                            "213211", "411161", "41441", "41821",
                            "41281", "411611", "321121", "321211",
                            "81181", "81241", "81421", "81811",
                        ]:
                            continue
                        configs.append(config)
    
    # Any special config goes here


    binding_manifest = ""
    header_manifest = ""
    python_manifest = ""
    python_manifest_sk = ""

    funcs_packed = []
    print("Configuration interfaces:")
    for config in configs:
        func1, func2 = config.generate_func_names()
        print(func1)
        print(func2)
        func2_sks = ["{}_sk{}".format(func2, sk) for sk in SPLIT_K_CHUNKS]
        for func2_sk in func2_sks:
            print(func2_sk)
        funcs_packed.append((func1, func2))
        binding_manifest += TEMPLATE_BINDING_MANIFEST.replace(
                    "[[[TEMPLATE_FUNC_NAME1]]]", func1).replace(
                    "[[[TEMPLATE_FUNC_NAME2]]]", func2) + "\\"
        header_manifest += TEMPLATE_HEADER_MANIFEST.replace(
                    "[[[TEMPLATE_FUNC_NAME1]]]", func1).replace(
                    "[[[TEMPLATE_FUNC_NAME2]]]", func2)
        python_manifest += "    torch.ops._fp8gemm_C.{},\n    torch.ops._fp8gemm_C.{},\n".format(
            func1, func2
        )
        for func2_sk in func2_sks:
            python_manifest_sk += "    torch.ops._fp8gemm_C.{},\n".format(func2_sk)
        
    if args.print_functions:
        exit()
    
    if not args.silence:
        print("-------------------------------------------------------")
        print("Binding manifest:")
        print(binding_manifest)

        print("-------------------------------------------------------")
        print("Kernel files:")
        print("\n".join(a.get_filename() for a in configs))


    if args.show_manifests:
        exit()

    for file in os.listdir(TARGET_DIR):
        file_path = os.path.join(TARGET_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    for config in configs:
        config.generate_kernel_file(TARGET_DIR, TEMPLATE_FUNC_NAME1, TEMPLATE_FUNC_NAME2, TEMPLATE_KERNEL)
    
    binding_text = TEMPLATE_BINDING.replace("[[[MANIFESTS]]]", binding_manifest)
    binding_file = os.path.join(TARGET_DIR, "benchmark_bindings.h")
    with open(binding_file, "w") as f:
        f.write(binding_text)
    
    header_file = os.path.join(TARGET_DIR, "ops.h")
    header_text = TEMPLATE_HEADER.replace("[[[MANIFESTS]]]", header_manifest)
    with open(header_file, "w") as f:
        f.write(header_text)

    python_file = os.path.join(TARGET_DIR, "fp8_scaled_mm_rocm_kernels.py")
    python_text = TEMPLATE_PYTHON_BINDINGS.replace("[[[TEMPLATE_FUNC_NAMES]]]", python_manifest)
    with open(python_file, "w") as f:
        f.write(python_text)

    python_file_sk = os.path.join(TARGET_DIR, "fp8_scaled_mm_rocm_kernels_sk.py")
    python_text_sk = TEMPLATE_PYTHON_BINDINGS.replace("[[[TEMPLATE_FUNC_NAMES]]]", python_manifest_sk)
    with open(python_file_sk, "w") as f:
        f.write(python_text_sk)

    print("-----------------------------------------------------")
    print("Kernel definitions written to {}, {}, {} and {}".format(binding_file, header_file, python_file, python_file_sk))

    

