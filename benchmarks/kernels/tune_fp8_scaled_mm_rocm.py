import time
import random
from collections import OrderedDict
from typing import List, Callable
import os

import torch

import vllm._fp8gemm_C  # noqa: F401
from vllm.platforms import current_platform
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, FlexibleArgumentParser

from fp8_scaled_mm_rocm_kernels import FP8_GEMM_KERNELS as KERNELS

MNKs = []

# M = [1, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 2048, 3072, 3584, 4096, 5120, 6144, 7168, 8192]
# N = [1, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 2048, 3072, 3584, 4096, 5120, 6144, 7168, 8192]
# K = [1024, 1280, 2048, 4096, 7168, 8192, 16384]
# for m in M:
#     for n in N:
#         for k in K:
            # MNKs.append((m, n, k))


# MKNs
#         (32, 1280, 8192),
#         (32, 8192, 1024),
#         (32, 7168, 8192),
#         (32, 8192, 3584),


MNKs += [
    (32, 8192, 1280),
    (8192, 32, 1280),
    (32, 1024, 8192),
    (1024, 32, 8192),
    (32, 8192, 7168),
    (8192, 32, 7168),
    (32, 3584, 8192),
    (3584, 32, 8192),
]

def rand_data(shape, dtype=torch.float16, scale=1):
    return (scale * torch.rand(shape, device="cuda") - 0.3).to(dtype)

@torch.inference_mode()
def main(save_file: str,
         seed: int = 0,
         do_profile: bool = False,
         num_warmup_iters: int = 5,
         num_iters: int = 100) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device("cuda")

    identity = torch.ones(1, dtype=torch.float32)

    def reference_scaled_mm_kernel(xq, wq, x_scale, w_scale, dummy1=None, dummy2=True, out_dtype=torch.float32):
        output = torch._scaled_mm(xq, wq, scale_a=identity, scale_b=identity, out_dtype=out_dtype)
        output = output * x_scale * w_scale.t()
        # output = output.to(dtype=torch.bfloat16)
        return output

    def real_reference_scaled_mm_kernel(xq, wq, x_scale, w_scale, dummy1=None, dummy2=True, out_dtype=torch.float32):
        output = torch._scaled_mm(xq, wq, scale_a=identity, scale_b=identity, out_dtype=torch.float32)
        output = output * x_scale * w_scale.t()
        output = output.to(dtype=torch.bfloat16)
        return output

    def run_cuda_benchmark(kernel: Callable,
                           xqs: List[torch.Tensor],
                           wqs: List[torch.Tensor],
                           x_scales: List[torch.Tensor],
                           w_scales: List[torch.Tensor],
                           num_warmup_iters: int, 
                           num_iters: int,
                           profile: bool = False) -> float:
        # shuffle data
        random.shuffle(xqs)
        random.shuffle(wqs)
        random.shuffle(x_scales)
        random.shuffle(w_scales)
        # warmup
        torch.cuda.synchronize()
        for i in range(num_warmup_iters):
            _ = kernel(xqs[i], wqs[i], x_scales[i], w_scales[i], None, True, out_dtype=torch.bfloat16)

        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        for i in range(num_warmup_iters, num_warmup_iters + num_iters):
            _ = kernel(xqs[i], wqs[i], x_scales[i], w_scales[i], None, True, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        return (end_time - start_time) / num_iters

    benchmark_kernels = KERNELS

    if os.path.exists(save_file):
        os.remove(save_file)
    with open(save_file, "w") as f:
        f.write("dimension, " + ", ".join([k.__name__ for k in benchmark_kernels]) + ", ref_scaled_mm" + "\n")

    # Benchmark
    dim_seq = []
    run_benchmark = run_cuda_benchmark
    x_dt = torch.float8_e4m3fnuz
    w_dt = torch.float8_e4m3fnuz
    scale_dt = torch.float32
    for m, n, k in MNKs:
        print("===============================")
        dim_seq.append((m, n, k))
        xqs = [rand_data((m, k), x_dt, 5) for _ in range(num_iters + num_warmup_iters)]
        wqs = [rand_data((n, k), w_dt, 5) for _ in range(num_iters + num_warmup_iters)]
        x_scales = [rand_data((m, 1), scale_dt, 5) for _ in range(num_iters + num_warmup_iters)]
        w_scales = [rand_data((n, 1), scale_dt, 5) for _ in range(num_iters + num_warmup_iters)]

        latencies = []

        for kernel in benchmark_kernels:
            if do_profile:
                latency = run_benchmark(
                    kernel, xqs, wqs, x_scales, w_scales, 
                    num_warmup_iters=num_warmup_iters, num_iters=1, profile=True)
            else:
                latency = run_benchmark(
                    kernel, xqs, wqs, x_scales, w_scales, 
                    num_warmup_iters=num_warmup_iters, num_iters=num_iters, profile=False)
                

            xq = torch.rand((m, k))
            xq = xq.to(dtype=torch.float8_e4m3fnuz)
            wq = torch.rand((n, k))
            wq = wq.to(dtype=torch.float8_e4m3fnuz)
            x_scale = torch.rand((m,1), dtype=torch.float)
            w_scale = torch.rand((n,1), dtype=torch.float)

            output = kernel(xq, wq, x_scale, w_scale, None, True, out_dtype=torch.bfloat16).t()

            torch.cuda.synchronize()

            try:
                ref_output = torch._scaled_mm(xq, wq.t(), scale_a=identity, scale_b=identity, out_dtype=torch.float32)
                ref_output = ref_output * x_scale * w_scale.t()
                ref_output = ref_output.to(dtype=torch.bfloat16)

                if not torch.allclose(output, ref_output, rtol=1e-2, atol=1e-5):
                    print("!!!!!! [{}, {}, {}] - {} Failed correctness test".format(m, n, k, kernel.__name__))
                    latency = 1000.0
            except RuntimeError:
                pass

            # Not part of the validation
            latencies.append(latency)
            print("[{}, {}, {}] - {}: {:.3f}us".format(m, n, k, kernel.__name__, latency * 1000000))

        # reference
        wqs_t = [wq.t().contiguous() for wq in wqs]
        try:
            latency = run_benchmark(
                reference_scaled_mm_kernel, xqs, wqs_t, x_scales, w_scales, 
                num_warmup_iters=num_warmup_iters, num_iters=1 if do_profile else num_iters, profile=do_profile)
        except RuntimeError:
            latency = -1

        latencies.append(latency)
        print("[{}, {}, {}] - {}: {:.3f}us".format(m, n, k, "reference_scaled_mm_kernel", latency * 1000000))

        with open(save_file, "a") as f:
            f.write("{}_{}_{}, ".format(m, n, k) + ", ".join([str(l*1000000.0) for l in latencies]) + "\n")

    # print(f"Kernel running time: {latency * 1000000:.3f} us")


if __name__ == '__main__':

    def to_torch_dtype(dt):
        if dt == "int8":
            return torch.int8
        if dt == "fp8":
            return torch.float8_e4m3fn
        raise ValueError(f"Unsupported dtype: {dt}")

    parser = FlexibleArgumentParser(
        description="Benchmark the ROCm fp8 scaled mm kernel.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num-warmup-iters", type=int, default=10)
    parser.add_argument("--num-iters",
                        type=int,
                        default=200,
                        help="Number of benchmark iterations. "
                        "If --profile is set, this number is ignored")
    
    parser.add_argument("-o", "--output", type=str, 
                        default="benchmark_fp8_scaled_mm_rocm.csv",
                        help="Path to write the results to")

    args = parser.parse_args()
    print(args)

    main(args.output,
         seed=args.seed,
         do_profile=args.profile,
         num_warmup_iters=args.num_warmup_iters,
         num_iters=args.num_iters)
    