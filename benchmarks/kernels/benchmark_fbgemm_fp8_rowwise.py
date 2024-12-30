import argparse
import copy
import itertools
import math
import os
import pickle as pkl
import time
from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, List, Optional, Tuple

import pandas as pd
import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_rows, quantize_weights)
from vllm.scalar_type import ScalarType, scalar_types
from vllm.utils import FlexibleArgumentParser

import vllm._fp8gemm_C # noqa: F401
from vllm import _custom_ops as ops
from vllm.platforms import current_platform

assert current_platform.is_rocm()

DEFAULT_MODELS = ["meta-llama/Llama-3-8b", "meta-llama/Llama-2-70b-hf"]
DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512, 1024]
DEFAULT_TP_SIZES = [1]

NVTX_PROFILE = os.environ.get("NVTX_PROFILE", False)

if NVTX_PROFILE:
    import nvtx


def terse_type_name(dt):
    return {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.int8: "int8",
        torch.float8_e4m3fn: "fp8",
        torch.float8_e4m3fnuz: "fp8",
        torch.bfloat16: "bf16",
        torch.float: "float",
        torch.int: "int",
    }[dt]


@dataclass
class BenchmarkTensors:
    w_ref: torch.Tensor
    a: torch.Tensor

    w_q: torch.Tensor
    a_q: torch.Tensor
    wtype: ScalarType
    atype: ScalarType
    w_ch_s: Optional[torch.Tensor]
    w_tok_s: Optional[torch.Tensor]


@dataclass
class TypeConfig:
    act_type: torch.dtype
    weight_type: ScalarType
    output_type: Optional[torch.dtype]
    group_scale_type: Optional[torch.dtype]
    group_zero_type: Optional[torch.dtype]
    channel_scale_type: Optional[torch.dtype]
    token_scale_type: Optional[torch.dtype]


def rand_data(shape, dtype=torch.float16, scale=1):
    if dtype.is_floating_point:
        return (scale * torch.rand(shape, device="cuda") - 0.3).to(dtype)
    else:
        return torch.randint(-15, 15, shape, dtype=dtype, device="cuda")


def quantize_and_pack(atype: torch.dtype,
                      w: torch.Tensor,
                      wtype: ScalarType,
                      stype: Optional[torch.dtype],
                      group_size: Optional[int],
                      zero_points: bool = False):
    assert wtype.is_integer(), "TODO: support floating point weights"

    w_ref, w_q, w_s, w_zp = quantize_weights(
        w,
        wtype,
        group_size=group_size,
        zero_points=zero_points,
        # to match how the kernel applies zps
        ref_zero_points_after_scales=True)

    w_q = pack_rows(w_q, wtype.size_bits, *w_q.shape)
    return w_ref, w_q, w_s, w_zp


def create_bench_tensors(shape: Tuple[int, int, int], types: TypeConfig,
                         group_size: Optional[int], workload_type_memory_hierarchy:str="L2") -> List[BenchmarkTensors]:
    m, n, k = shape

    # For MI300X
    # L1 Cache is 16KB per CU
    # L2 Cache is 16MB
    # L3 Cache is 256MB
    # we want to make sure that weights don't fit into L2 cache between runs so
    #  we construct enough weights to exceed L2 cache, which is 16mb on a MI300X
    #  so we target total weight size > 2*16mb
    # num_weights = math.ceil(2 * 50 * 1024**2 * 8 /
    #                         (k * n * types.weight_type.size_bits))

    num_weights = 1
    if workload_type_memory_hierarchy == "L1":
        num_weights = 1
    elif workload_type_memory_hierarchy == "L2":
        num_weights = math.ceil(2 * 16 * 1024**2 * 8 /
                                (k * n * types.weight_type.size_bits))
    elif workload_type_memory_hierarchy == "L3":
        num_weights = math.ceil(2 * 256 * 1024**2 * 8 /
                                (k * n * types.weight_type.size_bits))

    a = rand_data((m, k), types.act_type, scale=5)

    benchmark_tensors: List[BenchmarkTensors] = []
    for _ in range(num_weights):
        w = rand_data((k, n), types.act_type, scale=5)

        a_scale = rand_data((m, 1), torch.float32, scale=5)
        w_scale = rand_data((n, 1), torch.float32, scale=5)

        if not a.dtype.is_floating_point:
            aiinfo = torch.iinfo(a.dtype)
            w_ref = w_ref.round().clamp(aiinfo.min, aiinfo.max)

        benchmark_tensors.append(
            BenchmarkTensors(w_ref=w,
                             a=a,
                             a_q=a.contiguous(),
                             w_q=w.contiguous(),
                             wtype=types.weight_type,
                             atype=types.act_type,
                             w_ch_s=w_scale,
                             w_tok_s=a_scale))

    return benchmark_tensors


def torch_matmul_f16_create_bench_fn(bt: BenchmarkTensors) -> Callable:
    a = bt.a
    w = bt.w_ref.to(bt.a.dtype)  # use float reference tensor
    if a.dtype not in [torch.float16, torch.bfloat16]:
        a = a.to(torch.float16)
        w = w.to(torch.float16)
    return lambda: torch.matmul(a, w)


# FP8 benchmark functions
def fp8_row_create_bench_fn(bt: BenchmarkTensors) -> Callable:

    w_q = bt.w_q.transpose(1,0).contiguous()
    def run_gemm() -> torch.Tensor:
        return torch.ops._fp8gemm_C.f8f8bf16_rowwise(bt.a_q, w_q, bt.w_tok_s, bt.w_ch_s, None, True)

    return run_gemm

def bench_fns(label: str, sub_label: str, description: str,
              fns: List[Callable]):

    min_run_time = 1 if not NVTX_PROFILE else 0.1
    res = TBenchmark.Timer(
        stmt="""
        for fn in fns:
            fn()
        """,
        globals={
            "fns": fns
        },
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)

    if NVTX_PROFILE:
        with nvtx.annotate("mm-bench"), nvtx.annotate(
                f"{label}|{sub_label}|{description}"):
            fns[0]()

    return res


_SWEEP_SCHEDULES_RESULTS: Optional[pd.DataFrame] = None
_SWEEP_SCHEDULES_RESULTS_CSV: Optional[str] = None


def bench(types: TypeConfig,
          group_size: int,
          m: int,
          k: int,
          n: int,
          label: str,
          sub_label: str,
          workload_type_memory_hierarchy:str,
          sweep_schedules: bool = True) -> List[TMeasurement]:
    benchmark_tensors = create_bench_tensors((m, n, k), types, group_size, workload_type_memory_hierarchy)
    sub_label += f", L={len(benchmark_tensors)}+Mem_{workload_type_memory_hierarchy}"

    name_type_string = f"W{types.weight_type}"+\
                       f"-A{terse_type_name(types.act_type)}"
    if types.group_scale_type is not None:
        name_type_string += f"-GS{terse_type_name(types.group_scale_type)}"
    if types.group_zero_type is not None:
        name_type_string += f"-GZ{terse_type_name(types.group_zero_type)}"
    if group_size is not None:
        name_type_string += f"-G{group_size}"
    if types.channel_scale_type is not None:
        name_type_string += f"-CS{terse_type_name(types.channel_scale_type)}"
    if types.token_scale_type is not None:
        name_type_string += f"-TS{terse_type_name(types.token_scale_type)}"

    timers = []
    # pytorch impl
    timers.append(
        bench_fns(
            label, sub_label, "torch.matmul (fp16)",
            [torch_matmul_f16_create_bench_fn(bt)
             for bt in benchmark_tensors]))
    
    # FP8 benchmarks
    timers.append(
        bench_fns(label, sub_label, "fp8 row-wise gemm", [
            fp8_row_create_bench_fn(bt)
            for bt in benchmark_tensors
        ]))

    return timers


# runner
def print_timers(timers: List[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


def run(args, MKNs: Iterable[Tuple[int, int, int]]) -> Iterable[TMeasurement]:
    types = TypeConfig(
        act_type=args.act_type,
        # weight_type=scalar_types.uint4b8 if args.group_zero_type is None \
        #     else scalar_types.uint4,
        weight_type=scalar_types.float8_e4m3fnuz,
        output_type=args.out_type,
        group_scale_type=args.group_scale_type,
        group_zero_type=args.group_zero_type,
        channel_scale_type=args.channel_scale_type,
        token_scale_type=args.token_scale_type,
    )

    results: List[TMeasurement] = []
    for m, k, n in MKNs:
        for workload_type_memory_hierarchy in ["L1", "L2", "L3"]:
            timers = bench(types,
                        args.group_size,
                        m,
                        k,
                        n,
                        f"{args.act_type}-gemm",
                        f"MKN=({m}x{k}x{n})",
                        workload_type_memory_hierarchy=workload_type_memory_hierarchy,
                        sweep_schedules=args.sweep_schedules)
            print_timers(timers)
            results.extend(timers)

    return results


# output makers
def make_output(
    data: List[TMeasurement],
    MKNs: Iterable[Tuple[int, int, int]],
    base_description: str,
    timestamp=None,
):

    print(f"== All Results {base_description} ====")
    print_timers(data)

    # pickle all the results
    timestamp = int(time.time()) if timestamp is None else timestamp
    with open(f"{base_description}-{timestamp}.pkl", "wb") as f:
        pkl.dump(data, f)


# argparse runners


def run_square_bench(args):
    dim_sizes = list(
        range(args.dim_start, args.dim_end + 1, args.dim_increment))
    MKNs = list(zip(dim_sizes, dim_sizes, dim_sizes))
    data = run(args, MKNs)

    make_output(data, MKNs, f"square_bench-{args.act_type}")


def run_range_bench(args):
    m_start, k_start, n_start = (int(x) for x in args.dim_start.split(","))
    m_end, k_end, n_end = (int(x) for x in args.dim_end.split(","))
    m_increment, k_increment, n_increment = \
        (int(x) for x in args.dim_increment.split(","))
    Ms = list(range(m_start, m_end + 1, m_increment))
    Ks = list(range(k_start, k_end + 1, k_increment))
    Ns = list(range(n_start, n_end + 1, n_increment))
    MKNs = list(product(Ms, Ks, Ns))

    data = run(args, MKNs)

    make_output(data, MKNs, f"range_bench-{args.act_type}")

def run_uc_bench(args):
    MKNs = [
        (32, 1280, 8192),
        (32, 8192, 1024),
        (32, 7168, 8192),
        (32, 8192, 3584),
    ]

    data = run(args, MKNs)

    make_output(data, MKNs, f"uc_bench-{args.act_type}")


if __name__ == "__main__":

    def to_torch_dtype(dt):
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "int8": torch.int8,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "float8_e4m3fnuz": torch.float8_e4m3fnuz,
            "int": torch.int,
            "float": torch.float,
        }[dt]

    class ToTorchDtype(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, to_torch_dtype(values))

    parser = FlexibleArgumentParser(
        description="""
Benchmark Machete GEMM.

    To run square GEMMs:
        HIP_VISIBLE_DEVICES=5 python3 benchmarks/kernels/benchmark_fbgemm_ck_fp8.py --act-type float8_e4m3fnuz  square_bench --dim-start 128 --dim-end 512 --dim-increment 64
    
    To run constant N and K and sweep M:
        HIP_VISIBLE_DEVICES=5 python3 benchmarks/kernels/benchmark_fbgemm_ck_fp8.py --act-type float8_e4m3fnuz range_bench --dim-start 2048,4096,4096 --dim-end 4096,8192,4096 --dim-increment 1024,1024,1
    
    To run UC benchmark:
        HIP_VISIBLE_DEVICES=5 python3 benchmarks/kernels/benchmark_fbgemm_ck_fp8.py --act-type float8_e4m3fnuz uc_bench
        
    Output:
        - a .pkl file, that is a list of raw torch.benchmark.utils.Measurements for the pytorch and cutlass implementations for the various GEMMs.
            """,  # noqa: E501
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--act-type",
        action=ToTorchDtype,
        required=True,
        choices=['bfloat16', 'float16', 'int8', 'float8_e4m3fn', 'float8_e4m3fnuz'],
    )
    parser.add_argument(
        "--group-scale-type",
        action=ToTorchDtype,
        choices=['bfloat16', 'float16'],
    )
    parser.add_argument(
        "--group-zero-type",
        type=to_torch_dtype,
        choices=['bfloat16', 'float16'],
    )
    parser.add_argument(
        "--channel-scale-type",
        action=ToTorchDtype,
        choices=['float'],
    )
    parser.add_argument(
        "--token-scale-type",
        action=ToTorchDtype,
        choices=['float'],
    )
    parser.add_argument(
        "--out-type",
        action=ToTorchDtype,
        choices=['bfloat16', 'float16'],
    )
    parser.add_argument(
        "--group-size",
        type=int,
        help="Available options are ['None', '-1', '128'], default=128",
        default=128,
    )
    parser.add_argument(
        "--sweep-schedules",
        action="store_true",
        help="Run a sweep over all supported schedules",
    )
    parser.add_argument("--sweep-csv-out",
                        help="CSV to store sweep results",
                        default="sch_sweep_results.csv")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    square_parser = subparsers.add_parser("square_bench")
    square_parser.add_argument("--dim-start", type=int, required=True)
    square_parser.add_argument("--dim-end", type=int, required=True)
    square_parser.add_argument("--dim-increment", type=int, required=True)
    square_parser.set_defaults(func=run_square_bench)

    range_parser = subparsers.add_parser("range_bench")
    range_parser.add_argument(
        "--dim-start",
        type=str,
        required=True,
        help="Start value for M,K,N as common separated list")
    range_parser.add_argument(
        "--dim-end",
        type=str,
        required=True,
        help="End value (inclusive) for M,K,N as common separated list")
    range_parser.add_argument(
        "--dim-increment",
        type=str,
        required=True,
        help="Increment value for M,K,N as common separated list")
    range_parser.set_defaults(func=run_range_bench)

    uc_parser = subparsers.add_parser("uc_bench")
    uc_parser.set_defaults(func=run_uc_bench)

    args = parser.parse_args()

    _SWEEP_SCHEDULES_RESULTS_CSV = args.sweep_csv_out
    args.func(args)

    if _SWEEP_SCHEDULES_RESULTS is not None:
        _SWEEP_SCHEDULES_RESULTS.to_csv(_SWEEP_SCHEDULES_RESULTS_CSV)