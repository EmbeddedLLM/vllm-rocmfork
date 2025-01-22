#include "utils.hpp"

namespace {
    int max_threads_per_wavefront(){
        return 64;
    }

    std::vector<int64_t> compute_wavefront_padding(const int64_t K){
        const auto threads_per_wavefront = max_threads_per_wavefront();
        int64_t padding_size = (threads_per_wavefront - (K % threads_per_wavefront)) % threads_per_wavefront;
        return {0, padding_size};
    }
}


at::Tensor machete_rocm::utils::align_to_wavefront(
    const at::Tensor tensor
){
    const auto K = tensor.size(1);
    const auto padding = compute_wavefront_padding(K);
    const auto padding_options = 
    torch::nn::functional::PadFuncOptions(padding).mode(torch::kConstant);
    return torch::nn::functional::pad(tensor, padding_options);
}