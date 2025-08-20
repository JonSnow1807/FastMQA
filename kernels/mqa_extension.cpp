// kernels/mqa_extension.cpp
#include <torch/extension.h>
#include <vector>

// Forward declaration of CUDA kernel launcher - extern "C" to match
extern "C" void launch_mqa_kernel(
    float* Q, float* K, float* V, float* output,
    int batch_size, int num_heads, int seq_len, int head_dim
);

// PyTorch wrapper
torch::Tensor mqa_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    // Check inputs
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Only float32 supported");
    
    // Get dimensions
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    
    // Check K and V have single head
    TORCH_CHECK(K.size(1) == 1, "K must have single head for MQA");
    TORCH_CHECK(V.size(1) == 1, "V must have single head for MQA");
    
    // Allocate output
    auto output = torch::empty_like(Q);
    
    // Launch kernel
    launch_mqa_kernel(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, num_heads, seq_len, head_dim
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mqa_forward, "FastMQA forward (CUDA)");
}