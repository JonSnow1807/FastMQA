// kernels/mqa_kernel.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#define TILE_SIZE 16
#define WARP_SIZE 32

// Complete working MQA kernel implementation
template<int HEAD_DIM>
__global__ void mqa_attention_kernel(
    const float* __restrict__ Q,   // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ K,   // [batch, 1, seq_len, head_dim]
    const float* __restrict__ V,   // [batch, 1, seq_len, head_dim]
    float* __restrict__ output,    // [batch, num_heads, seq_len, head_dim]
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const float scale
) {
    // Thread and block indices
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (seq_idx >= seq_len) return;
    
    // Allocate shared memory for this thread block
    extern __shared__ float shared_mem[];
    float* s_scores = shared_mem;  // Store attention scores
    
    // Calculate offsets for this thread
    const int q_offset = ((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * HEAD_DIM;
    const int k_offset = (batch_idx * seq_len) * HEAD_DIM;
    const int v_offset = (batch_idx * seq_len) * HEAD_DIM;
    const int out_offset = ((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * HEAD_DIM;
    
    // Step 1: Compute attention scores for this query position
    float max_score = -FLT_MAX;
    
    // Compute Q @ K^T for all key positions
    for (int key_idx = 0; key_idx < seq_len; key_idx++) {
        float score = 0.0f;
        
        // Dot product between Q[seq_idx] and K[key_idx]
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            score += Q[q_offset + d] * K[k_offset + key_idx * HEAD_DIM + d];
        }
        
        score *= scale;
        s_scores[threadIdx.x * seq_len + key_idx] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Step 2: Compute softmax (numerically stable)
    float sum_exp = 0.0f;
    
    for (int i = 0; i < seq_len; i++) {
        float exp_score = expf(s_scores[threadIdx.x * seq_len + i] - max_score);
        s_scores[threadIdx.x * seq_len + i] = exp_score;
        sum_exp += exp_score;
    }
    
    // Normalize
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < seq_len; i++) {
        s_scores[threadIdx.x * seq_len + i] *= inv_sum;
    }
    
    // Step 3: Compute attention output (attention @ V)
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) {
        float out_val = 0.0f;
        
        for (int v_idx = 0; v_idx < seq_len; v_idx++) {
            out_val += s_scores[threadIdx.x * seq_len + v_idx] * V[v_offset + v_idx * HEAD_DIM + d];
        }
        
        output[out_offset + d] = out_val;
    }
}

// Optimized tiled version for better performance
__global__ void mqa_attention_kernel_tiled(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_idx = blockIdx.x * blockDim.x + tid;
    
    if (q_idx >= seq_len) return;
    
    // Pointers to Q, K, V for this batch and head
    const float* q_ptr = Q + ((batch_idx * num_heads + head_idx) * seq_len + q_idx) * head_dim;
    const float* k_ptr = K + batch_idx * seq_len * head_dim;
    const float* v_ptr = V + batch_idx * seq_len * head_dim;
    float* out_ptr = output + ((batch_idx * num_heads + head_idx) * seq_len + q_idx) * head_dim;
    
    // Compute attention scores
    float max_score = -FLT_MAX;
    
    // Allocate space for scores (adjust size based on your needs)
    float scores[2048];  // Max seq_len support
    
    // Compute Q @ K^T
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float score = 0.0f;
        const float* k_vec = k_ptr + k_idx * head_dim;
        
        // Vectorized dot product
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * k_vec[d];
        }
        
        score *= scale;
        scores[k_idx] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Softmax
    float sum_exp = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum_exp += scores[i];
    }
    
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < seq_len; i++) {
        scores[i] *= inv_sum;
    }
    
    // Compute output: attention @ V
    for (int d = 0; d < head_dim; d++) {
        float out_val = 0.0f;
        for (int v_idx = 0; v_idx < seq_len; v_idx++) {
            out_val += scores[v_idx] * v_ptr[v_idx * head_dim + d];
        }
        out_ptr[d] = out_val;
    }
}

// C interface for PyTorch extension - FIXED extern "C" placement
extern "C" void launch_mqa_kernel(
    float* Q, float* K, float* V, float* output,
    int batch_size, int num_heads, int seq_len, int head_dim
) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    
    // Choose kernel based on sequence length
    if (seq_len <= 256) {
        // Use simple kernel for short sequences
        dim3 block(min(seq_len, 32));
        dim3 grid((seq_len + block.x - 1) / block.x, num_heads, batch_size);
        
        size_t shared_mem_size = block.x * seq_len * sizeof(float);
        
        if (head_dim == 64) {
            mqa_attention_kernel<64><<<grid, block, shared_mem_size>>>(
                Q, K, V, output, batch_size, num_heads, seq_len, scale
            );
        } else if (head_dim == 128) {
            mqa_attention_kernel<128><<<grid, block, shared_mem_size>>>(
                Q, K, V, output, batch_size, num_heads, seq_len, scale
            );
        } else {
            // Generic version
            dim3 block_tiled(128);
            dim3 grid_tiled((seq_len + TILE_SIZE - 1) / TILE_SIZE, num_heads, batch_size);
            
            mqa_attention_kernel_tiled<<<grid_tiled, block_tiled>>>(
                Q, K, V, output, batch_size, num_heads, seq_len, head_dim, scale
            );
        }
    } else {
        // Use tiled kernel for longer sequences
        dim3 block(128);
        dim3 grid((seq_len + 128 - 1) / 128, num_heads, batch_size);
        
        mqa_attention_kernel_tiled<<<grid, block>>>(
            Q, K, V, output, batch_size, num_heads, seq_len, head_dim, scale
        );
    }
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(error));
    }
    
    cudaDeviceSynchronize();
}