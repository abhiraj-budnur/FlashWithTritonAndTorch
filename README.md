Flash Attention using Triton and PyTorch

A from-scratch implementation of Flash Attention using PyTorch and Triton, exploring memory-efficient attention computation, online softmax, GPU kernel programming, and performance benchmarking on an NVIDIA T4 GPU.

This project implements attention in three ways:

Standard PyTorch Attention
PyTorch Online Softmax Attention
Custom Triton Flash Attention

The primary goal is to understand how Flash Attention reduces GPU memory consumption by avoiding the materialization of the full attention matrix.

Table of Contents
Overview
What is Flash Attention?
Attention Computation
Online Softmax
Implementation
GPU Kernel Design
Experimental Setup
Benchmark Results
Memory Improvement
Execution Time Analysis
Correctness
Limitations and Future Improvements
How to Run
Overview

Traditional attention computes an attention score matrix between every query and key:

𝑆
=
𝑄
𝐾
𝑇
𝑑

It then applies softmax and multiplies the resulting attention weights by the value matrix:

𝑂
=
softmax
⁡
(
𝑆
)
𝑉

The problem is that the attention score matrix has dimensions:

𝑁
×
𝑁

For long sequences, this matrix consumes a significant amount of GPU memory.

Flash Attention addresses this problem through tiling and online softmax, allowing attention to be computed incrementally without storing the entire attention matrix.

This project implements that approach using Triton, a Python-based GPU programming language.

What is Flash Attention?

Flash Attention is an IO-aware attention algorithm that reduces the amount of data transferred between GPU memory and on-chip memory.

Instead of computing the complete attention matrix at once, it processes query, key, and value matrices in smaller blocks.

The implementation maintains three important quantities:

Running maximum: 
𝑚
𝑖
Running normalization sum: 
𝑙
𝑖
Running output accumulator: 
𝐴
𝑖

These quantities allow the softmax computation to be updated incrementally.

The final output is:

𝑂
𝑖
=
𝐴
𝑖
𝑙
𝑖

This avoids explicitly storing the full attention matrix.

Attention Computation

The standard scaled dot-product attention equation is:

Attention
⁡
(
𝑄
,
𝐾
,
𝑉
)
=
softmax
⁡
(
𝑄
𝐾
𝑇
𝑑
)
𝑉

Where:

𝑄
: Query matrix
𝐾
: Key matrix
𝑉
: Value matrix
𝑑
: Head dimension

In this implementation:

Parameter	Value
Head dimension	64
Value dimension	64
Data type	FP16 inputs
Accumulation	FP32
Sequence lengths	4096, 8192, 16284, 32768
Online Softmax

The central idea behind this implementation is online softmax.

Instead of calculating softmax over the complete sequence, the algorithm processes smaller key/value blocks.

For each block, it calculates:

𝑚
𝑛
𝑒
𝑤
=
max
⁡
(
𝑚
𝑜
𝑙
𝑑
,
max
⁡
(
𝑆
𝑏
𝑙
𝑜
𝑐
𝑘
)
)

The previous normalization contribution is rescaled:

𝛼
=
𝑒
𝑚
𝑜
𝑙
𝑑
−
𝑚
𝑛
𝑒
𝑤

The new attention weights are:

𝑃
=
𝑒
𝑆
𝑏
𝑙
𝑜
𝑐
𝑘
−
𝑚
𝑛
𝑒
𝑤

The running normalization sum becomes:

𝑙
𝑛
𝑒
𝑤
=
𝛼
𝑙
𝑜
𝑙
𝑑
+
∑
𝑃

The output accumulator becomes:

𝐴
𝑛
𝑒
𝑤
=
𝛼
𝐴
𝑜
𝑙
𝑑
+
𝑃
𝑉
𝑏
𝑙
𝑜
𝑐
𝑘

Finally:

𝑂
=
𝐴
𝑙

This allows the algorithm to process attention blocks without materializing the full 
𝑁
×
𝑁
 attention matrix.

Implementation

The project contains three implementations.

1. Standard PyTorch Attention
python
def standard_Attention(q, k, v):
    d = q.shape[-1]

    scores = q @ k.T / math.sqrt(d)
    weights = torch.softmax(scores, dim=-1)

    return weights @ v


This implementation directly computes the complete attention matrix.

It serves as the baseline for correctness and memory benchmarking.

2. PyTorch Online Softmax
python
def flash_attention(q, k, v, kv_block_size=64):

    n_q, d = q.shape
    n_k = k.shape[0]

    scale = 1.0 / math.sqrt(d)

    running_max = q.new_full((n_q, 1), -torch.inf)
    running_sum = q.new_zeros((n_q, 1))

    accumulator = q.new_zeros((n_q, v.shape[1]))

    for start in range(0, n_k, kv_block_size):

        k_tile = k[start:start + kv_block_size]
        v_tile = v[start:start + kv_block_size]

        scores = (q @ k_tile.T) * scale

        new_max = torch.maximum(
            running_max,
            scores.max(dim=-1, keepdim=True).values
        )

        old_scale = torch.exp(running_max - new_max)

        tile_weights = torch.exp(scores - new_max)

        running_sum = (
            old_scale * running_sum
            + tile_weights.sum(dim=-1, keepdim=True)
        )

        accumulator = (
            old_scale * accumulator
            + tile_weights @ v_tile
        )

        running_max = new_max

    return accumulator / running_sum


This implementation demonstrates online softmax using PyTorch operations.

3. Triton Flash Attention

The custom Triton kernel implements the same online softmax algorithm directly on the GPU.

Its responsibilities include:

Loading query, key, and value tiles.
Computing attention scores.
Applying numerically stable online softmax.
Updating the running normalization sum.
Updating the output accumulator.
Writing the final normalized output.

The kernel uses FP32 intermediate calculations and accumulation to maintain numerical stability.

The final configuration uses:

Parameter	Configuration
BLOCK_M	16
BLOCK_N	32
BLOCK_D	64
BLOCK_DV	64
num_warps	4
num_stages	1

The smaller key/value tile was selected to satisfy the shared-memory constraints of the NVIDIA T4.

Experimental Setup

All experiments were performed using Google Colab.

Component	Configuration
GPU	NVIDIA Tesla T4
Framework	PyTorch
GPU kernel language	Triton
Query dtype	FP16
Key dtype	FP16
Value dtype	FP16
Intermediate calculations	FP32
Head dimension	64
Value dimension	64
Warmup iterations	5
Benchmark iterations	20

Sequence lengths tested:

python
SEQUENCE_LENGTHS = [
    4096,
    8192,
    16284,
    32768
]

Benchmark Results

The following results were obtained on the NVIDIA T4.

Execution Time
Sequence Length	Standard PyTorch	Triton Flash Attention
4,096	1.70 ms	7.47 ms
8,192	4.44 ms	14.14 ms
16,284	20.85 ms	52.15 ms
32,768	67.94 ms	212.47 ms
Speedup

Speedup is calculated as:

Speedup
=
Standard PyTorch Time
Triton Time
Sequence Length	Speedup
4,096	0.23x
8,192	0.31x
16,284	0.40x
32,768	0.32x

Important: The current Triton implementation is slower than the standard PyTorch baseline. This is expected for an initial educational kernel that has not yet been fully optimized for Tensor Core utilization and GPU occupancy.

Memory Improvement

Memory efficiency is the strongest result of this implementation.

Sequence Length	Standard PyTorch	Triton Flash Attention
4,096	83.63 MB	19.62 MB
8,192	277.63 MB	21.62 MB
16,284	1037.58 MB	25.58 MB
32,768	4129.63 MB	25.62 MB
Memory Reduction

Memory reduction is calculated as:

Reduction
=
𝑀
𝑠
𝑡
𝑎
𝑛
𝑑
𝑎
𝑟
𝑑
−
𝑀
𝑡
𝑟
𝑖
𝑡
𝑜
𝑛
𝑀
𝑠
𝑡
𝑎
𝑛
𝑑
𝑎
𝑟
𝑑
×
100
Sequence Length	Memory Reduction
4,096	76.5%
8,192	92.2%
16,284	97.5%
32,768	99.4%

At a sequence length of 32,768:

Standard PyTorch attention uses approximately 4.13 GB.
Triton Flash Attention uses approximately 25.62 MB.
Memory consumption is reduced by approximately 99.4%.

This demonstrates the advantage of tiled attention computation for long sequences.

Execution Time Analysis

Although memory consumption is significantly lower, the current Triton kernel has not achieved a speed improvement.

There are several possible reasons:

Small tile size: BLOCK_N = 32 results in many iterations through the key/value sequence.
FP32 computation: The current implementation uses FP32 intermediate calculations, which may limit performance.
GPU utilization: The kernel has not yet been extensively tuned for occupancy and Tensor Core utilization.
Kernel overhead: A custom Triton implementation can have additional overhead compared with highly optimized CUDA libraries.

The next optimization target is improving execution speed while preserving the memory efficiency and numerical correctness.

Correctness

The custom Triton implementation was compared against standard PyTorch attention.

Sequence Length	Maximum Absolute Error	Mean Absolute Error	Correct
4,096	0.000305	0.0000232	Yes
8,192	0.000366	0.0000226	Yes
16,284	0.000366	0.0000210	Yes
32,768	0.000366	0.0000196	Yes

All four configurations passed the following numerical comparison:

python
torch.allclose(
    output_standard,
    output_flash,
    atol=1e-5,
    rtol=1e-5
)


The small numerical differences are expected due to floating-point computation and operation ordering.

Limitations and Future Improvements

This implementation focuses on understanding the algorithm and developing a working GPU kernel.

Potential future improvements include:

Increasing BLOCK_N where shared-memory constraints allow.
Experimenting with different BLOCK_M configurations.
Improving FP16 Tensor Core utilization.
Optimizing FP32 accumulation and mixed-precision operations.
Tuning the number of warps and pipeline stages.
Benchmarking against PyTorch SDPA.
Testing on more powerful GPUs.
Supporting causal attention and multiple attention heads.

A particularly important next step is comparing against PyTorch SDPA, which can dispatch to highly optimized attention implementations.

How to Run
1. Clone the repository
bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>

2. Install dependencies
bash
pip install torch triton


Use a CUDA-enabled environment with a compatible PyTorch and Triton installation.

3. Select a GPU

For Google Colab:

Open Runtime.
Select Change runtime type.
Choose T4 GPU.
4. Execute the implementation
bash
python flash_attention.py


The program will:

Generate random query, key, and value tensors.
Test numerical correctness.
Benchmark standard PyTorch attention.
Benchmark the custom Triton implementation.
Measure peak GPU memory consumption.
Print execution time, speedup, and memory statistics.
Key Takeaways

This project demonstrates how online softmax and tiled computation can significantly reduce the memory requirements of attention.

The main results are:

Successfully implemented online softmax using PyTorch.
Developed a custom Triton Flash Attention kernel.
Achieved numerical correctness across four sequence lengths.
Reduced peak memory consumption by up to 99.4% compared with standard PyTorch attention.
Identified GPU resource constraints on the NVIDIA T4.
Established a baseline for future GPU kernel optimization.

The most important takeaway: Efficient GPU programming is not only about implementing the correct mathematical algorithm. It is also about understanding memory hierarchy, GPU hardware constraints, tiling strategies, and computational throughput.

Acknowledgements

This project was developed as a learning exercise in GPU programming, attention mechanisms, online softmax, and Triton kernel development.