import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt

# CUDA 內核代碼
kernel_code = """
__global__ void julia(float *z_real, float *z_imag, int *output, int width, int height, float c_real, float c_imag, int max_iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float zr = z_real[idx];
    float zi = z_imag[idx];

    int iter;
    for (iter = 0; iter < max_iter; ++iter) {
        float zr2 = zr * zr;
        float zi2 = zi * zi;
        if (zr2 + zi2 > 4.0f) break;
        float new_zr = zr2 - zi2 + c_real;
        float new_zi = 2.0f * zr * zi + c_imag;
        zr = new_zr;
        zi = new_zi;
    }
    output[idx] = iter;
}
"""

# 編譯內核代碼
mod = SourceModule(kernel_code)
julia_gpu = mod.get_function("julia")

# Julia 集合計算的參數
width, height = 800, 800
c_real, c_imag = -0.7, 0.27015
max_iter = 256

# 初始化坐標
x = np.linspace(-1.5, 1.5, width).astype(np.float32)
y = np.linspace(-1.5, 1.5, height).astype(np.float32)
z_real, z_imag = np.meshgrid(x, y)
z_real = z_real.astype(np.float32).ravel()
z_imag = z_imag.astype(np.float32).ravel()
output_cpu = np.zeros(width * height, dtype=np.int32)
output_gpu = np.zeros(width * height, dtype=np.int32)

# GPU 計算
z_real_gpu = cuda.mem_alloc(z_real.nbytes)
z_imag_gpu = cuda.mem_alloc(z_imag.nbytes)
output_gpu_mem = cuda.mem_alloc(output_gpu.nbytes)

cuda.memcpy_htod(z_real_gpu, z_real)
cuda.memcpy_htod(z_imag_gpu, z_imag)

block_size = (16, 16, 1)
grid_size = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1])

start_time = time.time()
julia_gpu(z_real_gpu, z_imag_gpu, output_gpu_mem, np.int32(width), np.int32(height), np.float32(c_real), np.float32(c_imag), np.int32(max_iter), block=block_size, grid=grid_size)
cuda.Context.synchronize()
gpu_time = time.time() - start_time

cuda.memcpy_dtoh(output_gpu, output_gpu_mem)

print(f'GPU count: {gpu_time:.6f} sec')


# CPU 計算
start_time = time.time()
for i in range(width * height):
    zr, zi = z_real[i], z_imag[i]
    iter = 0
    for _ in range(max_iter):
        zr2, zi2 = zr * zr, zi * zi
        if zr2 + zi2 > 4.0:
            break
        zr, zi = zr2 - zi2 + c_real, 2.0 * zr * zi + c_imag
        iter += 1
    output_cpu[i] = iter
cpu_time = time.time() - start_time
print(f'CPU count: {cpu_time:.6f} sec')

# 繪製結果
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title(f'CPU count ({cpu_time:.6f} sec)')
plt.imshow(output_cpu.reshape((height, width)), cmap='inferno', extent=(-1.5, 1.5, -1.5, 1.5))

plt.subplot(122)
plt.title(f'GPU count ({gpu_time:.6f} sec)')
plt.imshow(output_gpu.reshape((height, width)), cmap='inferno', extent=(-1.5, 1.5, -1.5, 1.5))

plt.show()
