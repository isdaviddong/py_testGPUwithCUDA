### 程式碼說明

以下是使用 CUDA 和 PyCUDA 在 GPU 上計算和繪製 Julia 集合，並比較 CPU 和 GPU 運行時間的完整 Python 程式碼範例。
這段程式碼展示了如何使用 CUDA 和 PyCUDA 加速 Julia 集合的計算，並比較 GPU 和 CPU 的運行時間，最終通過 Matplotlib 繪製結果。

#### 引入必要的函式庫

```python
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
```

#### 定義 CUDA 核心程式碼

```python
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
```

#### 編譯 CUDA 核心程式碼

```python
mod = SourceModule(kernel_code)
julia_gpu = mod.get_function("julia")
```

#### 設置 Julia 集合計算的參數

```python
width, height = 800, 800
c_real, c_imag = -0.7, 0.27015
max_iter = 256
```

#### 初始化坐標

```python
x = np.linspace(-1.5, 1.5, width).astype(np.float32)
y = np.linspace(-1.5, 1.5, height).astype(np.float32)
z_real, z_imag = np.meshgrid(x, y)
z_real = z_real.astype(np.float32).ravel()
z_imag = z_imag.astype(np.float32).ravel()
output_cpu = np.zeros(width * height, dtype=np.int32)
output_gpu = np.zeros(width * height, dtype=np.int32)
```

#### GPU 計算

1. **分配 GPU 記憶體**：
    ```python
    z_real_gpu = cuda.mem_alloc(z_real.nbytes)
    z_imag_gpu = cuda.mem_alloc(z_imag.nbytes)
    output_gpu_mem = cuda.mem_alloc(output_gpu.nbytes)
    ```

2. **將數據從主機複製到設備**：
    ```python
    cuda.memcpy_htod(z_real_gpu, z_real)
    cuda.memcpy_htod(z_imag_gpu, z_imag)
    ```

3. **設置網格和區塊尺寸**：
    ```python
    block_size = (16, 16, 1)
    grid_size = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1])
    ```

4. **啟動 CUDA 核心並計算時間**：
    ```python
    start_time = time.time()
    julia_gpu(z_real_gpu, z_imag_gpu, output_gpu_mem, np.int32(width), np.int32(height), np.float32(c_real), np.float32(c_imag), np.int32(max_iter), block=block_size, grid=grid_size)
    cuda.Context.synchronize()
    gpu_time = time.time() - start_time
    ```

5. **將結果從設備記憶體複製回主機記憶體**：
    ```python
    cuda.memcpy_dtoh(output_gpu, output_gpu_mem)
    ```

6. **打印 GPU 計算時間**：
    ```python
    print(f'GPU 計算時間: {gpu_time:.6f} 秒')
    ```

#### CPU 計算

1. **開始計時並計算 Julia 集合**：
    ```python
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
    ```

2. **打印 CPU 計算時間**：
    ```python
    print(f'CPU 計算時間: {cpu_time:.6f} 秒')
    ```

#### 繪製結果

1. **創建圖形並設置尺寸**：
    ```python
    plt.figure(figsize=(12, 6))
    ```

2. **繪製 CPU 結果**：
    ```python
    plt.subplot(121)
    plt.title(f'CPU 計算 ({cpu_time:.6f} 秒)')
    plt.imshow(output_cpu.reshape((height, width)), cmap='inferno', extent=(-1.5, 1.5, -1.5, 1.5))
    ```

3. **繪製 GPU 結果**：
    ```python
    plt.subplot(122)
    plt.title(f'GPU 計算 ({gpu_time:.6f} 秒)')
    plt.imshow(output_gpu.reshape((height, width)), cmap='inferno', extent=(-1.5, 1.5, -1.5, 1.5))
    ```

4. **顯示圖形**：
    ```python
    plt.show()
    ```
