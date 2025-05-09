## generate
**VidKV-main/transformers/src/transformers/generation/utils.py**

```
max_cache_length = generation_config.max_length - 1
self._prepare_cache_for_generation(
    generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
)
```
md
这里初始化了一个逻辑上的cache，方法里面指定了新设计的backend

## vidkv pack
**VidKV-main/transformers/src/transformers/vidkv_quant.py**
### 1. 核心函数：triton_quantize_and_pack_along_last_dim
作用：对输入张量进行量化和打包，支持不同位宽（1、1.58、2、4、8 bit）的压缩。

流程：
输入处理：接受张量 data 和量化参数（group_size, bit）。
分组与重塑：将张量 reshape 为 (B*nh*D, num_groups, group_size)，便于按组处理。

量化策略：
bit=1：二值量化（-1/1），将数据映射到 [0, 1] 并打包。
bit=1.58：三值量化（-1/0/1），阈值分割数据。
其他整数 bit：线性量化，计算每个组的 min/max，缩放为 [0, 2^bit-1]。

打包：调用 _pack_along_last_dim（Triton 内核）将低比特数据压缩到 int32。

返回结果：返回压缩后的 code 张量及量化参数 scale 和 mn。
### 2. 量化辅助函数
#### (1) _minmax_along_last_dim (Triton 内核)
作用：在最后一个维度上计算每个组的最小值 (mn) 和最大值 (mx)。

关键逻辑：
使用 Triton 的线程块并行处理张量。
对每个组内的 group_size 元素计算极值。
将结果写入输出张量 mn 和 mx。
#### (2) _pack_along_last_dim (Triton 内核)
作用：将低比特量化数据打包到 int32 张量中。

关键逻辑：
每个 int32 可存储 32//bit 个低比特值（如 bit=2 时，每个 int32 存储 16 个值）。
通过位操作将多个量化值合并到一个整数中（左移 + 或运算）。
### 3. 解包与反量化函数
#### (1) unpack_tensor
作用：将打包的 int32 张量解包为低比特整数张量。

关键逻辑：
根据 bit 计算每个 int32 对应的低比特值数量（feat_per_int = 32//bit）。
通过位移操作提取每个低比特值（掩码 0xFF >> (8-bit) 保留有效位）。
#### (2) unpack_and_dequant_vcache
作用：将量化后的 code 张量反量化为原始精度（float16/float32）。

关键逻辑：
调用 unpack_tensor 解包低比特整数。
根据量化参数 scale 和 mn 进行反量化：data = code * scale + mn。

特殊处理：
bit=1.58：直接返回解包后的值（无 scale/mn）。
bit=1：反量化时需将 [0,1] 映射为 [-1,1]。
### 4. 辅助函数
#### (1) pack_tensor
作用：将低比特整数张量手动打包到 int32 张量（非 Triton 实现）。

用途：测试或调试时验证打包逻辑。


## cache backend
**/VidKV-main/transformers/src/transformers/cache_utils.py**

### 默认缓存类：DynamicCache
存储结构：DynamicCache 使用 key_cache 和 value_cache 列表存储每一层的键值张量。

更新逻辑：每次生成新 token 时，update 方法会将当前的 key_states 和 value_states 直接拼接到对应的缓存中（无量化）。

内存占用：由于未进行量化，内存占用较高，尤其在长序列生成时。

### QuantizedTensorFunction 类
作用：提供了一个统一的接口，用于对输入张量进行量化和反量化、试不同位宽的量化效果。

后续的子类更进一步确定了更新策略：如何对缓存进行量化、如何动态更新缓存。
 - _quantize_key() 和 _quantize_value()
 - _dequantize_key() 和 _dequantize_value()

### 2-bit quantization: QuantizedCacheLM
'''
QUANTIZATION_CONFIG="{'backend': 'QuantizedCacheLM', 'nbits': 2, 'q_group_size': 32, 'residual_length': 128, 'axis_key': -1, 'axis_value': -1}"
'''
### 1.5-bit and 2-bit quantization: [1.5, 2] means K-1.5-bit & V-2-bit QuantizedCacheVLM
'''
QUANTIZATION_CONFIG="{'backend': 'QuantizedCacheVLM', 'nbits': [1.5, 2], 'q_group_size': 32, 'residual_length': 128, 'axis_key': -1, 'axis_value': -1}"
'''
### 1.5-bit and 1.58-bit quantization:QuantizedCacheVLM
'''
QUANTIZATION_CONFIG="{'backend': 'QuantizedCacheVLM', 'nbits': [1.5, 1.58], 'q_group_size': 32, 'residual_length': 128, 'axis_key': -1, 'axis_value': -1}"
'''
### Quantization with STP: QuantizedCacheVLMSTP
'''
QUANTIZATION_CONFIG="{'backend': 'QuantizedCacheVLMSTP', 'nbits': [1.5, 1.58], 'q_group_size': 32, 'residual_length': 128, 'axis_key': -1, 'axis_value': -1, "vidkv_stp": 0.2}"
'''