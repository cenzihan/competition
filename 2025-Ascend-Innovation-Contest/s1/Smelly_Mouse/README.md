# Smelly Mouse团队 昇腾AI创新大赛-昇思模型开发挑战赛MultiModal模型优化详细说明 (Janus-Pro-7B & Qwen2-VL-2B-Instruct)

本文档记录了针对多模态赛道两个模型的关键性能优化点。

## 1. Janus-Pro-7B 模型优化

### 1.1 SwiGLU 融合算子优化

在 `LlamaMLP` 的前向传播中，将原本分离的 gate_proj 和 up_proj 计算替换为 MindSpore 的融合算子 `mindspore.ops.swiglu`：

**优化前：**
```python
intermediate_states = self.act_fn(gate_proj) * up_proj
```

**优化后：**
```python
intermediate_states = mindspore.ops.swiglu(ops.cat([gate_proj, up_proj], dim=-1), dim=-1)
```

融合算子将激活函数和逐元素乘法合并为单个内核调用，减少了内存访问次数和内核启动开销。

### 1.2 Tensor 切片操作优化

使用 `ops.narrow` 替代 Python 切片语法，提升昇腾 NPU 上的执行效率：

**优化前：**
```python
logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
```

**优化后：**
```python
sliced_hidden = ops.narrow(hidden_states, 1, hidden_states.shape[1] - num_logits_to_keep, num_logits_to_keep)
logits = self.lm_head(sliced_hidden).float()
shift_logits = ops.narrow(logits, -2, 0, logits.shape[-2] - 1)
shift_labels = ops.narrow(labels, -1, 1, labels.shape[-1] - 1)
```

`ops.narrow` 是显式的算子调用，更利于计算图优化和算子融合。

## 2. Qwen2-VL-2B-Instruct 模型优化

### 2.1 旋转位置编码 (RoPE) 融合算子优化

将手动实现的旋转位置编码替换为 MindSpore 内置的 `mindspore.ops.rotary_position_embedding` 融合算子：

**优化前：**
```python
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```

**优化后：**
```python
q_embed = mindspore.ops.rotary_position_embedding(q, cos, sin)
k_embed = mindspore.ops.rotary_position_embedding(k, cos, sin)
```

融合算子减少了多次张量乘法和拼接操作，显著提升了注意力计算的效率。

### 2.2 rotate_half 函数优化

使用 `ops.split` 替代 Python 切片：

**优化前：**
```python
x1 = x[..., : x.shape[-1] // 2]
x2 = x[..., x.shape[-1] // 2 :]
```

**优化后：**
```python
x1, x2 = ops.split(x, x.shape[-1] // 2, dim=-1)
```

### 2.3 RMSNorm 融合算子优化

将手动实现的 RMS 归一化替换为 `F.rms_norm` 融合算子：

**优化前：**
```python
input_dtype = hidden_states.dtype
hidden_states = hidden_states.to(mindspore.float32)
variance = ops.mean(hidden_states.pow(2), -1, keepdim=True)
hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)
```

**优化后：**
```python
return F.rms_norm(hidden_states, self.weight, self.variance_epsilon)
```

`F.rms_norm` 将方差计算、rsqrt 和缩放操作融合为单个高效内核，减少了中间张量的内存分配和数据搬运。

## 评测结果

| 评测指标 | 平均得分 |
|---------|---------|
| 峰值显存得分 | 100 |
| Prefill时延得分 | 107.6535 |
| Decode时延得分 | 112.7979 |
| **总分** | **106.8171** |