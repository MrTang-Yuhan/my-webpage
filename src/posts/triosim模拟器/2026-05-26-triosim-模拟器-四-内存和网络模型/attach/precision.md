对，你理解得很准确。完整流程可以这样看：

1. 底层类型与算术（C++）
- `bfloat16`：直接用 `Eigen::bfloat16`。
- `float8/float6/float4/int2/int4/uint2/uint4`：在 `ml_dtypes/include/*.h` 里定义类型、位表示、转换、比较、`numeric_limits` 等。
- 一些通用数值能力会借助 Eigen（如 `half`、`numext` 等）。

2. Python/NumPy 绑定层（C++ 扩展）
- 在 `ml_dtypes/_src/dtypes.cc` 里给每个类型写 `TypeDescriptor<...>`（类型名、kind、type char、文档等）。
- 调用 `RegisterFloatDtype` / `RegisterIntNDtype` 把它们注册成 NumPy 认识的自定义 dtype。
- 同时注册 cast 规则（类型互转）和 ufunc 实现（加减乘除、exp/log、比较等，见 `custom_float.h` + `ufuncs.h`）。

3. 生成扩展模块 `_ml_dtypes_ext`
- 构建后得到 Python 可导入的二进制模块。
- 模块初始化时（`PyInit__ml_dtypes_ext`）把各 dtype 类型对象挂到模块属性上。

4. Python 包导出
- `ml_dtypes/__init__.py` 从 `_ml_dtypes_ext` 导入 `bfloat16`、`float8_*`、`int*` 等名称。
- 同时导出 `finfo` / `iinfo`，让这些 dtype 在 Python 里像普通 NumPy dtype 一样可用。

一句话总结：
“底层 C++（部分依赖 Eigen）实现数值类型与运算 → C++ 扩展注册为 NumPy dtype/ufunc/cast → Python 层直接导出给用户使用。”

```markdown
# ml_dtypes

[![单元测试](https://github.com/jax-ml/ml_dtypes/actions/workflows/test.yml/badge.svg)](https://github.com/jax-ml/ml_dtypes/actions/workflows/test.yml)
[![Wheel 构建](https://github.com/jax-ml/ml_dtypes/actions/workflows/wheels.yml/badge.svg)](https://github.com/jax-ml/ml_dtypes/actions/workflows/wheels.yml)
[![PyPI 版本](https://badge.fury.io/py/ml_dtypes.svg)](https://badge.fury.io/py/ml_dtypes)

`ml_dtypes` 是一个独立实现的库，提供了多种机器学习库中常用的 NumPy 数据类型扩展，包括：

- [`bfloat16`](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)：
  一种可替代标准 [`float16`](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) 的浮点数格式。
- 8 位浮点数表示。此类格式通过指数位数、尾数位数、偏置值，以及是否支持无穷大、NaN 和有符号零等特性进行参数化。
  - `float8_e3m4`
  - `float8_e4m3`
  - `float8_e4m3b11fnuz`
  - `float8_e4m3fn`
  - `float8_e4m3fnuz`
  - `float8_e5m2`
  - `float8_e5m2fnuz`
  - `float8_e8m0fnu`
- Microscaling（MX）子字节浮点数表示：
  - `float4_e2m1fn`
  - `float6_e2m3fn`
  - `float6_e3m2fn`
- 窄位宽整数编码：
  - `int2`
  - `int4`
  - `uint2`
  - `uint4`

这些数值格式的具体规范见下文。

## 安装

`ml_dtypes` 包已在 Python 3.9–3.12 版本上进行测试，可以使用以下命令安装：

```bash
pip install ml_dtypes
```

若要测试安装是否成功，可以运行：

```bash
pip install absl-py pytest
pytest --pyargs ml_dtypes
```

若要从源码构建，请克隆仓库并运行：

```bash
git submodule init
git submodule update
pip install .
```

## 使用示例

```python
>>> from ml_dtypes import bfloat16
>>> import numpy as np
>>> np.zeros(4, dtype=bfloat16)
array([0, 0, 0, 0], dtype=bfloat16)
```

导入 `ml_dtypes` 后，这些数据类型也会注册到 NumPy 中，因此可以通过字符串名称引用它们：

```python
>>> np.dtype('bfloat16')
dtype(bfloat16)
>>> np.dtype('float8_e5m2')
dtype(float8_e5m2)
```

## 已实现浮点格式规范

### `bfloat16`

`bfloat16` 数值可以理解为将单精度浮点数截断为 16 位后的格式。

指数位：8，尾数位：7，指数偏置：127。符合 IEEE 754，支持 NaN 和 inf。

### `float4_e2m1fn`

指数位：2，尾数位：1，偏置：1。

扩展范围：不支持 inf，不支持 NaN。

Microscaling 格式，4 位，编码为 `0bSEEM`，使用字节存储，其中高 4 位未使用。NaN 表示未定义。

可能的绝对值包括：[`0`, `0.5`, `1`, `1.5`, `2`, `3`, `4`, `6`]

### `float6_e2m3fn`

指数位：2，尾数位：3，偏置：1。

扩展范围：不支持 inf，不支持 NaN。

Microscaling 格式，6 位，编码为 `0bSEEMMM`，使用字节存储，其中高 2 位未使用。NaN 表示未定义。

可表示的取值范围：[`-7.5`; `7.5`]

### `float6_e3m2fn`

指数位：3，尾数位：2，偏置：3。

扩展范围：不支持 inf，不支持 NaN。

Microscaling 格式，4 位，编码为 `0bSEEEMM`，使用字节存储，其中高 2 位未使用。NaN 表示未定义。

可表示的取值范围：[`-28`; `28`]

### `float8_e3m4`

指数位：3，尾数位：4，偏置：3。符合 IEEE 754，支持 NaN 和 inf。

### `float8_e4m3`

指数位：4，尾数位：3，偏置：7。符合 IEEE 754，支持 NaN 和 inf。

### `float8_e4m3b11fnuz`

指数位：4，尾数位：3，偏置：11。

扩展范围：不支持 inf，NaN 由 `0b1000'0000` 表示。

### `float8_e4m3fn`

指数位：4，尾数位：3，偏置：7。

扩展范围：不支持 inf，NaN 由 `0bS111'1111` 表示。

后缀 `fn` 是为了与对应的 LLVM/MLIR 类型保持一致，表示该类型并不符合 IEEE 754。其中，`f` 表示仅支持有限值（finite values only），`n` 表示包含 NaN，但 NaN 只位于可表示范围的边界处。

### `float8_e4m3fnuz`

带 3 位尾数的 8 位浮点数。

这是一种 8 位浮点类型，包含 1 位符号位、4 位指数位和 3 位尾数位。后缀 `fnuz` 与 LLVM/MLIR 的命名保持一致，来源于该类型与 IEEE 浮点约定之间的差异。其中，`F` 表示“有限值”（finite，即不支持无穷大），`N` 表示采用特殊的 NaN 编码，`UZ` 表示无符号零（unsigned zero）。

该类型具有以下特性：

- 位编码：S1E4M3，即 `0bSEEEEMMM`
- 指数偏置：8
- 无穷大：不支持
- NaN：支持。当符号位为 1，指数位和尾数位全为 0 时表示 NaN，即 `0b10000000`
- 当指数位为 0 时表示非规格化数

### `float8_e5m2`

指数位：5，尾数位：2，偏置：15。符合 IEEE 754，支持 NaN 和 inf。

### `float8_e5m2fnuz`

带 2 位尾数的 8 位浮点数。

这是一种 8 位浮点类型，包含 1 位符号位、5 位指数位和 2 位尾数位。后缀 `fnuz` 与 LLVM/MLIR 的命名保持一致，来源于该类型与 IEEE 浮点约定之间的差异。其中，`F` 表示“有限值”（finite，即不支持无穷大），`N` 表示采用特殊的 NaN 编码，`UZ` 表示无符号零（unsigned zero）。

该类型具有以下特性：

- 位编码：S1E5M2，即 `0bSEEEEEMM`
- 指数偏置：16
- 无穷大：不支持
- NaN：支持。当符号位为 1，指数位和尾数位全为 0 时表示 NaN，即 `0b10000000`
- 当指数位为 0 时表示非规格化数

### `float8_e8m0fnu`

[OpenCompute MX](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) 缩放格式 E8M0，具有以下属性：

- 无符号格式
- 8 位指数位
- 指数范围为 -127 到 127
- 不支持零和无穷大
- 只有一个 NaN 值，即 `0xFF`

## `int2`、`int4`、`uint2` 和 `uint4`

这些是 2 位和 4 位整数类型，其中每个元素都以非打包形式表示，也就是说，在内存中会被填充到一个字节。

NumPy 不支持小于单个字节的类型。例如，数组中相邻元素之间的距离（`.strides`）是以整数个字节表示的。放宽这一限制将是一个相当大的工程项目。因此，这些类型采用非打包表示，即数组中的每个元素都会在内存中填充到一个字节。每个字节的低 2 位或低 4 位存储该数值的表示，其余高位会被忽略。

## 低精度算术的注意事项

如果你正在代码中尝试使用低精度数据类型，需要特别注意精度损失可能带来的意外结果。一个典型例子是 `sum` 这类聚合操作的行为。以下是在 NumPy 中对 `bfloat16` 进行求和的示例（使用版本 1.24.2 运行）：

```python
>>> from ml_dtypes import bfloat16
>>> import numpy as np
>>> rng = np.random.default_rng(seed=0)
>>> vals = rng.uniform(size=10000).astype(bfloat16)
>>> vals.sum()
256
```

真实的求和结果应该接近 5000，但 NumPy 返回的结果却恰好是 256。这是因为 `bfloat16` 的精度不足，无法用小于 1 的数值继续增加 `256`：

```python
>>> bfloat16(256) + bfloat16(1)
256
```

在 `bfloat16` 中，256 之后下一个可表示的数值是 258：

```python
>>> np.nextafter(bfloat16(256), bfloat16(np.inf))
258
```

若想获得更好的结果，可以指定使用更高精度的类型进行累加，例如 `float32`：

```python
>>> vals.sum(dtype='float32').astype(bfloat16)
4992
```

与 NumPy 不同，像 [JAX](http://jax.readthedocs.io/) 这类对低精度算术支持更加原生的项目，通常会自动使用更高精度来完成此类累加：

```python
>>> import jax.numpy as jnp
>>> jnp.array(vals).sum()
Array(4992, dtype=bfloat16)
```

## 许可证

*这不是 Google 官方支持的产品。*

`ml_dtypes` 源代码采用 Apache 2.0 许可证发布，详见 [LICENSE](LICENSE)。预编译 wheel 包使用 [EIGEN](https://eigen.tuxfamily.org/) 项目构建，EIGEN 采用 MPL 2.0 许可证发布，详见 [LICENSE.eigen](LICENSE.eigen)。
```