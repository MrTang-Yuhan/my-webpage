---
layout: post.njk
post_id: 2026-10-07-onnxim-模拟器-1-gemm-tile-脉动阵列与存储映射
archive: 模拟器
title: ONNXim 模拟器（1）：GEMM、Tile、脉动阵列与存储映射
date: 2026-10-07
updated: 2026-10-07
tags:
  - post
---
# 1. 从完整 GEMM 到 Tile，再到一条计算指令

## 1.1 先明确这几个维度在本文中的含义

理解分块只需要先知道：**N 是输入和输出的行数，M 是输出列数，C 是每个输出元素要遍历的归约长度。** 源码还会把 C 称为 K；本文中的 C、K 指同一个归约维度。

| 名称 | 含义 | 在逻辑矩阵中的作用 |
|---|---|---|
| `N` | 输入行数；可能包含展平后的批次和序列维度 | 激活和输出的行数 |
| `C`，也称 `K` | 归约长度 | 每个输出元素需要累加多少个乘积 |
| `M` | 输出列数或输出特征数 | 输出的列数 |
| `A` | 输入激活的逻辑矩阵 | 形状为 $N\times C$ |
| `W` | 权重的二维访问视图 | GemmWS 生成权重地址时按 $M\times C$ 索引 |
| `Y` | 输出的逻辑矩阵 | 形状为 $N\times M$ |

因此，本文用 $AW^T$ 表示逻辑计算，其中 $W^T$ 是形状为 $C\times M$ 的转置视图。这个写法用于说明分块范围，不表示 ONNXim 执行了一次数值转置。权重地址生成代码构造的是 M/C 二维下标；分块算法则采用 I=N、J=M、K=C 的命名。


## 1.2 Mapping 中的三个字段分别回答三个问题

下面是一条**手工指定**的 Mapping，后文一直使用它：

~~~text
[T] N32 C64 M32 - [O] N2 C2 M2 - [I] N16 C32 M16
~~~

`Mapping` 是保存循环划分方案的结构体；变量 `mapping` 表示当前 GEMM 使用的那个 Mapping 对象。这条字符串的三个层级对应：

| 字符串层级 | 源码字段 | 回答的问题 | 本例数值及单位 |
|---|---|---|---|
| `[T]` | `mapping.total_loop` | 整个 GEMM 有多大？ | N=32、C=64、M=32，均为元素长度 |
| `[O]` | `mapping.tile_out_loop` | 每个方向生成多少个外层 Tile？ | N=2、C=2、M=2，均为 Tile 数 |
| `[I]` | `mapping.tile_in_loop` | 一个规则 Tile 在每个方向标称覆盖多少元素？ | N=16、C=32、M=16，均为元素长度 |

`total_loop`、`tile_out_loop`、`tile_in_loop` 都使用 `Mapping::LoopCounts` 结构。这里的 `.N`、`.C`、`.M` 是三个维度的字段名；同样的字段名放在不同对象下，单位和含义不同。

例如：

- `mapping.total_loop.M=32`：完整 GEMM 有 32 个输出列。
- `mapping.tile_out_loop.M=2`：M 方向有两个调度 Tile。
- `mapping.tile_in_loop.M=16`：一个规则 Tile 标称覆盖 16 个输出列。

主例各维度恰好整除，所以有 $32=2\times16$、$64=2\times32$、$32=2\times16$。一般情况下，最后一个 Tile 可以只使用标称范围的一部分；**源码不会因此改写 `tile_in_loop`，而是在生成指令时裁剪有效长度。**



## 1.3 Tile 是调度任务，包含一次 C 维归约阶段

Core 是模拟的 NPU 计算核心；`GemmWS` 是为 Weight Stationary（权重驻留）核心生成 GEMM 任务的算子实现。`Tile` 是 Core 接收的调度任务对象，持有一组 `Instruction`。`Instruction` 是指令描述结构，保存操作类型、地址、尺寸和其他执行元数据。一个 Tile 不是一条计算指令，也不一定已经覆盖一个输出区域的全部归约长度。

主例中的一个 Tile 标称处理 N16/C32/M16，即：

| 当前 Tile 涉及的数据 | 逻辑范围 | 为什么需要它 |
|---|---|---|
| 激活 | $16\times32$ | 为 16 个输入行提供当前 32 个归约项 |
| 权重 | $16\times32$ | 为 16 个输出列提供当前 32 个归约项 |
| 输出贡献或部分和 | $16\times16$ | 当前 C 分片对这一输出区域的贡献 |

固定一个 N/M 输出区域时，全局 C=64 被切成两个 C Tile。用 `P_0`、`P_1` 分别表示前、后 32 个归约项的逻辑贡献，则：

$$
Y_{\mathrm{区域}}=P_0+P_1.
$$

这里的 $Y_{\mathrm{区域}}$ 是这个 $16\times16$ 输出区域的完整结果；$P_0$、$P_1$ 是解释用的数学记号，源码没有保存这样的数值矩阵对象。

因此，同一个 N/M 输出区域对应：

~~~text
C Tile 0：处理 C=0…31，建立部分和
C Tile 1：处理 C=32…63，继续归约，并生成最终写回指令
~~~

主例共有 $2_N\times2_M=4$ 个输出区域，每个区域需要 2 个 C 阶段，所以共有 8 个代码 Tile。**数输出区域时只乘 N/M；数调度 Tile 时还要乘 C。**


## 1.4 Scratchpad entry 和 Accumulator entry 表示什么

一个 Tile 要先让输入可用，再发射计算，最后让输出可写回。entry（条目）是一个数据块的状态记录，并不等于一个矩阵元素。ONNXim 使用两类片上 entry 记录这些状态：

| 对象 | 所属片上存储 | 在当前 GEMM 路径中的用途 |
|---|---|---|
| **Scratchpad entry** | 普通 Scratchpad，由 `Sram` 建模 | 记录一个激活块或权重块是否已完成搬入，能否被计算指令使用 |
| **Accumulator entry** | 累加存储，也由 `Sram` 建模 | 记录一个输出微块是否已分配、还有多少已发射并登记的计算贡献未完成、是否可写回 |

`Sram` 是片上存储的模拟类，`SramEntry` 是它内部的 entry 元数据结构。需要理解的成员如下：

| entry 成员 | 含义 |
|---|---|
| `valid` | 就绪标志；未完成计数归零后为 true |
| `size` | 为该 entry 预留的请求粒度数量 |
| `remain_req_count` | 尚未完成的事件数量；搬入时对应读响应，普通 GEMM 累加时对应计算贡献 |
| `timestamp` | 记录访问或创建时的核心周期，用于跟踪状态 |

entry 通过**局部地址标签和缓冲编号**查找。指令中的 `spad_id` 是 Scratchpad 的缓冲编号，`accum_spad_id` 是 Accumulator 的缓冲编号；同一个地址标签可以存在于不同缓冲中。

这些 entry **没有保存真实的矩阵元素数组**。Scratchpad entry 就绪，表示它对应的搬入请求已经完成；Accumulator entry 的计数，则用于表达同一输出标签上已发射的计算贡献是否完成。源码的 `count_up()` 增加未完成计数，`fill()` 减少计数并在归零时设置 `valid`。计数不会预先登记尚未发射的后续 K/C 指令，因此 `valid` 只表示当前已登记事件完成。这就是本文所说的 ONNXim 依赖与时序抽象。

容量也按元数据记账：entry 的占用字节数为 `size × dram_req_size`，其中 `dram_req_size` 是配置中的单个 DRAM 请求字节数。它不能直接解释为 entry 内保存了多少个真实矩阵元素。


## 1.5 一个 Tile 用哪些指令连接这些对象

| 指令名 | 在当前 GemmWS 路径中的作用 | 与 entry 的关系 |
|---|---|---|
| **`MOVIN`** | 从 DRAM 搬入激活或权重；本文主例不含 Bias | 建立目标 Scratchpad entry，登记读请求，响应全部完成后可用 |
| **`GEMM_PRELOAD`** | 发射一条带预装载时序的 GEMM 计算指令 | 检查激活和权重 entry 就绪，向目标 Accumulator entry 登记一次计算贡献 |
| **`MOVOUT`** | 把最终输出微块写回 DRAM | 等待对应 Accumulator entry 就绪，再生成写请求 |

本文的 **PRELOAD 是 `GEMM_PRELOAD` 的简称**。`Opcode` 是源码中的指令类型枚举，当前枚举没有独立的 `PRELOAD` 项。这里的一条 `GEMM_PRELOAD` 同时走 GEMM 计算路径，并带有预装载相关的启动时序；不能把它数成“一条只搬权重、不计算的指令”。

还要区分：`MOVIN` 把数据从 DRAM 搬到片上 entry；`GEMM_PRELOAD` 使用已经就绪的片上 entry 进入阵列计算模型。两者属于不同步骤。

至此可以把三层关系连起来：

| 层次 | 主例范围 | 对应对象 |
|---|---|---|
| 完整 GEMM | N32/C64/M32 | `total_loop` |
| 一个调度 Tile | N16/C32/M16 | `Tile`、`tile_in_loop` |
| 一条完整阵列计算指令 | N8/K8/M8 | `Instruction`、`GEMM_PRELOAD` |



# 2. 自动 Mapping 怎样根据阵列和容量选取 Tile

## 2.1 先区分查表和自动生成

`MappingTable` 是保存 Mapping 的查找表。`MappingTable::at()` 先查询显式提供或此前缓存的方案；只有缺失时才调用 `fallback_mapping()`。对于没有卷积空间循环的情况，即查询键的 P/Q/S/R 字段均为 1，回退路径调用 `gemm_mapping()`。

这里的 P、Q、S、R 是 `LoopCounts` 中用于卷积的循环字段，本文 GEMM 将它们置为 1，不参与后面的 N/C/M 数量推导。**已经提供手工 Mapping 时，不会先运行自动算法，再强制改成手工尺寸。**

自动算法涉及的输入变量是：

| 变量或配置字段 | 含义 |
|---|---|
| `_config` | `SimulationConfig` 硬件配置对象 |
| `key` | 本次查询的 `Mapping::LoopCounts` 键，包含全局长度和目标配置编号 |
| `key.N / key.M / key.C` | 完整 GEMM 的 N/M/C 元素长度 |
| `key.target_core` | 选择哪一个 Core 的配置生成 Mapping |
| `core_config` | `_config` 中保存各 Core 配置的数组 |
| `core_height / core_width` | 所选 Core 的阵列高度和宽度 |
| `spad_size / accum_spad_size` | 所选 Core 的 Scratchpad、Accumulator 总容量；配置按 KiB 数值给出 |
| `precision` | 输入、权重等逻辑元素的字节数 |
| `num_cores` | 配置中的 Core 总数 |
| `KB` | 源码宏，展开为乘以 1024，用来将容量配置换成字节 |

`target_core` 在生成 Mapping 时选择用于计算容量和阵列尺寸的硬件配置；第 5 节中的 `core_id` 则记录某个 Tile 被分配给哪个 Core，两者不能混为同一用途。还要注意当前 `Mapping::LoopCounts` 的比较键不包含 `target_core`：如果不同 Core 配置下查询相同的 N/C/M 形状，缓存可能复用先前生成的 Mapping；因此这里的“按 `target_core` 选择配置”是生成阶段的行为，不能理解为缓存一定按 Core 隔离。


## 2.2 把全局长度补齐到阵列基础块的边界

源码取 `dim` 为所选 Core 的 `core_height`，并断言 `core_height==core_width`，因此这条自动 GEMM Mapping 路径按方阵处理。随后使用：

~~~cpp
dim_I = key.N;
dim_J = key.M;
dim_K = key.C;

dim_I_padded = (dim_I / dim + (dim_I % dim != 0)) * dim;
dim_J_padded = (dim_J / dim + (dim_J % dim != 0)) * dim;
dim_K_padded = (dim_K / dim + (dim_K % dim != 0)) * dim;
~~~

这些变量的单位都为元素：

| 变量 | 含义 |
|---|---|
| `dim` | 阵列边长；主例为 8 |
| `dim_I` | GEMM 全局 N 长度 |
| `dim_J` | GEMM 全局 M 长度 |
| `dim_K` | GEMM 全局 C/K 长度 |
| `dim_I_padded` | 将 N 向上补齐到 `dim` 倍数后的长度 |
| `dim_J_padded` | 将 M 向上补齐到 `dim` 倍数后的长度 |
| `dim_K_padded` | 将 C 向上补齐到 `dim` 倍数后的长度 |

式中的整数除法先求完整块数；`%` 求余；`!=0` 在有尾部时贡献 1，使块数向上取整。比如 N=50、`dim=8`，得到 $7\times8=56$ 的补齐长度。

补齐后的长度用于规划规则 Tile。原始全局长度仍是 50，实际有效工作范围还要在指令生成阶段按全局边界裁剪；补齐不表示增加了有效矩阵行。



## 2.3 容量先换成“行数”，再换成“基础块数量”

自动算法先为双缓冲预算空间。下面摘录容量公式，省略变量类型声明：

~~~cpp
kNumBuffers = 2;

max_spad_rows =
    (_config.core_config[key.target_core].spad_size KB) /
    (dim * _config.precision * kNumBuffers);

max_acc_rows =
    (_config.core_config[key.target_core].accum_spad_size KB) /
    (dim * 4 * kNumBuffers);
~~~

`kNumBuffers` 是固定为 2 的缓冲数量。`max_spad_rows` 是按每行 `dim×precision` 字节计算的单缓冲 Scratchpad 行数预算；`max_acc_rows` 是按每行 `dim×4` 字节计算的单缓冲 Accumulator 行数预算。

这里的 **4 是 Accumulator 采用的每元素 4 B 累加精度，即 FP32**。

接下来，Scratchpad 的单缓冲预算再分给激活和权重两类数据：

~~~cpp
db_partitions_rows = max_spad_rows / 2;
db_mats_in_partition = db_partitions_rows / dim;
db_mats_in_acc = max_acc_rows / dim;
db_max_tile_i_j = (uint32_t)sqrt(db_mats_in_acc);
db_max_tile_k = db_mats_in_partition / db_max_tile_i_j;
~~~

| 变量 | 单位 | 在算法中的作用 |
|---|---|---|
| `db_partitions_rows` | 行数 | 激活或权重各自获得的行数预算 |
| `db_mats_in_partition` | 基础块数 | 一个激活或权重预算能放多少个 `dim×dim` 块 |
| `db_mats_in_acc` | 基础块数 | 一个 Accumulator 缓冲预算能放多少个 `dim×dim` 输出块 |
| `db_max_tile_i_j` | 单边基础块数 | 从输出块容量取平方根，得到 I/J 方向的候选块数上限 |
| `db_max_tile_k` | 单边基础块数 | 用输入块容量除以 I/J 方向块数，得到 K 方向的候选块数上限 |

`sqrt()` 求平方根；转换为无符号整数类型 `uint32_t` 时截去小数部分。因此最后两个变量是**基础块数量**，乘以 `dim` 才是元素长度。例如 `db_max_tile_i_j=5`、`dim=8`，候选 I/J 长度是 40 个元素。

这里两个“除以 2”的用途不同：第一次通过 `kNumBuffers` 为两份缓冲预算空间，第二次给激活和权重各分一半算法预算。后者是 Mapping 的选块规则；运行时 `Sram` 没有因此创建四个 bank，也没有独立强制激活、权重各自只能占半个缓冲。

可以把容量意图近似写成：

$$
p(T_NT_C+T_MT_C)\le \frac{S_{\mathrm{SPAD}}}{2},
\qquad
4T_NT_M\le \frac{S_{\mathrm{ACC}}}{2}.
$$

其中 $p=\mathrm{precision}$，单位为 B/元素；$T_N,T_C,T_M$ 分别表示单个 Tile 的标称 N/C/M 长度；$S_{\mathrm{SPAD}}$、$S_{\mathrm{ACC}}$ 分别是 Scratchpad、Accumulator 的**总容量字节数**。这两个 S 是容量符号，与前面的卷积 S 字段无关。

> 上次看到这儿。
## 2.4 从候选长度算出外层 Tile 数，再调整内部长度

`tile_I`、`tile_J`、`tile_K` 分别是 I/J/K 方向的外层 Tile 数，也就是 N/M/C 方向的 Tile 数。初选公式为：

~~~cpp
tile_I = std::min(dim_I_padded / dim,
                 ceil_div(dim_I, db_max_tile_i_j * dim));
tile_J = std::min(dim_J_padded / dim,
                 ceil_div(dim_J, db_max_tile_i_j * dim));
tile_K = std::min(dim_K_padded / dim,
                 ceil_div(dim_K, db_max_tile_k * dim));
~~~

`std::min()` 取两个值的较小者；`ceil_div()` 表示整数向上除法，即对正整数求 $\lceil\text{分子}/\text{分母}\rceil$。每个公式的第一项是该维度补齐后包含的阵列基础块数，第二项是按容量候选长度覆盖全局维度所需的 Tile 数。

随后源码计算 `num_tiles=tile_I×tile_J`。这里的 `num_tiles` **只数独立输出区域，不包含 K/C 归约阶段**。算法按 `_config.num_cores` 调整 I/J 的切分：

- 当输出区域数少于 Core 数时，局部变量 `increase_tile` 是 $\lceil\mathrm{num\_cores}/\mathrm{num\_tiles}\rceil$，用作乘数。
- 当输出区域数不能被 Core 数整除时，另一个同名局部变量 `increase_tile` 是余数，用作加数。
- 两个分支只在 I 或 J 严格更大、且该维度大于 Core 数时调整相应方向。它们是启发式规则，不保证任意形状都能完全均衡。

接着计算 `inner_I`、`inner_J`、`inner_K`，它们分别是单个 Tile 的标称 N/M/C 元素长度。以 I 方向的实际表达式为例：

~~~cpp
inner_I = ceil_div(dim_I_padded, tile_I);
inner_I -= inner_I & (dim) - 1;
tile_I = ceil_div(dim_I, inner_I);
~~~

J、K 方向执行同样步骤。第二行按运算符优先级等价于减去 `inner_I & (dim-1)`；`&` 是按位与。**只有 `dim` 为 2 的幂时，这个写法才等价于向下对齐到 `dim` 的倍数。** 对齐后的内部长度可能变小，因此第三行重新计算最终外层 Tile 数。

最终写回：

~~~cpp
mapping.total_loop = {dim_I, dim_K, dim_J, 1, 1, 1, 1};
mapping.tile_out_loop = {tile_I, tile_K, tile_J, 1, 1, 1, 1};
mapping.tile_in_loop = {inner_I, inner_K, inner_J, 1, 1, 1, 1};
~~~

`LoopCounts` 的字段顺序是 N、C、M、S、R、Q、P。因此初始化列表的前三项按 **I/K/J → N/C/M** 保存；后面四个 1 是本 GEMM 的 S/R/Q/P 单位循环。`mapping` 是最终生成并缓存的 Mapping 对象。



## 2.5 用主例的形状说明：自动结果与手工结果可以不同

取单 Core、8×8 阵列、Scratchpad 总容量 64 KiB、Accumulator 总容量 16 KiB、`precision=1 B`。按上述算法：

| 中间量 | 计算 | 数值 |
|---|---|---:|
| `max_spad_rows` | $65536/(8\times1\times2)$ | 4096 |
| `max_acc_rows` | $16384/(8\times4\times2)$ | 256 |
| `db_partitions_rows` | $4096/2$ | 2048 |
| `db_mats_in_partition` | $2048/8$ | 256 |
| `db_mats_in_acc` | $256/8$ | 32 |
| `db_max_tile_i_j` | $\lfloor\sqrt{32}\rfloor$ | 5 |
| `db_max_tile_k` | $\lfloor256/5\rfloor$ | 51 |

因此候选 I/J 长度为 $5\times8=40$，候选 K 长度为 $51\times8=408$。对于 N32/C64/M32，三个初选外层数量均为 1；单 Core 下无需增加输出区域。内部长度得到 N32/C64/M32，最终自动方案是：

~~~text
[T] N32 C64 M32 - [O] N1 C1 M1 - [I] N32 C64 M32
~~~

逻辑上，完整激活和权重合计 $32\times64+32\times64=4096$ B，按 4 B 估计的输出部分和为 $32\times32\times4=4096$ B，都在单缓冲预算内。

而本文选择：

~~~text
[T] N32 C64 M32 - [O] N2 C2 M2 - [I] N16 C32 M16
~~~

这是为了展示**跨 C Tile 归约和 Tile 内再次切分**而显式选定的手工方案。后文的 8 个 Tile 来自这条手工 Mapping，不是由前面的容量配置自动推导出来的。

# 3. 按手工 Mapping 逐步推导 Tile、PRELOAD 和 MAC 数

## 3.1 固定本例条件，避免混用不同单位

本例使用单 Core、无 Bias 的 GemmWS 路径：

| 参数 | 数值 | 含义 |
|---|---:|---|
| `total_loop.N/C/M` | 32/64/32 | 全局 N/C/M 元素长度 |
| `tile_out_loop.N/C/M` | 2/2/2 | 各方向外层 Tile 数 |
| `tile_in_loop.N/C/M` | 16/32/16 | 每个规则 Tile 的标称元素长度 |
| `core_height/core_width` | 8/8 | 阵列高度、宽度 |
| `precision` | 1 B | 每个输入、权重等逻辑元素的字节数 |
| `dram_req_size` | 32 B | 一个 DRAM 请求的字节数 |
| `has_bias` | false | 此例不生成 Bias 搬入指令 |

后文用 $H=8$ 表示 `core_height`，即生成器采用的阵列切分步长。`has_bias` 是 GemmWS 的 Bias 开关；显式关闭它，才能得到本文第 6 节所列的无 Bias 指令序列。

## 3.2 第一步：为什么有 8 个调度 Tile

沿三个维度分别切分：

| 维度 | 全局长度 | 单 Tile 长度 | Tile 数 | 两个 Tile 的有效范围 |
|---|---:|---:|---:|---|
| N | 32 | 16 | 2 | 0–15；16–31 |
| M | 32 | 16 | 2 | 0–15；16–31 |
| C | 64 | 32 | 2 | 0–31；32–63 |

生成器对每一个 N/M/C Tile 索引组合都创建任务，所以：

$$
\text{调度 Tile 数}=2_N\times2_M\times2_C=8.
$$

也可以先数输出区域：$2_N\times2_M=4$。每个 $16\times16$ 输出区域需要前、后两个 C 阶段，故 $4\times2=8$。

**一个 Tile 只处理当前 32 个归约项；两个 C Tile 才共同完成该输出区域的全部 64 个归约项。**

## 3.3 第二步：为什么每个 Tile 有 16 条 PRELOAD

一个 Tile 的范围是 N16/C32/M16。计算生成器用阵列高度 $H=8$ 切分 M、N 和 K/C：

| 循环变量 | 含义 | 本例取值 | 有效计算长度 | 次数 |
|---|---|---|---|---:|
| `Ms` | 当前 Tile 内的 M 起始偏移 | 0、8 | 每次 `m_loop=8` | 2 |
| `Cs` | 当前 Tile 内的 C 子块起始偏移 | 0 | `c_in_loop=32` | 1 |
| `Ns` | 当前 Tile 内的 N 起始偏移 | 0、8 | 每次 `n_loop=8` | 2 |
| `c_iter` | 当前 C 子块内的 K 微块起始偏移 | 0、8、16、24 | 每次 `c_iter_size=8` | 4 |

`m_loop`、`n_loop` 是当前 M/N 微块的有效长度；`c_in_loop` 是当前 C 子块的有效长度；`c_iter_size` 是其中一条 PRELOAD 的有效 K 长度。它们都是**长度**；`Ms`、`Ns`、`Cs`、`c_iter` 都是**起始偏移**，不能混用。

这里 `Cs` 的步长是整个内部 C 长度 32，所以只有 `Cs=0`。真正把 32 个归约项拆成四段的是 `c_iter`：

~~~text
c_iter=0 ：当前 C 子块的第 0…7 项
c_iter=8 ：当前 C 子块的第 8…15 项
c_iter=16：当前 C 子块的第 16…23 项
c_iter=24：当前 C 子块的第 24…31 项
~~~

每个 Ms/Ns 组合需要四条 PRELOAD：

| Ms | Ns | 对应输出微块的 Tile 内行、列范围 | PRELOAD 条数 |
|---:|---:|---|---:|
| 0 | 0 | N=0–7，M=0–7 | 4 |
| 0 | 8 | N=8–15，M=0–7 | 4 |
| 8 | 0 | N=0–7，M=8–15 | 4 |
| 8 | 8 | N=8–15，M=8–15 | 4 |

因此：

$$
\text{每 Tile 的 PRELOAD 数}
=2_{M_s}\times1_{C_s}\times2_{N_s}\times4_{c\_iter}
=16.
$$

这 16 条指令是一个 Tile 中的计算任务，复用同一个 Core 的阵列模型；表中的四个输出微块不表示新建了四个物理阵列。



## 3.4 第三步：为什么全 GEMM 有 128 条 PRELOAD

主例三个全局长度都被内部长度整除，每个 Tile 都是完整 N16/C32/M16，没有边界缩短。因此每个 Tile 都生成 16 条：

$$
\text{全 GEMM 的 PRELOAD 数}=8\times16=128.
$$

这个 128 只统计 `GEMM_PRELOAD`，没有把 `MOVIN`、`MOVOUT` 加进去。一条搬运指令还可以生成多个 DRAM 请求，所以“指令数”和“请求数”也不能互换。

## 3.5 第四步：为什么一条完整 PRELOAD 对应 512 MAC

**MAC（Multiply-Accumulate）**指一次“乘法并累加到部分和”的逻辑工作：

$$
\text{部分和}\leftarrow\text{部分和}+\text{激活}\times\text{权重}.
$$

一次 MAC 包含一次乘法和一次加法。本文数的是 MAC 次数，既不是元素字节数，也不是模拟周期数。

计算指令中的三个尺寸字段为：

| 字段 | 本例数值 | 含义 |
|---|---:|---|
| `tile_m` | 8 | 当前指令覆盖的输出列数，来自 `m_loop` |
| `tile_k` | 8 | 当前指令覆盖的归约项数，来自 `c_iter_size` |
| `tile_n` | 8 | 当前指令覆盖的输入/输出行数，来自 `n_loop` |

一条指令更新 $8\times8=64$ 个逻辑输出位置，每个位置计算当前 8 个 K 项的贡献，所以：

$$
\text{每条 PRELOAD 的 MAC 数}
=\mathrm{tile_n}\times\mathrm{tile_m}\times\mathrm{tile_k}
=8\times8\times8=512.
$$

这里的三个 8 来自三个不同维度：8 行输出、8 列输出、每个位置的 8 个归约项。**8×8 阵列大小本身只有两个维度，不能省略时间上处理的 N 行，也不能把一次指令理解成只有 64 次 MAC。** 源码通过 `tile_n` 记录这一批 N 行的数量。

## 3.6 第五步：从一条指令推到一个 Tile，再推到全 GEMM

一个 Tile 有 16 条完整 PRELOAD，每条 512 MAC：

$$
\text{每 Tile 的 MAC 数}=16\times512=8192.
$$

直接从 Tile 的逻辑范围核对也是：

$$
16_N\times16_M\times32_C=8192.
$$

全 GEMM 有 8 个 Tile，或等价地有 128 条完整 PRELOAD：

$$
\text{全 GEMM 的 MAC 数}
=8\times8192
=128\times512
=65536.
$$

再用全局长度核对：

$$
32_N\times32_M\times64_C=65536.
$$

整个推导可以用一张表检查：

| 要数的对象 | 推导 | 结果 |
|---|---|---:|
| 最终输出区域 | $2_N\times2_M$ | 4 个 |
| 调度 Tile | $4\times2_C$ | 8 个 |
| 每 Tile 的 PRELOAD | $2_{M_s}\times2_{N_s}\times4_{c\_iter}$ | 16 条 |
| 全 GEMM 的 PRELOAD | $8\times16$ | 128 条 |
| 每条完整 PRELOAD 的工作量 | $8_N\times8_M\times8_K$ | 512 MAC |
| 每 Tile 的工作量 | $16\times512$ | 8192 MAC |
| 全 GEMM 的工作量 | $128\times512$ | 65536 MAC |

这些乘积吻合，说明该手工划分在逻辑范围上覆盖了全部工作。第 5、6 节将把这些数字与实际生成循环一一对应。

# 4. GemmWS 怎样创建这 8 个 Tile

## 4.1 外层循环的变量是 Tile 索引

`GemmWS::initialize_tiles()` 查到 `mapping` 后，按 N→M→C 的顺序创建 Tile。下面保留循环和 Core 轮转语句，省略对象构造细节：

~~~cpp
int core_id = -1;
for (uint32_t N = 0; N < mapping.tile_out_loop.N; N++) {
  for (uint32_t M = 0; M < mapping.tile_out_loop.M; M++) {
    for (uint32_t C = 0; C < mapping.tile_out_loop.C; C++) {
      if (C == 0) {
        core_id = (core_id + 1) % _config.num_cores;
      }
      // 创建当前索引组合的 Tile，并生成其指令。
    }
  }
}
~~~

| 片段中的变量 | 含义 |
|---|---|
| `mapping` | 当前 GEMM 的 Mapping 对象 |
| `mapping.tile_out_loop.N/M/C` | N/M/C 方向的外层循环上界，单位为 Tile 数 |
| 循环变量 `N`、`M`、`C` | 当前 N/M/C 方向的 Tile 索引，从 0 开始；此处不是全局元素坐标 |
| `core_id` | 分配给当前输出区域的 Core 编号 |
| `_config.num_cores` | 可轮转的 Core 总数 |
| `uint32_t` | 循环索引使用的无符号整数类型，不是维度变量 |

`core_id` 初值为 −1，第一次遇到 `C==0` 时先加 1，因此第一个输出区域分配到 Core 0。后面的 C Tile 不再推进编号，**同一 N/M 输出区域的归约阶段被分配给同一个 Core**。本例只有一个 Core，所有 Tile 的 `core_id` 都为 0。

这段 GemmWS 循环直接固定了 N→M→C 的创建顺序。



## 4.2 Tile 中保存的是哪些索引和归约标记

对象构造中的相关字段摘录如下；省略的其他字段不参与本例数量推导：

~~~cpp
.batch = N,
.M = M,
.C = C,
.accum = C != 0,
.core_id = core_id
~~~

| Tile 字段 | 保存的值 | 在这里的含义 |
|---|---|---|
| `batch` | 外层循环 `N` | N 方向 Tile 索引；不一定是原始 ONNX Batch 编号 |
| `M` | 外层循环 `M` | M 方向 Tile 索引 |
| `C` | 外层循环 `C` | C/K 归约 Tile 索引 |
| `accum` | `C!=0` 的布尔值 | false 表示首个 C 阶段；true 表示后续归约阶段 |
| `core_id` | 局部变量 `core_id` | 该任务分配的 Core 编号 |

源码中的 `tile` 是指向当前 Tile 对象的指针，`tile->M` 等写法表示读取该对象的字段。`tile->accum` 用于表达后续 C 阶段延续累加存储的意图；它不是数值加法，也不是“该 Tile 已经得到完整结果”的标记。

生成后，Tile 进入 Operation 的 `_tiles` 队列；`initialize_instructions()` 为其填入 `instructions` 指令队列。若指令队列为空，源码会移除这个 Tile。因此，任意手工 Mapping 的外层数量乘积不一定都等于最终非空任务数；主例的 8 个 Tile 均非空。



## 4.3 从 Tile 索引换算成全局起点

进入 `GemmWS::initialize_instructions()` 后，代码首先计算：

~~~cpp
int tout_m_offset = tile->M * mapping.tile_in_loop.M;
int tout_c_offset = tile->C * mapping.tile_in_loop.C;
int tout_n_offset = tile->batch * mapping.tile_in_loop.N;
~~~

| 变量或表达式 | 含义及单位 |
|---|---|
| `tile` | 当前 Tile 的指针 |
| `tile->M / tile->C / tile->batch` | M/C/N 方向的 Tile 索引 |
| `mapping.tile_in_loop.M/C/N` | 各方向规则 Tile 的标称元素长度 |
| `tout_m_offset` | 当前 Tile 在完整 M 维的起点，单位为元素 |
| `tout_c_offset` | 当前 Tile 在完整 C 维的起点，单位为元素 |
| `tout_n_offset` | 当前 Tile 在完整 N 维的起点，单位为行 |

例如主例的 `tile->M=1`，所以 `tout_m_offset=1×16=16`；`tile->C=1`，所以 `tout_c_offset=1×32=32`。这是“第几个 Tile”转换成“从第几个元素开始”的步骤。

## 4.4 主例的完整创建顺序

| 创建序号 | `(Tile.batch, Tile.M, Tile.C)` | 全局 N 范围 | 全局 M 范围 | 全局 C 范围 | `accum` | 生成 MOVOUT |
|---:|---|---|---|---|---|---|
| 0 | (0,0,0) | 0–15 | 0–15 | 0–31 | false | 否 |
| 1 | (0,0,1) | 0–15 | 0–15 | 32–63 | true | 是 |
| 2 | (0,1,0) | 0–15 | 16–31 | 0–31 | false | 否 |
| 3 | (0,1,1) | 0–15 | 16–31 | 32–63 | true | 是 |
| 4 | (1,0,0) | 16–31 | 0–15 | 0–31 | false | 否 |
| 5 | (1,0,1) | 16–31 | 0–15 | 32–63 | true | 是 |
| 6 | (1,1,0) | 16–31 | 16–31 | 0–31 | false | 否 |
| 7 | (1,1,1) | 16–31 | 16–31 | 32–63 | true | 是 |

每两行是同一个输出区域的两个归约阶段。这张表表达**任务与指令的生成顺序**，不要求搬运、计算、写回在执行时间上逐行串行完成。

## 4.5 为什么只有后一个 C Tile 生成 MOVOUT

写回阶段外面的条件是：

~~~cpp
if (tout_c_offset + mapping.tile_in_loop.C >= mapping.total_loop.C)
~~~

其中 `tout_c_offset` 是当前 C Tile 的全局起点，`mapping.tile_in_loop.C` 是标称 C 长度，`mapping.total_loop.C` 是全局归约长度。

主例两个 C Tile 的判断分别是：

| C Tile 索引 | 条件代入 | 是否生成写回 |
|---:|---|---|
| 0 | $0+32\ge64$ | 否 |
| 1 | $32+32\ge64$ | 是 |

所以首个 C Tile 只生成搬入和计算，后一个 C Tile 在计算指令之后再生成输出微块的 MOVOUT。这个条件判断的是**标称 C 范围是否覆盖全局末端**；最后一个 C Tile 即使有效长度短于标称长度，也会进入写回阶段。


# 5. Tile 内怎样生成指令，以及边界怎样裁剪

## 5.1 先区分局部偏移、全局坐标和有效长度

第 5 节算出的 `tout_*_offset` 是 Tile 全局起点。接下来生成器在 Tile 内继续分块：

| 变量 | 类别 | 精确含义 |
|---|---|---|
| `loop_size` | 步长 | 取所选 Core 的 `core_height`，用于推进 Ms、Ns |
| `cloop_size` | 步长 | 等于 `mapping.tile_in_loop.C`，用于推进 Cs |
| `Ms` | Tile 内偏移 | 当前 M 微块在 Tile 内的起点 |
| `Ns` | Tile 内偏移 | 当前 N 微块在 Tile 内的起点 |
| `Cs` | Tile 内偏移 | 当前 C 子块在 Tile 内的起点 |
| `M_offset` | 全局坐标 | `tout_m_offset+Ms` |
| `N_offset` | 全局坐标 | `tout_n_offset+Ns` |
| `C_offset` | 全局坐标 | `tout_c_offset+Cs` |
| `m_loop` | 有效长度 | 当前 M 微块实际覆盖的输出列数 |
| `n_loop` | 有效长度 | 当前 N 微块实际覆盖的行数 |
| `c_in_loop` | 有效长度 | 当前 C 子块在全局边界内的归约长度 |
| `c_iter` | C 子块内偏移 | 当前 K 微块相对于该 C 子块的起点 |
| `c_iter_size` | 有效长度 | 当前一条 PRELOAD 实际覆盖的 K 项数 |

这些偏移和长度都按元素计数，不是字节地址。对于正常的正长度 Mapping，`cloop_size` 等于整个内部 C 长度，所以 `Cs` 只取 0；`c_in_loop` 此时就是当前 C Tile 的有效 C 长度。`c_iter` 才负责把它进一步切成阵列计算指令。

例如主例中，后一个 C Tile 的 `tout_c_offset=32`、`Cs=0`，所以 `C_offset=32`；该 Tile 的 `c_iter=8` 在逻辑范围上对应全局 C=40 开始的 K 微块。



## 5.2 搬入阶段：为什么激活和权重的 MOVIN 数不同于 PRELOAD 数

以下是保留源码循环结构的简化片段；省略有效长度计算、地址枚举和可选 Bias 路径，注释表示在对应位置生成指令：

~~~cpp
for (int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size) {
  for (int Cs = 0; Cs < mapping.tile_in_loop.C; Cs += cloop_size) {
    // 生成当前 Ms/Cs 权重块的一条 MOVIN。
    for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns += loop_size) {
      if (Ms == 0) {
        // 生成当前 Ns/Cs 激活块的一条 MOVIN。
      }
    }
  }
}
~~~

片段中的 `mapping`、`loop_size`、`cloop_size`、`Ms`、`Cs`、`Ns` 均按第 6.1 节定义：内部长度是循环上界，局部偏移按相应步长推进。

主例 `Ms=0,8`、`Ns=0,8`、`Cs=0`，因此一个无 Bias Tile 的搬入顺序是：

| 次序 | MOVIN 内容 | 当前 Tile 内逻辑形状 | 后续复用关系 |
|---:|---|---|---|
| 1 | `Ms=0` 的权重块 | M8×C32 | 被 Ns=0、8 两批行使用 |
| 2 | `Ns=0` 的激活块 | N8×C32 | 被 Ms=0、8 两组输出列使用 |
| 3 | `Ns=8` 的激活块 | N8×C32 | 被 Ms=0、8 两组输出列使用 |
| 4 | `Ms=8` 的权重块 | M8×C32 | 被 Ns=0、8 两批行使用 |

权重 MOVIN 位于 Ns 循环之外，所以不随 Ns 重复搬入；激活 MOVIN 仅在 `Ms==0` 时生成，所以不随 Ms 重复搬入。这是**片上 entry 的复用**。一个搬入块可以供多条计算指令使用，不需要每条 PRELOAD 都对应一条新 MOVIN。

每条 MOVIN 的 `src_addrs` 是 DRAM 请求地址列表，`dest_addr` 是目标 entry 的局部标签，`size` 是该地址列表的请求数。一个 MOVIN 因此可能展开成多个请求；源码不在这里保存矩阵数值。



## 5.3 计算与写回阶段怎样使用指令字段

搬入指令生成完后，计算阶段按 **Ms→Cs→Ns→c_iter** 枚举。对于固定的 Ms/Cs/Ns，K 切分循环是：

~~~cpp
for (int c_iter = 0; c_iter < c_in_loop;
     c_iter += _config.core_config[target_core].core_height) {
  // 计算 c_iter_size，并生成一条 GEMM_PRELOAD。
}
~~~

`c_iter` 是当前 C 子块内的归约偏移；`c_in_loop` 是这个子块的有效归约长度；`_config.core_config[target_core].core_height` 是推进步长。这里的 `target_core` 仍是选择配置的编号，`_config` 和 `core_config` 分别是配置对象和 Core 配置数组。

每次迭代生成的指令字段如下，省略压入指令队列的包装语句：

~~~cpp
Instruction{
    .opcode = Opcode::GEMM_PRELOAD,
    .dest_addr = out_sp_addr,
    .size = (uint32_t)n_loop,
    .compute_size = (uint32_t)n_loop,
    .src_addrs = std::vector<addr_type>{act_sp_addr, weight_sp_addr},
    .tile_m = static_cast<unsigned int>(m_loop),
    .tile_k = static_cast<unsigned int>(c_iter_size),
    .tile_n = static_cast<unsigned int>(n_loop)
}
~~~

| 展示的字段或变量 | 在这条计算指令中的含义 |
|---|---|
| `Instruction` | 一条指令的描述结构 |
| `opcode`、`Opcode::GEMM_PRELOAD` | 指令类型字段，以及本条指令的计算类型 |
| `out_sp_addr` | 当前输出微块的 Accumulator entry 标签，填入 `dest_addr` |
| `act_sp_addr` | 当前激活块的 Scratchpad entry 标签 |
| `weight_sp_addr` | 当前权重块的 Scratchpad entry 标签 |
| `src_addrs` | 本条计算的输入标签列表，顺序是激活、权重 |
| `size` | 目标 entry 首次创建时的申请粒度数；当前生成器填 `n_loop`。entry 已存在时只追加完成事件计数，不重新分配或扩容 |
| `compute_size` | 计算时序使用的参数；当前 GemmWS 同样填 `n_loop`，不是 MAC 总数 |
| `tile_m` | 有效 M 长度，填 `m_loop` |
| `tile_k` | 有效 K 长度，填 `c_iter_size` |
| `tile_n` | 有效 N 长度，填 `n_loop` |

`addr_type` 是源码的地址整数类型，`std::vector` 是地址列表容器；`uint32_t`、`unsigned int` 是整数类型，转换只用于把有效长度写入字段。

固定同一个 Ms/Ns 输出微块时，`out_sp_addr` 不随 `c_iter` 变化。因此主例四条 K 微块指令向同一个 Accumulator entry 登记贡献。计算发射前要求激活和权重 entry 就绪；完成贡献由 Accumulator 的未完成计数跟踪。**逻辑上是归约相加，C++ 实现中是同一标签上的依赖和完成事件。**

当前生成器也没有把 `c_iter` 加入 `act_sp_addr` 或 `weight_sp_addr` 的公式。它们指向此前搬入的整块 entry；K 微块范围主要通过循环次数和 `tile_k` 表达，而不是逐元素片上数组访问。不能据此构造模拟器实际上执行的逐元素数值轨迹。

最后一个 C Tile 还会按 Ms→Ns 为有效输出微块生成 MOVOUT；写回不再遍历 `c_iter`。主例有两个 Ms、两个 Ns，因此每个最后 C Tile 生成 $2\times2=4$ 条 MOVOUT，分别对应四个输出 entry。前一个 C Tile 不生成这些写回指令。

同名地址字段在不同指令中的角色也需要区分：

| 指令 | `src_addrs` | `dest_addr` | `size` |
|---|---|---|---|
| MOVIN | 要读取的 DRAM 请求地址列表 | 被填充的片上 entry 标签 | 列表中的请求数 |
| GEMM_PRELOAD | 激活、权重的片上 entry 标签 | 被更新的 Accumulator entry 标签 | 本路径填 `n_loop`，用于 entry 容量分配 |
| MOVOUT | 要写入的 DRAM 请求地址列表 | **被读取的片上输出 entry 标签** | 列表中的请求数 |

因此 MOVOUT 的 `dest_addr` 在实现中选择片上源 entry，不能按字段名理解为最终 DRAM 目的地址。Core 检查这个 entry 就绪后，才为 `src_addrs` 中的地址生成写请求。



## 5.4 m_loop、n_loop、c_in_loop、c_iter_size 分别怎样裁剪

裁剪的出发点是：标称内部尺寸可能覆盖到矩阵末端之外，但只能生成仍在全局范围内的工作。

为便于理解，设某维度全局长度为 $D$，规则 Tile 长度为 $T_D$，从 0 开始的 Tile 索引为 $q$，则 Tile 起点 $S_D$、剩余长度 $R_D$、概念有效长度 $E_D$ 为：

$$
S_D=qT_D,\qquad R_D=D-S_D,\qquad
E_D=\min(T_D,\max(0,R_D)).
$$

这里的 $D,T_D,q,S_D,R_D,E_D$ 都是教学记号，不是源码新增变量；$E_D$ 描述整个边界 Tile 的概念范围。M/N 方向还需要标称内部尺寸与阵列步长相容，实际代码限制见第 6.7 节。

**M/N 的实际裁剪发生在每个微块上。** 计算阶段的表达式为：

~~~cpp
int m_loop = M_offset + loop_size > mapping.total_loop.M
                 ? mapping.total_loop.M - M_offset
                 : loop_size;
if (m_loop <= 0) break;

int n_loop = N_offset + loop_size > mapping.total_loop.N
                 ? mapping.total_loop.N - N_offset
                 : loop_size;
if (n_loop <= 0) break;
~~~

`M_offset`、`N_offset` 是当前微块的全局起点；`loop_size` 是通常取 8 的阵列高度步长；`mapping.total_loop.M/N` 是对应维度的全局长度。三元表达式的含义是：

- 若当前起点加一个完整步长仍在全局范围内，就用完整 `loop_size`。
- 否则只使用全局末端之前剩下的元素数。
- 若已经没有合法元素，`m_loop<=0` 或 `n_loop<=0`，直接 `break`，停止当前方向的循环。

因此 `m_loop`、`n_loop` 描述**当前一个阵列微块**，不一定等于整个边界 Tile 的有效 M/N 长度。

**C 先裁剪到当前 C 子块，再切成 K 微块。** 第一层是：

~~~cpp
int c_in_loop = C_offset + cloop_size > mapping.total_loop.C
                    ? mapping.total_loop.C - C_offset
                    : cloop_size;
~~~

`C_offset` 是当前 C 子块的全局起点，`cloop_size` 是标称内部 C 长度，`mapping.total_loop.C` 是全局归约长度。由此得到的 `c_in_loop` 是当前 C 子块的有效长度，在本路径的正常映射下即当前 C Tile 的有效长度。

第二层对每个 `c_iter` 求：

$$
\mathrm{c\_iter\_size}
=\min(H,\ \mathrm{c\_in\_loop}-\mathrm{c\_iter}),
$$

其中 $H=\mathrm{core\_height}$，`c_in_loop−c_iter` 是当前 C 子块尚未处理的归约项数。源码用三元表达式实现同样的选择，并将结果写入 `tile_k`。

例如 $H=8$：

| `c_in_loop` | `c_iter` 的取值 | 各条指令的 `c_iter_size` |
|---:|---|---|
| 32 | 0、8、16、24 | 8、8、8、8 |
| 20 | 0、8、16 | 8、8、4 |
| 6 | 0 | 6 |

把四个变量放在一起看：

| 变量 | 裁剪层次 | 最后用于什么 |
|---|---|---|
| `m_loop` | 当前 M 微块与全局 M 边界 | PRELOAD 的 `tile_m`；加载和写回的 M 枚举 |
| `n_loop` | 当前 N 微块与全局 N 边界 | PRELOAD 的 `tile_n`、`size`、`compute_size`；加载和写回的 N 枚举 |
| `c_in_loop` | 当前 C 子块与全局 C 边界 | 搬入范围，以及 K 微块循环上界 |
| `c_iter_size` | 当前 K 微块与 `c_in_loop` 末端 | PRELOAD 的 `tile_k` |



## 5.5 仓库形状例子：边界 Tile 少生成完整微块

仓库 `Bert_512x512x1024` 测试显式提供：

~~~text
[T] N512 C512 M1024 - [O] N13 C2 M26 - [I] N40 C256 M40
~~~

取最后一个角落 Tile：`Tile.batch=12`、`Tile.M=25`、`Tile.C=1`。这里 `Tile.batch` 仍表示 N 方向 Tile 索引。

| 维度 | 全局长度 | 标称内部长度 | Tile 索引 | 全局起点 | 实际有效长度 |
|---|---:|---:|---:|---:|---:|
| N | 512 | 40 | 12 | $12\times40=480$ | $512-480=32$ |
| M | 1024 | 40 | 25 | $25\times40=1000$ | $1024-1000=24$ |
| C | 512 | 256 | 1 | $1\times256=256$ | 256 |

取 8×8 阵列，M 方向的实际迭代为：

| `Ms` | `M_offset` | `m_loop` | 行为 |
|---:|---:|---:|---|
| 0 | 1000 | 8 | 生成有效工作 |
| 8 | 1008 | 8 | 生成有效工作 |
| 16 | 1016 | 8 | 生成有效工作 |
| 24 | 1024 | 0 | break，不生成工作 |

N 方向为：

| `Ns` | `N_offset` | `n_loop` | 行为 |
|---:|---:|---:|---|
| 0 | 480 | 8 | 生成有效工作 |
| 8 | 488 | 8 | 生成有效工作 |
| 16 | 496 | 8 | 生成有效工作 |
| 24 | 504 | 8 | 生成有效工作 |
| 32 | 512 | 0 | break，不生成工作 |

C 方向仍有 `c_in_loop=256`，所以有 $256/8=32$ 个 K 微块。角落 Tile 的 PRELOAD 数为：

$$
3_M\times4_N\times32_K=384.
$$

完整 N40/M40/C256 Tile 则为：

$$
5_M\times5_N\times32_K=800.
$$

这里的差异是边界位置**少生成了 M/N 微块**；`mapping.tile_in_loop` 仍为 N40/C256/M40。这个例子没有短于 8 的合法计算微块，只是微块数量减少。



## 5.6 人工短尾例子：一条指令也可以小于完整 8×8×8

再选一个手工 Mapping，专门展示三个维度同时出现短尾：

~~~text
[T] N50 C70 M46 - [O] N3 C3 M2 - [I] N24 C32 M24
~~~

它不是仓库已有测试形状。取最后的 `Tile.batch=2`、`Tile.M=1`、`Tile.C=2`：

| 维度 | 全局长度 | 标称内部长度 | 起点计算 | 有效长度 |
|---|---:|---:|---|---:|
| N | 50 | 24 | $2\times24=48$ | 2 |
| M | 46 | 24 | $1\times24=24$ | 22 |
| C | 70 | 32 | $2\times32=64$ | 6 |

M 的三个合法微块分别是：

~~~text
Ms=0 ：M_offset=24，m_loop=8
Ms=8 ：M_offset=32，m_loop=8
Ms=16：M_offset=40，m_loop=6
~~~

N 的第一个微块 `Ns=0` 得到 `N_offset=48`、`n_loop=2`；下一次 `Ns=8` 得到起点 56，计算出的 `n_loop=-6`，因而 break。这个负数只是停止循环的中间结果，不会写入一条负长度指令。

C 得到 `c_in_loop=70−64=6`；`c_iter` 只取 0，所以 `c_iter_size=6`。最终生成：

| PRELOAD | `tile_m=m_loop` | `tile_k=c_iter_size` | `tile_n=n_loop` | 逻辑 MAC 数 |
|---:|---:|---:|---:|---:|
| 1 | 8 | 6 | 2 | $8\times6\times2=96$ |
| 2 | 8 | 6 | 2 | $8\times6\times2=96$ |
| 3 | 6 | 6 | 2 | $6\times6\times2=72$ |

合计：

$$
96+96+72=264=22_M\times6_C\times2_N\ \mathrm{MAC}.
$$

这个结果展示两层缩短：整个边界 Tile 的有效范围是 N2/C6/M22，而每条 PRELOAD 又使用当前微块的有效尺寸。物理阵列仍是 8×8，指令的有效工作范围变小。

最后一个 C Tile 仍会生成 MOVOUT，因为第 5.5 节的条件是 $64+32\ge70$，并不要求有效 `c_in_loop` 必须等于标称 32。

这三条指令的尺寸及 264 MAC 已通过实际生成器核对；核对对象是指令范围，不是矩阵数值或任意张量布局下的搬运字节完整性。

## 5.7 阅读裁剪代码时，还需要保留三个实现限制

**第一，裁剪不会重写标称布局。** 局部地址标签继续使用 `mapping.tile_in_loop` 的标称步长。例如 Accumulator 标签的偏移项是：

~~~text
Ns * mapping.tile_in_loop.M + Ms
~~~

`Ns`、`Ms` 是当前输出微块的 Tile 内行、列偏移；`mapping.tile_in_loop.M` 是标称输出行跨度。BERT 角落 Tile 虽然只有 24 个有效 M 元素，`Ns=8`、`Ms=0` 时仍使用 $8\times40$，不会改成 $8\times24$。这是局部标签规则，不能直接当作真实矩阵字节数组的连续分配证明。

**第二，MOVIN 的尺寸元数据与 PRELOAD 的有效尺寸不同。** 当前生成器填写：

| 指令 | 明确填写的尺寸字段 | 字段值的性质 |
|---|---|---|
| 权重 MOVIN | `tile_m=tile_in_loop.M`、`tile_k=tile_in_loop.C` | 标称内部长度 |
| 激活 MOVIN | `tile_k=tile_in_loop.C`、`tile_n=tile_in_loop.N` | 标称内部长度 |
| GEMM_PRELOAD | `tile_m=m_loop`、`tile_k=c_iter_size`、`tile_n=n_loop` | 当前微块的有效长度 |

表中的 `tile_in_loop` 指当前 `mapping.tile_in_loop`。MOVIN 的实际请求数由裁剪后的下标循环及地址集合决定，不能从它的标称 `tile_m/tile_k/tile_n` 字段直接推算边界数据量。计算范围裁剪也不把固定粒度的 DRAM 请求变成逐元素大小的搬运。

**第三，M/N 只检查全局矩阵末端，没有独立检查手工 Tile 的局部末端。** 源码没有另外取“阵列步长”和“内部长度减去局部偏移”的较小值。例如手工内部 M 长度若为 10，而全局 M 仍很长，`Ms=8` 时仍可能得到 `m_loop=8`，超出当前 Tile 标称的 10 个元素。

因此，本文主例及两个边界例子的**标称 M/N 内部长度均为 8 的倍数**。这些例子足以说明当前生成器的规则分块、全局尾部裁剪和有效指令尺寸；不能把这段实现概括为支持任意手工内部尺寸的通用裁剪器。


