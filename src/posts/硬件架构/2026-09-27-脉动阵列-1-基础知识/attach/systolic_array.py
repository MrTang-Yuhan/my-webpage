import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_systolic_history(history, m, n):
    """
    画出每个 cycle 的数据流动和 PE 累加值
    红色箭头: A 数据流（向右）
    蓝色箭头: B 数据流（向下）
    """
    total_cycles = len(history)

    cols = min(3, total_cycles)
    rows = (total_cycles + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))

    # 统一 axes 的索引形式为二维
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    cell_w = 1.2
    cell_h = 1.2

    for idx, item in enumerate(history):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        ax.set_title(f"Cycle {item['cycle'] + 1}", fontsize=12)

        # 画 PE 网格
        for i in range(m):
            for j in range(n):
                x = j * cell_w
                y = (m - 1 - i) * cell_h

                rect = Rectangle(
                    (x, y), cell_w, cell_h,
                    edgecolor='black', facecolor='#f2f2f2', linewidth=1.5
                )
                ax.add_patch(rect)

                in_a = item["pe_in_a"][i, j]
                in_b = item["pe_in_b"][i, j]
                acc = item["accum_after"][i, j]

                text = (
                    f"PE({i},{j})\n"
                    f"a={in_a}, b={in_b}\n"
                    f"acc={acc}"
                )
                ax.text(
                    x + cell_w / 2, y + cell_h / 2, text,
                    ha='center', va='center', fontsize=10
                )

        # 画左侧输入 A（红色）
        for i in range(m):
            val = item["left_inputs"][i]
            y = (m - 1 - i) * cell_h + cell_h / 2

            ax.annotate(
                f"{val}",
                xy=(0, y),
                xytext=(-0.7, y),
                color='red',
                fontsize=11,
                ha='center',
                va='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=2)
            )

        # 画上侧输入 B（蓝色）
        for j in range(n):
            val = item["top_inputs"][j]
            x = j * cell_w + cell_w / 2

            ax.annotate(
                f"{val}",
                xy=(x, m * cell_h),
                xytext=(x, m * cell_h + 0.6),
                color='blue',
                fontsize=11,
                ha='center',
                va='center',
                arrowprops=dict(arrowstyle='->', color='blue', lw=2)
            )

        # 画 PE 之间的 A 流动（红色，向右）
        for i in range(m):
            for j in range(n - 1):
                val = item["out_a_after"][i, j]
                y = (m - 1 - i) * cell_h + cell_h / 2
                x1 = j * cell_w + cell_w
                x2 = (j + 1) * cell_w

                ax.annotate(
                    "",
                    xy=(x2, y),
                    xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.8)
                )
                ax.text(
                    (x1 + x2) / 2, y + 0.12, f"{val}",
                    color='red', fontsize=9, ha='center'
                )

        # 画 PE 之间的 B 流动（蓝色，向下）
        for i in range(m - 1):
            for j in range(n):
                val = item["out_b_after"][i, j]
                x = j * cell_w + cell_w / 2
                y1 = (m - 1 - i) * cell_h
                y2 = (m - 2 - i) * cell_h + cell_h

                ax.annotate(
                    "",
                    xy=(x, y2),
                    xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.8)
                )
                ax.text(
                    x + 0.10, (y1 + y2) / 2, f"{val}",
                    color='blue', fontsize=9, va='center'
                )

        ax.set_xlim(-1.0, n * cell_w + 0.3)
        ax.set_ylim(-0.3, m * cell_h + 1.0)
        ax.set_aspect('equal')
        ax.axis('off')

    # 如果子图有多余，隐藏
    for idx in range(total_cycles, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis('off')

    plt.tight_layout()
    plt.show()

class PE:
    """处理单元 (Processing Element)，脉动阵列的最小细胞"""
    def __init__(self):
        self.c_accum = 0       # 当前累计结果
        self.out_a = 0         # 传给右边 PE 的 A
        self.out_b = 0         # 传给下边 PE 的 B

        # 时钟打拍前暂存
        self._next_a = 0
        self._next_b = 0

    def compute(self, in_a, in_b):
        """组合逻辑：执行乘加"""
        self.c_accum += in_a * in_b
        self._next_a = in_a
        self._next_b = in_b

    def clock_tick(self):
        """时序逻辑：时钟沿到来，更新输出"""
        self.out_a = self._next_a
        self.out_b = self._next_b

def generate_skewed_inputs(A, B):
    """
    根据原始矩阵 A, B 自动生成 skew/倾斜输入
    A: m x k
    B: k x n

    返回:
        A_inputs: shape = (total_cycles, m)
        B_inputs: shape = (total_cycles, n)
        total_cycles
    """

    # 输入转化成 numpy 数组
    A = np.asarray(A)
    B = np.asarray(B)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A 和 B 必须是二维矩阵")

    m, k = A.shape
    k2, n = B.shape

    if k != k2:
        raise ValueError("矩阵维度不匹配：A 的列数必须等于 B 的行数")

    # 乘法 + 数据穿过阵列需要的总周期
    total_cycles = m + n + k - 2

    dtype = np.result_type(A.dtype, B.dtype) # 推导 A 和 B 一起参与运算后，结果应该使用什么数据类型
                                                             # 这是为了处理 A 与 B 数据精度不一致的情况

    # 初始化 skew 输入矩阵，默认值为 0
    A_inputs = np.zeros((total_cycles, m), dtype=dtype) # A_inputs[cycle, i] 表示在 cycle 时刻送入第 i 行左侧的 A 值
    B_inputs = np.zeros((total_cycles, n), dtype=dtype) # B_inputs[cycle, j] 表示在 cycle 时刻送入第 j 列上侧的 B 值

    # A[i, p] 在 cycle = i + p 时送入第 i 行左侧
    for i in range(m):
        for p in range(k):
            cycle = i + p
            A_inputs[cycle, i] = A[i, p]

    # B[p, j] 在 cycle = j + p 时送入第 j 列上侧
    for p in range(k):
        for j in range(n):
            cycle = j + p
            B_inputs[cycle, j] = B[p, j]

    return A_inputs, B_inputs, total_cycles

def simulate_systolic_array(A, B):
    """
    模拟脉动阵列矩阵乘法

    返回:
        C: 最终结果矩阵（numpy.ndarray）
        history: 每个 cycle 的状态，用于可视化
        A_inputs, B_inputs: 自动生成的 skew 输入（numpy.ndarray）
    """

    # 输入转换为 numpy 数组
    A = np.asarray(A)
    B = np.asarray(B)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A 和 B 必须是二维矩阵")

    m, k = A.shape    # A = (m, k)
    k2, n = B.shape   # B = (k2, n)

    if k != k2:
        raise ValueError("矩阵维度不匹配：A 的列数必须等于 B 的行数")

    A_inputs, B_inputs, total_cycles = generate_skewed_inputs(A, B)

    # 建立 m x n 的 PE 网格
    pe_grid = [[PE() for _ in range(n)] for _ in range(m)]

    history = [] # 保存所有 cycle 的历史关键信息

    # 模拟整个脉动阵列完成矩阵乘运算的过程
    for cycle in range(total_cycles):
        # 保存“当前这个 cycle”的所有关键信息
        cycle_info = {
            "cycle": cycle, # 当前 cycle
            "left_inputs": A_inputs[cycle].copy(),  # 当前这个 cycle，从阵列左边输入到每一行的 A 数据
            "top_inputs": B_inputs[cycle].copy(),   # 当前这个 cycle，从阵列上面输入到每一列的 B 数据
            "pe_in_a": np.zeros((m, n), dtype=A_inputs.dtype), # 记录当前 cycle 里，每个 PE 实际收到的 in_a 值
            "pe_in_b": np.zeros((m, n), dtype=B_inputs.dtype), # 记录当前 cycle 里，每个 PE 实际收到的 in_b 值
            "accum_after": np.zeros((m, n), dtype=np.result_type(A.dtype, B.dtype)), # 在本 cycle 结束后，每个 PE 的累计乘加结果 c_accum
            "out_a_after": np.zeros((m, n), dtype=A_inputs.dtype), # 在当前 cycle 打拍结束后，每个 PE 输出给右边 PE 的 A 数据
            "out_b_after": np.zeros((m, n), dtype=B_inputs.dtype), # 在当前 cycle 打拍结束后，每个 PE 输出给下面 PE 的 B 数据
        }

        # 阶段 A：计算
        for i in range(m):
            for j in range(n):
                # 对于每个 PE(i,j)，计算它在这个 cycle 实际收到的输入：
                # 对于最左侧 PE（j=0），它的 in_a 来自 A_inputs；对于其他 PE，它的 in_a 来自左边 PE 的 out_a
                # 对于最上侧 PE（i=0），它的 in_b 来自 B_inputs；对于其他 PE，它的 in_b 来自上边 PE 的 out_b
                in_a = A_inputs[cycle, i] if j == 0 else pe_grid[i][j - 1].out_a
                in_b = B_inputs[cycle, j] if i == 0 else pe_grid[i - 1][j].out_b

                cycle_info["pe_in_a"][i, j] = in_a
                cycle_info["pe_in_b"][i, j] = in_b

                pe_grid[i][j].compute(in_a, in_b)

        # 阶段 B：打拍
        for i in range(m):
            for j in range(n):
                pe_grid[i][j].clock_tick()
                cycle_info["accum_after"][i, j] = pe_grid[i][j].c_accum
                cycle_info["out_a_after"][i, j] = pe_grid[i][j].out_a
                cycle_info["out_b_after"][i, j] = pe_grid[i][j].out_b

        history.append(cycle_info)

    # 收集结果矩阵 C
    C = np.array([[pe_grid[i][j].c_accum for j in range(n)] for i in range(m)],
                 dtype=np.result_type(A.dtype, B.dtype))
    return C, history, A_inputs, B_inputs


def print_cycle_history(history):
    """命令行打印每个 cycle 的 pe_grid 累加值"""
    for item in history:
        cycle = item["cycle"]
        acc = item["accum_after"]
        print(f"\n--- Clock Cycle {cycle + 1} ---")
        for row in acc:
            print(" ".join(f"[{int(v):3d}]" for v in row))



if __name__ == "__main__":
    # ----------------- 测试代码 -----------------
    # 用户现在只需要提供原始矩阵，不需要手动写 skew 输入

    # A = np.random.randint(0, 10, size=(5, 3))
    # B = np.random.randint(0, 10, size=(3, 2))

    A = np.array([
        [1, 2, 5],
        [3, 4, 4]
    ])

    B = np.array([
        [5, 6, 1],
        [7, 8, 2],
        [2, 8, 3]
    ])

    C, history, A_inputs, B_inputs = simulate_systolic_array(A, B)

    print("自动生成的 skew 输入 A_inputs：")
    for i, row in enumerate(A_inputs):
        print(f"cycle {i + 1}: {row}")

    print("\n自动生成的 skew 输入 B_inputs：")
    for i, row in enumerate(B_inputs):
        print(f"cycle {i + 1}: {row}")

    print_cycle_history(history)

    print("\n最终结果矩阵 C：")
    print(C)

    print("\nNumPy 直接计算 A @ B：")
    print(A @ B)

    # 画图展示每个 cycle 的数据流动和 PE 内部值
    m, _ = A.shape
    _, n = B.shape
    plot_systolic_history(history, m=m, n=n)