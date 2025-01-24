import numpy as np

# 计算期望收益的函数，分别计算玩家 A 和 B 的期望收益
def calculate_expected_payoffs_A(p_B, payoff_matrix_A):
    """
    计算玩家 A 的期望收益，给定策略概率 p_B 和玩家 A 的支付矩阵 payoff_matrix_A
    """
    return np.dot(payoff_matrix_A, p_B)  # 玩家 A 的期望收益

def calculate_expected_payoffs_B(p_A, payoff_matrix_B):
    """
    计算玩家 B 的期望收益，给定策略概率 p_A 和玩家 B 的支付矩阵 payoff_matrix_B
    """
    return np.dot(payoff_matrix_B, p_A)  # 玩家 B 的期望收益

# 梯度上升法计算玩家 A 和玩家 B 的策略概率
def gradient_ascent_A(p_B, payoff_matrix_A, learning_rate=0.01):
    """
    根据玩家 B 的策略 p_B，计算玩家 A 的梯度并更新其策略概率 p_A
    """
    expected_A = calculate_expected_payoffs_A(p_B, payoff_matrix_A)
    gradient_A = expected_A - np.mean(expected_A)  # 基于期望收益计算梯度
    p_A_new = p_B + learning_rate * gradient_A  # 根据梯度更新玩家 A 的策略

    # 限制策略概率为非负值
    p_A_new = np.maximum(p_A_new, 0)

    # 归一化策略概率，确保其和为 1
    p_A_new = p_A_new / np.sum(p_A_new)

    return p_A_new

def gradient_ascent_B(p_A, payoff_matrix_B, learning_rate=0.01):
    """
    根据玩家 A 的策略 p_A，计算玩家 B 的梯度并更新其策略概率 p_B
    """
    expected_B = calculate_expected_payoffs_B(p_A, payoff_matrix_B)
    gradient_B = expected_B - np.mean(expected_B)  # 基于期望收益计算梯度
    p_B_new = p_A + learning_rate * gradient_B  # 根据梯度更新玩家 B 的策略

    # 限制策略概率为非负值
    p_B_new = np.maximum(p_B_new, 0)

    # 归一化策略概率，确保其和为 1
    p_B_new = p_B_new / np.sum(p_B_new)

    return p_B_new

# 输入支付矩阵
payoff_matrix_A = np.array([
    [10, 10, 10, 10],   # Player A 的策略 R 对应的收益
    [1, 0, 0, 0],   # Player A 的策略 D 对应的收益
    [0, 1, 0, 0],   # Player A 的策略 A 对应的收益
    [0, 0, 1, 0]    # Player A 的策略 B 对应的收益
])

payoff_matrix_B = np.array([
    [0, 0, 0, 1],   # Player B 的策略 R 对应的收益
    [1, 0, 0, 0],   # Player B 的策略 D 对应的收益
    [0, 1, 0, 0],   # Player B 的策略 A 对应的收益
    [0, 0, 1, 0]    # Player B 的策略 B 对应的收益
])

# 初始策略概率
p_A = np.array([0.25, 0.25, 0.25, 0.25])  # 初始时所有策略的概率相等
p_B = np.array([0.25, 0.25, 0.25, 0.25])  # 初始时所有策略的概率相等

# 设置迭代次数和收敛阈值
max_iterations = 10000  # 强制进行最多 10000 次迭代
threshold = 1e-8  # 精度要求：更小的阈值表示要求更精确的收敛
learning_rate = 0.01  # 学习率

# 迭代过程
for iteration in range(max_iterations):
    # 优化 Player A 的策略
    new_p_A = gradient_ascent_A(p_B, payoff_matrix_A, learning_rate)
    
    # 优化 Player B 的策略
    new_p_B = gradient_ascent_B(p_A, payoff_matrix_B, learning_rate)
    
    # 检查是否收敛
    if np.linalg.norm(new_p_A - p_A) < threshold and np.linalg.norm(new_p_B - p_B) < threshold:
        print(f"Converged after {iteration} iterations.")
        break
    
    # 更新策略
    p_A = new_p_A
    p_B = new_p_B

    # 每 100 次打印一次状态
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, p_A: {np.round(p_A, 6)}, p_B: {np.round(p_B, 6)}")

# 打印最终策略和期望收益
p_A_rounded = np.round(p_A, 6)
p_B_rounded = np.round(p_B, 6)
print("Player A's mixed strategy probabilities (p1, p2, p3, p4):", p_A_rounded)
print("Player B's mixed strategy probabilities (q1, q2, q3, q4):", p_B_rounded)

# 计算最终的期望收益
E_A = np.round(np.dot(payoff_matrix_A, p_B), 6)
E_B = np.round(np.dot(payoff_matrix_B, p_A), 6)

print("Player A's expected payoffs (when B chooses p_B):", E_A)
print("Player B's expected payoffs (when A chooses p_A):", E_B)
