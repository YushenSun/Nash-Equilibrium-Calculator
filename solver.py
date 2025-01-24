import numpy as np
from scipy.optimize import minimize

# 计算期望收益的函数，分别计算玩家 A 和 B 的期望收益
def calculate_expected_payoffs_A(p, payoff_matrix_A, q):
    """
    计算玩家 A 的期望收益，给定策略概率 p 和玩家 A 的支付矩阵 payoff_matrix_A
    以及玩家 B 的策略概率 q
    """
    return np.dot(payoff_matrix_A.T, q) @ p  # 玩家 A 的期望收益

def calculate_expected_payoffs_B(p, payoff_matrix_B, q):
    """
    计算玩家 B 的期望收益，给定策略概率 p 和玩家 B 的支付矩阵 payoff_matrix_B
    以及玩家 A 的策略概率 p
    """
    return np.dot(payoff_matrix_B.T, p) @ q  # 玩家 B 的期望收益

# 定义目标函数：同时最大化玩家 A 和 B 的期望收益
def payoff(vars, payoff_matrix_A, payoff_matrix_B):
    p1, p2, p3, p4, q1, q2, q3, q4 = vars  # 玩家 A 和 B 的策略概率
    p = np.array([p1, p2, p3, p4])  # 玩家 A 的策略概率
    q = np.array([q1, q2, q3, q4])  # 玩家 B 的策略概率
    
    # 计算玩家 A 和 B 的期望收益
    E_A = calculate_expected_payoffs_A(p, payoff_matrix_A, q)
    E_B = calculate_expected_payoffs_B(p, payoff_matrix_B, q)
    
    # 最大化两位玩家的期望收益
    return -(E_A + E_B)  # 取负以进行最小化操作

# 定义约束条件：策略概率之和为1
def constraint(vars):
    p1, p2, p3, p4, q1, q2, q3, q4 = vars
    return [p1 + p2 + p3 + p4 - 1, q1 + q2 + q3 + q4 - 1]  # 保证概率之和为1

# 输入支付矩阵
payoff_matrix_A = np.array([
    [0, 0, 0, 10],   # Player A 的策略 R 对应的收益
    [1, 0, 0, 0],  # Player A 的策略 D 对应的收益
    [0, 1, 0, 0], # Player A 的策略 A 对应的收益
    [0, 0, 1, 0]   # Player A 的策略 B 对应的收益
])

payoff_matrix_B = np.array([
    [0, 0, 0, 1],   # Player B 的策略 R 对应的收益
    [1, 0, 0, 0],  # Player B 的策略 D 对应的收益
    [0, 1, 0, 0], # Player B 的策略 A 对应的收益
    [0, 0, 1, 0]   # Player B 的策略 B 对应的收益
])

# 初始猜测
initial_guess = np.ones(8) * 0.125  # 初始时所有策略的概率相等

# 优化目标函数
constraints = [{'type': 'eq', 'fun': constraint}]
result = minimize(payoff, initial_guess, args=(payoff_matrix_A, payoff_matrix_B), constraints=constraints, bounds=[(0, 1)] * 8)

# 输出结果
def round_small_probabilities(probabilities, threshold=1e-8):
    """
    如果概率小于给定阈值，则将其设为零。
    """
    return [0 if abs(p) < threshold else p for p in probabilities]

if result.success:
    p1, p2, p3, p4, q1, q2, q3, q4 = result.x
    p = round_small_probabilities([p1, p2, p3, p4])
    q = round_small_probabilities([q1, q2, q3, q4])
    
    # 计算平衡点下的期望收益
    E_A = calculate_expected_payoffs_A(p, payoff_matrix_A, q)
    E_B = calculate_expected_payoffs_B(p, payoff_matrix_B, q)

    print("Player A's mixed strategy probabilities (p1, p2, p3, p4):", p)
    print("Player B's mixed strategy probabilities (q1, q2, q3, q4):", q)
    print("Player A's expected payoff at equilibrium:", E_A)
    print("Player B's expected payoff at equilibrium:", E_B)
else:
    print("Optimization failed:", result.message)
