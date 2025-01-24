import numpy as np
from scipy.optimize import minimize

# 计算期望收益的函数
def calculate_expected_payoffs_A(p_A, p_B, payoff_matrix_A):
    return np.sum(np.dot(payoff_matrix_A, p_B) * p_A)

def calculate_expected_payoffs_B(p_A, p_B, payoff_matrix_B):
    return np.sum(np.dot(payoff_matrix_B.T, p_A) * p_B)

# 定义目标函数：同时最大化A和B的期望收益
def objective(vars, payoff_matrix_A, payoff_matrix_B):
    p_A = vars[:4]
    p_B = vars[4:]
    
    # 归一化策略概率
    p_A = np.maximum(p_A, 0)
    p_B = np.maximum(p_B, 0)
    p_A /= np.sum(p_A)
    p_B /= np.sum(p_B)
    
    # 计算期望收益
    E_A = calculate_expected_payoffs_A(p_A, p_B, payoff_matrix_A)
    E_B = calculate_expected_payoffs_B(p_A, p_B, payoff_matrix_B)
    
    # 返回负的期望收益，因为minimize是最小化，目标是最大化期望收益
    return -(E_A + E_B)

# 定义约束条件：策略概率之和为1
def constraint(vars):
    p_A = vars[:4]
    p_B = vars[4:]
    return [np.sum(p_A) - 1, np.sum(p_B) - 1]

# 输入支付矩阵
payoff_matrix_A = np.array([
    [0, 2, -3, 3],   # Player A 的策略 R 对应的收益
    [1, -1, -1, 4],  # Player A 的策略 D 对应的收益
    [-1, -3, -2, 10], # Player A 的策略 A 对应的收益
    [2, -2, -10, 0]   # Player A 的策略 B 对应的收益
])

payoff_matrix_B = np.array([
    [0, 2, -3, 3],   # Player B 的策略 R 对应的收益
    [1, -1, -1, 4],   # Player B 的策略 D 对应的收益
    [-1, -3, -2, 10],  # Player B 的策略 A 对应的收益
    [2, -2, -10, 0]      # Player B 的策略 B 对应的收益
])

# 初始猜测，假设两位玩家策略概率相等
initial_guess = np.ones(8) * 0.25  # 8个元素

# 设置约束条件
constraints = [{'type': 'eq', 'fun': constraint}]

# 使用 `minimize` 求解
result = minimize(objective, initial_guess, args=(payoff_matrix_A, payoff_matrix_B.T), constraints=constraints, bounds=[(0, 1)] * 8, options={'disp': True, 'maxiter': 10000, 'ftol': 1e-8})

# 输出结果
if result.success:
    vars = result.x
    p_A = vars[:4]  # 玩家 A 的策略概率
    p_B = vars[4:]  # 玩家 B 的策略概率

    # 归一化并打印最终策略
    p_A = np.round(np.maximum(p_A, 0) / np.sum(np.maximum(p_A, 0)), 4)
    p_B = np.round(np.maximum(p_B, 0) / np.sum(np.maximum(p_B, 0)), 4)
    
    print("Best Player A's mixed strategy probabilities (p1, p2, p3, p4):", p_A)
    print("Best Player B's mixed strategy probabilities (q1, q2, q3, q4):", p_B)

    # 计算最终期望收益
    E_A_final = np.round(calculate_expected_payoffs_A(p_A, p_B, payoff_matrix_A), 4)
    E_B_final = np.round(calculate_expected_payoffs_B(p_A, p_B, payoff_matrix_B), 4)

    print("Best Player A's expected payoffs (when B chooses best p_B):", E_A_final)
    print("Best Player B's expected payoffs (when A chooses best p_A):", E_B_final)
else:
    print("Optimization failed:", result.message)
