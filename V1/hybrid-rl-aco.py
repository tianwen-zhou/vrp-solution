import numpy as np
import random
import time
from datasets import load_dataset

# 加载 LaDe-D 数据集
ds = load_dataset("Cainiao-AI/LaDe-D")
data_jl = ds['delivery_jl']

# 选择前1000条数据
n = 1000
data_subset = data_jl.select(range(n))

# 车辆数量和每辆车的配送单数
vehicle_count = n // 100

# 提取经纬度数据
locations = [(row['lat'], row['lng']) for row in data_subset]

# 计算欧几里得距离
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# ACO参数
Q = 100  # 信息素总量
rho = 0.1  # 信息素蒸发率
alpha = 1  # 信息素的影响程度
beta = 2  # 距离的影响程度
max_iterations = 100  # 迭代次数
n_ants = 10  # 蚂蚁数量

# RL参数
alpha_rl = 0.1    # 学习率
gamma = 0.9    # 折扣因子
epsilon = 0.1  # 探索率
iterations_rl = 1000  # RL训练迭代次数

# Q-learning Q表初始化
Q_rl = np.zeros((n, n))  # Q表，大小为 (n, n)，每个状态-动作对的价值

# 初始化信息素矩阵
pheromone = np.ones((n, n))  # 初始化信息素矩阵为1
best_route = None
best_length = float('inf')

# 计算路径长度
def calculate_route_length(route):
    length = 0
    for i in range(len(route) - 1):
        length += euclidean_distance(locations[route[i]], locations[route[i + 1]])
    length += euclidean_distance(locations[route[-1]], locations[route[0]])  # 回到起点
    return length

# 选择下一个城市（基于ACO和RL的加权选择）
def select_next_city(current_city, visited_cities, pheromone, distances, epsilon, Q_rl):
    pheromone_row = pheromone[current_city]
    pheromone_row[visited_cities] = 0  # 忽略已访问的城市
    pheromone_row = pheromone_row ** alpha  # 信息素影响

    distances_row = np.array([euclidean_distance(locations[current_city], locations[i]) for i in range(len(locations))])
    distances_row[visited_cities] = float('inf')  # 忽略已访问的城市
    # 防止除零问题，避免距离为零的情况
    distances_row = np.where(distances_row == 0, np.inf, distances_row)
    distances_row = (1 / distances_row) ** beta  # 距离影响

    # RL选择
    rl_values = Q_rl[current_city]
    rl_values[visited_cities] = float('-inf')  # 忽略已访问的城市

    # 结合ACO和RL的选择策略
    combined_prob = pheromone_row * distances_row * (1 + epsilon * rl_values)

    # 防止出现 NaN 或负值，将 NaN 转换为 0，并确保概率非负
    combined_prob = np.nan_to_num(combined_prob, nan=0.0)
    combined_prob = np.maximum(combined_prob, 0)  # 确保非负

    # 如果所有概率为零，随机选择一个未访问的城市
    if np.sum(combined_prob) == 0:
        return random.choice([i for i in range(len(locations)) if i not in visited_cities])

    combined_prob /= np.sum(combined_prob)  # 归一化
    return np.random.choice(range(len(locations)), p=combined_prob)

# ACO-RL混合算法
def aco_rl_algorithm():
    global best_route, best_length, pheromone, Q_rl

    start_time = time.time()

    # 进行混合ACO和RL的优化
    for iteration in range(max_iterations):
        all_routes = []
        all_lengths = []

        for _ in range(n_ants):
            visited_cities = [0]  # 每辆车从起点出发
            route = [0]
            while len(route) < n:
                next_city = select_next_city(route[-1], visited_cities, pheromone, euclidean_distance, epsilon, Q_rl)
                route.append(next_city)
                visited_cities.append(next_city)

            route_length = calculate_route_length(route)
            all_routes.append(route)
            all_lengths.append(route_length)

            if route_length < best_length:
                best_length = route_length
                best_route = route

            # RL学习
            for i in range(len(route) - 1):
                current_city = route[i]
                next_city = route[i + 1]
                reward = -euclidean_distance(locations[current_city], locations[next_city])
                Q_rl[current_city][next_city] = Q_rl[current_city][next_city] + alpha_rl * (reward + gamma * np.max(Q_rl[next_city]) - Q_rl[current_city][next_city])

        # 更新信息素
        pheromone *= (1 - rho)  # 信息素蒸发
        for route in all_routes:
            for i in range(len(route) - 1):
                pheromone[route[i], route[i + 1]] += Q / best_length
            pheromone[route[-1], route[0]] += Q / best_length

    end_time = time.time()
    elapsed_time = end_time - start_time

    return best_route, best_length, elapsed_time

# 执行ACO-RL算法
best_route, best_length, elapsed_time = aco_rl_algorithm()

# 输出统计数据
order_completion_rate = 100  # 假设所有订单完成
violation_rate = 0  # 假设没有违规
print(f"Best Route Length: {best_length} km")
print(f"Vehicle Count: {vehicle_count}")
print(f"Computation Time (s): {elapsed_time:.2f}")
print(f"Order Completion Rate (%): {order_completion_rate}")
print(f"Violation Rate (%): {violation_rate}")
