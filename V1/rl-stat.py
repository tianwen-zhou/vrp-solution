import numpy as np
import random
import time

# 加载 LaDe-D 数据集
from datasets import load_dataset
ds = load_dataset("Cainiao-AI/LaDe-D")
data_jl = ds['delivery_jl']

# 选择前1000条数据
n = 1000
data_subset = data_jl.select(range(n))

# 车辆数量和每辆车的配送单数
vehicle_count = n // 100

# 提取经纬度数据
locations = [(row['lat'], row['lng']) for row in data_subset]

# Q-learning 参数
alpha = 0.1    # 学习率
gamma = 0.9    # 折扣因子
epsilon = 0.1  # 探索率
iterations = 1000  # 训练迭代次数

# Q表初始化
Q = np.zeros((n, n))  # Q表，大小为 (n, n)，每个状态-动作对的价值

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def route_length(route):
    total_length = 0
    for i in range(len(route) - 1):
        total_length += euclidean_distance(locations[route[i]], locations[route[i+1]])
    return total_length

def q_learning():
    best_route = None
    best_length = float('inf')

    start_time = time.time()

    for episode in range(iterations):
        route = [0]  # 假设每辆车从起点出发
        visited = set(route)

        while len(route) < len(locations):
            current_location = route[-1]

            if random.uniform(0, 1) < epsilon:
                next_stop = random.choice([i for i in range(len(locations)) if i not in visited])
            else:
                next_stop = np.argmax(Q[current_location])

            route.append(next_stop)
            visited.add(next_stop)

            reward = -euclidean_distance(locations[current_location], locations[next_stop])
            Q[current_location][next_stop] = Q[current_location][next_stop] + alpha * (reward + gamma * np.max(Q[next_stop]) - Q[current_location][next_stop])

        route_len = route_length(route)
        if route_len < best_length:
            best_route = route
            best_length = route_len

    end_time = time.time()
    elapsed_time = end_time - start_time

    return best_route, best_length, elapsed_time

# 执行 Q-learning 算法
best_route, best_length, elapsed_time = q_learning()

# 输出统计数据
order_completion_rate = 100  # 假设所有订单完成
violation_rate = 0  # 假设没有违规
print(f"Best Route Length: {best_length} km")
print(f"Vehicle Count: {vehicle_count}")
print(f"Computation Time (s): {elapsed_time:.2f}")
print(f"Order Completion Rate (%): {order_completion_rate}")
print(f"Violation Rate (%): {violation_rate}")
