import numpy as np
import random
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


# 计算城市间的距离矩阵（假设是地理距离，使用 Haversine 公式）
def haversine(lat1, lon1, lat2, lon2):
    # 将经纬度从度转换为弧度
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine 公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    radius = 6371  # 地球半径，单位为千米
    return radius * c  # 返回距离，单位为千米


# 构建距离矩阵
distances = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        lat1, lon1 = locations[i]
        lat2, lon2 = locations[j]
        dist = haversine(lat1, lon1, lat2, lon2)
        distances[i, j] = dist
        distances[j, i] = dist  # 对称矩阵

# ACO算法的相关参数
Q = 100  # 信息素总量
rho = 0.1  # 信息素蒸发率
alpha = 1  # 信息素的影响程度
beta = 2  # 距离的影响程度
max_iterations = 100  # 迭代次数
n_ants = 10  # 蚂蚁数量

# 初始化信息素矩阵
pheromone = np.ones((n, n))  # 初始化信息素矩阵为1
best_route = None
best_length = float('inf')


# 计算路径长度
def calculate_route_length(route):
    length = 0
    for i in range(len(route) - 1):
        length += distances[route[i], route[i + 1]]
    length += distances[route[-1], route[0]]  # 返回起点
    return length


# 选择下一个城市（基于信息素和距离的加权选择）
def select_next_city(current_city, visited_cities, pheromone, distances):
    pheromone_row = pheromone[current_city]
    pheromone_row[visited_cities] = 0  # 忽略已访问的城市
    pheromone_row = pheromone_row ** alpha  # 信息素影响

    distances_row = distances[current_city]
    distances_row[visited_cities] = float('inf')  # 忽略已访问的城市
    distances_row = (1 / distances_row) ** beta  # 距离影响

    # 处理距离为零的情况，避免除零错误
    distances_row = np.nan_to_num(distances_row, nan=np.inf)  # 将 NaN 转换为无穷大

    # 计算每个城市的选择概率
    probability = pheromone_row * distances_row
    total_probability = np.sum(probability)

    if total_probability == 0:  # 如果总概率为零（所有城市不可达），则随机选择
        probability = np.ones_like(probability)

    probability /= np.sum(probability)  # 归一化概率

    return np.random.choice(range(n), p=probability)  # 根据概率选择下一个城市


# ACO算法的主函数
def aco_algorithm():
    global best_route, best_length, pheromone

    for iteration in range(max_iterations):
        all_routes = []
        all_lengths = []

        for _ in range(n_ants):
            visited_cities = [random.randint(0, n - 1)]  # 随机选择起始城市
            while len(visited_cities) < n:
                next_city = select_next_city(visited_cities[-1], visited_cities, pheromone, distances)
                visited_cities.append(next_city)

            route_length = calculate_route_length(visited_cities)
            all_routes.append(visited_cities)
            all_lengths.append(route_length)

            # 更新最优路径
            if route_length < best_length:
                best_length = route_length
                best_route = visited_cities

        # 信息素更新（蒸发 + 沉积）
        pheromone *= (1 - rho)  # 信息素蒸发

        # 在最佳路径上沉积信息素
        for i in range(len(best_route) - 1):
            pheromone[best_route[i], best_route[i + 1]] += Q / best_length
        pheromone[best_route[-1], best_route[0]] += Q / best_length  # 返回起点

        print(f"Iteration {iteration + 1}/{max_iterations}, Best Length: {best_length}")

    return best_route, best_length


# 运行ACO算法
best_route, best_length = aco_algorithm()
print(f"Best route: {best_route}")
print(f"Best route length: {best_length}")
