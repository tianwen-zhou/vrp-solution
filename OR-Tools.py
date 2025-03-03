from datasets import load_dataset
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# 加载 LaDe-D 数据集
ds = load_dataset("Cainiao-AI/LaDe-D")

# 选择重庆的 delivery 数据
# dict_keys(['delivery_cq', 'delivery_hz', 'delivery_jl', 'delivery_sh', 'delivery_yt'])
# City	     numbers     Description
# Shanghai	 1483864  One of the most prosperous cities in China, with a large number of orders per day.
# Hangzhou	 1861600  A big city with well-developed online e-commerce and a large number of orders per day.
# Chongqing	 931351   A big city with complicated road conditions in China, with a large number of orders.
# Jilin	     31415    A middle-size city in China, with a small number of orders each day.
# Yantai	 206431   A small city in China, with a small number of orders every day.


# 选择吉林的 delivery 数据
data_jl = ds['delivery_jl']

# 选择任意数量的数据进行运行，例如前1000条数据
n = 100  # 你可以根据需要修改这个值
data_subset = data_jl.select(range(n))  # 选择前1000条数据
# 打印数据的前几行，检查加载的正确性
# print(data_subset[:5])  # 打印前5条数据进行查看

# 计算需要的车辆数量，每辆车送100单
vehicle_count = n // 50  # 每辆车送100单，计算需要多少辆车

print(vehicle_count)

# 提取经纬度信息
stop_locations = {item['order_id']: (item['lat'], item['lng']) for item in data_subset}

# 获取所有订单的 ID
stop_ids = list(stop_locations.keys())

# 假设第一个订单作为起点
depot = stop_ids[0]  # 起点设为第一个订单


# 计算距离矩阵（欧几里得距离）
def compute_distance_matrix(locations):
    n = len(locations)
    dist_matrix = np.zeros((n, n))
    loc_list = list(locations.values())

    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(np.array(loc_list[i]) - np.array(loc_list[j]))
    return dist_matrix


# 生成距离矩阵
dist_matrix = compute_distance_matrix(stop_locations)

# 创建 OR-Tools RoutingIndexManager
manager = pywrapcp.RoutingIndexManager(len(stop_ids), 1, stop_ids.index(depot))  # 1辆车
routing = pywrapcp.RoutingModel(manager)


# 定义距离回调函数
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return int(dist_matrix[from_node][to_node])


# 注册回调函数
transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# 设置搜索参数
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

# 求解 VRP
solution = routing.SolveWithParameters(search_parameters)


# 解析并打印结果
def print_solution(manager, routing, solution):
    total_distance = 0
    for vehicle_id in range(vehicle_count):  # 一辆车送100单
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))

        route_distance = sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
        total_distance += route_distance
        print(f"Vehicle {vehicle_id + 1} route: {route} | Distance: {route_distance}")

    print(f"Total distance: {total_distance}")


if solution:
    print_solution(manager, routing, solution)
else:
    print("No solution found.")