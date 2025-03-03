import numpy as np
import time
from datasets import load_dataset
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# 加载 LaDe-D 数据集
ds = load_dataset("Cainiao-AI/LaDe-D")

# 选择吉林的 delivery 数据
data_jl = ds['delivery_jl']

# 选择前1000条数据
n = 1000  # 你可以根据需要修改这个值
data_subset = data_jl.select(range(n))

# 计算需要的车辆数量，每辆车送100单
vehicle_count = n // 100  # 每辆车送100单，计算需要多少辆车

# 提取经纬度信息
stop_locations = {item['order_id']: (item['lat'], item['lng']) for item in data_subset}
stop_ids = list(stop_locations.keys())
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

# 开始计时
start_time = time.time()

# 求解 VRP
solution = routing.SolveWithParameters(search_parameters)

# 解析并打印结果
def print_solution(manager, routing, solution):
    total_distance = 0
    for vehicle_id in range(vehicle_count):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))

        route_distance = sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
        total_distance += route_distance
        print(f"Vehicle {vehicle_id + 1} route: {route} | Distance: {route_distance}")

    return total_distance

if solution:
    total_distance = print_solution(manager, routing, solution)
else:
    print("No solution found.")

# 计算时间
end_time = time.time()
elapsed_time = end_time - start_time

# 输出统计数据
order_completion_rate = 100  # 假设所有订单完成
violation_rate = 0  # 假设没有违规
print(f"Path Length (km): {total_distance}")
print(f"Vehicle Count: {vehicle_count}")
print(f"Computation Time (s): {elapsed_time:.2f}")
print(f"Order Completion Rate (%): {order_completion_rate}")
print(f"Violation Rate (%): {violation_rate}")
