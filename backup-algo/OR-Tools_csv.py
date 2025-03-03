import pandas as pd
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# 读取本地CSV文件，并确保第一行是列名
file_path = "../data/data_jilin.csv"  # 请替换为实际文件路径
data = pd.read_csv(file_path, header=0)

# 清理列名，去除空格
data.columns = data.columns.str.strip()

# 选择前1000条数据进行测试
data_subset = data.head(1000)

# 打印前几行数据，检查数据是否正确加载
print(data_subset.head())

# 提取经纬度信息，并进行异常处理
stop_locations = {}

for _, row in data_subset.iterrows():
    try:
        # 提取每一行数据
        order_id = row['order_id']
        lat = row['lat']
        lng = row['lng']

        # 检查数据是否完整且合理
        if pd.isna(order_id) or pd.isna(lat) or pd.isna(lng):
            raise ValueError(f"Invalid data at order_id {order_id}: missing lat or lng")

        # 如果数据有效，添加到 stop_locations 字典中
        stop_locations[order_id] = (lat, lng)

    except Exception as e:
        # 捕获任何异常并打印错误信息，跳过当前行
        print(f"Skipping row due to error: {e}")

# 打印有效的数据条数
print(f"Valid stops: {len(stop_locations)}")

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
    for vehicle_id in range(1):  # 这里只有一辆车
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))

        route_distance = sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
        total_distance += route_distance
        print(f"Vehicle {vehicle_id} route: {route} | Distance: {route_distance}")

    print(f"Total distance: {total_distance}")


if solution:
    print_solution(manager, routing, solution)
else:
    print("No solution found.")
