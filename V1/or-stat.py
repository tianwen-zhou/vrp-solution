import numpy as np
import time
from datasets import load_dataset
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from geopy.distance import geodesic

# 加载 LaDe-D 数据集
print("Loading dataset...")
ds = load_dataset("Cainiao-AI/LaDe-D")

# 选择吉林的 delivery 数据
data_jl = ds['delivery_jl']

# 减少数据量，先尝试一个较小的问题
n = 30  # 减少到30条数据，使问题更易解决
print(f"Using {n} data points for the VRP problem")
data_subset = data_jl.select(range(n))

# 计算需要的车辆数量，增加车辆数量以提高解决方案找到的概率
vehicle_count = max(3, (n + 19) // 20)  # 每车最多20单，至少3辆车
print(f"Using {vehicle_count} vehicles with capacity 20 orders each")

# 提取订单的经纬度信息
stop_locations = [(item['lat'], item['lng']) for item in data_subset]
stop_ids = [item['order_id'] for item in data_subset]

# 选择数据集中的第一个点作为仓库（起点）
depot_index = 0

# 计算球面距离矩阵（Haversine 公式）
print("Computing distance matrix...")
def compute_distance_matrix(locations):
    n = len(locations)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                # 使用整数化距离（米）但不乘以1000，保持较小的数值范围
                dist_matrix[i][j] = int(geodesic(locations[i], locations[j]).kilometers)

    return dist_matrix

# 生成距离矩阵
dist_matrix = compute_distance_matrix(stop_locations)

# 检查距离矩阵有效性
print(f"Distance matrix shape: {dist_matrix.shape}")
print(f"Distance matrix min: {dist_matrix.min()}, max: {dist_matrix.max()}")

# 创建 OR-Tools RoutingIndexManager
print("Creating routing model...")
manager = pywrapcp.RoutingIndexManager(len(stop_locations), vehicle_count, depot_index)
routing = pywrapcp.RoutingModel(manager)

# 定义距离回调函数
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return int(dist_matrix[from_node][to_node])

# 注册回调函数
transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# 添加距离维度
dimension_name = 'Distance'
routing.AddDimension(
    transit_callback_index,
    0,  # no slack
    3000,  # 最大距离限制（公里）
    True,  # 从起点开始累计
    dimension_name)
distance_dimension = routing.GetDimensionOrDie(dimension_name)
distance_dimension.SetGlobalSpanCostCoefficient(100)

# 定义容量回调函数
def demand_callback(from_index):
    from_node = manager.IndexToNode(from_index)
    return 1  # 每个订单占 1 个容量

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

# 添加容量约束
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,  # 无额外容量
    [20] * vehicle_count,  # 每辆车最多配送 20 单（减少每车负载）
    True,
    "Capacity"
)

# 添加距离惩罚项
for i in range(routing.Size()):
    if i != routing.Start(0):  # 不包括起点
        routing.AddDisjunction([i], 1000)  # 高惩罚项使订单尽量被服务

# 设定搜索参数，使用更复杂的策略
print("Setting solver parameters...")
search_parameters = pywrapcp.DefaultRoutingSearchParameters()

# 使用更适合VRP的首解策略
search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS

# 使用导引局部搜索 - 通常比模拟退火更有效
search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

# 增加时间限制到2分钟
search_parameters.time_limit.seconds = 120
search_parameters.log_search = True

# 记录求解开始时间
start_time = time.time()

# 求解 VRP 问题
print("Solving the problem...")
solution = routing.SolveWithParameters(search_parameters)

# 解析并打印结果
def print_solution(manager, routing, solution, stop_ids):
    print("Solution found!")
    total_distance = 0
    total_orders_served = 0

    for vehicle_id in range(vehicle_count):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        order_count = 0

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index < len(stop_ids):
                order_id = stop_ids[node_index]
                plan_output += f" {node_index}(Order:{order_id}) -> "
            else:
                plan_output += f" {node_index} -> "

            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
            order_count += 1

        plan_output += f"{manager.IndexToNode(index)}"
        plan_output += f"\nDistance of route: {route_distance:.2f}km"
        plan_output += f"\nLoad of route: {order_count-1}"  # 减去1是因为计数包含了返回仓库
        print(plan_output)
        total_distance += route_distance
        total_orders_served += (order_count-1)

    completion_rate = (total_orders_served / (len(stop_ids)-1)) * 100 if len(stop_ids) > 1 else 0
    print(f"Total distance of all routes: {total_distance:.2f}km")
    print(f"Total orders served: {total_orders_served}/{len(stop_ids)-1}")
    print(f"Completion rate: {completion_rate:.2f}%")
    return total_distance, total_orders_served, len(stop_ids)-1

# 处理求解结果
if solution:
    total_distance, orders_served, total_orders = print_solution(manager, routing, solution, stop_ids)
    elapsed_time = time.time() - start_time
    print(f"总路径长度 (km): {total_distance:.2f}")
    print(f"车辆总数: {vehicle_count}")
    print(f"计算时间 (s): {elapsed_time:.2f}")
    print(f"订单完成率 (%): {(orders_served/total_orders*100):.2f}")

    # 可视化路径（如果需要的话，可以添加matplotlib代码）
else:
    print("未找到解决方案 - 检查问题约束和参数")
    print("尝试诊断问题...")

    # 检查问题是否过于受限
    print(f"1. 检查车辆容量: 每辆车容量为20, 总订单数为{n}, 总车辆数{vehicle_count}")
    capacity_ok = vehicle_count * 20 >= n
    print(f"   容量检查: {'通过' if capacity_ok else '不足'}")

    # 检查距离矩阵是否合理
    nan_values = np.isnan(dist_matrix).any()
    inf_values = np.isinf(dist_matrix).any()
    print(f"2. 检查距离矩阵: NaN值: {nan_values}, 无穷值: {inf_values}")

    # 尝试放宽约束再次求解
    print("尝试放宽约束再次求解...")

    # 放宽容量约束
    for i in range(vehicle_count):
        routing.GetDimensionOrDie("Capacity").CumulVar(routing.End(i)).RemoveInterval(0, 20)
        routing.GetDimensionOrDie("Capacity").CumulVar(routing.End(i)).SetRange(0, 25)

    # 使用更简单的首解策略
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH

    print("使用PARALLEL_CHEAPEST_INSERTION策略和TABU_SEARCH元启发式重新求解...")
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        total_distance, orders_served, total_orders = print_solution(manager, routing, solution, stop_ids)
        elapsed_time = time.time() - start_time
        print(f"总路径长度 (km): {total_distance:.2f}")
        print(f"车辆总数: {vehicle_count}")
        print(f"计算时间 (s): {elapsed_time:.2f}")
        print(f"订单完成率 (%): {(orders_served/total_orders*100):.2f}")
    else:
        print("放宽约束后仍未找到解决方案")
        print("建议使用以下方法:")
        print("1. 进一步减少问题规模至20个订单")
        print("2. 增加车辆数量至n/10辆")
        print("3. 采用递增求解策略 - 先解决小规模问题，逐步添加更多订单")
        print("4. 考虑采用聚类方法预处理数据，将近距离的订单分组")