import numpy as np


def load_vrp_file(file_path):
    """ 解析 .vrp 文件，提取车辆数量、容量、节点坐标等信息 """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    coordinates = []
    demand = []
    vehicle_capacity = 0
    num_vehicles = 0
    parsing_nodes = False
    parsing_demand = False

    for line in lines:
        line = line.strip()
        if line.startswith("DIMENSION"):
            num_nodes = int(line.split()[-1])
        elif line.startswith("VEHICLES"):
            num_vehicles = int(line.split()[-1])
        elif line.startswith("CAPACITY"):
            vehicle_capacity = int(line.split()[-1])
        elif line.startswith("NODE_COORD_SECTION"):
            parsing_nodes = True
        elif line.startswith("DEMAND_SECTION"):
            parsing_nodes = False
            parsing_demand = True
        elif line.startswith("DEPOT_SECTION") or line.startswith("EOF"):
            parsing_demand = False
        elif parsing_nodes:
            parts = list(map(float, line.split()))
            coordinates.append((parts[1], parts[2]))
        elif parsing_demand:
            demand.append(int(line.split()[-1]))

    coordinates = np.array(coordinates)
    demand = np.array(demand)
    return num_nodes, num_vehicles, vehicle_capacity, coordinates, demand


# 加载数据
file_path = "../data/X-n439-k37.vrp"
num_nodes, num_vehicles, vehicle_capacity, coordinates, demand = load_vrp_file(file_path)
print(f"Nodes: {num_nodes}, Vehicles: {num_vehicles}, Capacity: {vehicle_capacity}")
