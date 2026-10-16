from utilities import *

# sample inputs
cust_demands = [0, 2, 2, 3]
dist_matrix = [
    [0, 2.5, 5.6, 6.0],
    [2.5, 0, 3.9, 6.5],
    [5.6, 3.9, 0, 5.2],
    [6.0, 6.5, 5.2, 0],
]
vehicle_capacity = 5
num_vehicles = 2

# sample output
total_cost = 21.8
routes = [
    [0, 1, 0],
    [0, 2, 3, 0],
]

def algo(cust_demands, dist_matrix, vehicle_capacity, num_vehicles):
    # returns output: (total_cost, routes)
    return (total_cost, routes)


total_cost, routes = algo(cust_demands, dist_matrix, vehicle_capacity, num_vehicles)
feasibility_chek = is_feasible(routes, cust_demands, vehicle_capacity, num_vehicles)
cost = calculate_total_cost(routes, dist_matrix)

