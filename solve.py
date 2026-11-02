'''
Solves one instance file and prints the answer in the submission format.

Input file (the first line sets the time limit, see TIME_LIMITS in final_solution.py):
    small                   small / medium / large: stops at 28.5 / 57 / 171 s
    0 2 2 3                 demands, depot first
    0 2.5 5.6 6.0           distance matrix, one row per line
    ...
    Capacity: 5             or just 5
    Vehicles: 2             or just 2

Output (printed, and written to --output if given):
    21.8
    0 1 0
    0 2 3 0

Usage (from the project folder):
    python solve.py grp5_small.txt
    python solve.py grp5_small.txt --output grp5_small_solution.txt
'''

import argparse

from final_solution import TIME_LIMIT_FRACTION, TIME_LIMITS, run_algo


def to_number(text):
    '''Converts "5" to 5 and "2.5" to 2.5'''
    value = float(text)
    return int(value) if value.is_integer() else value


def read_instance(path):
    '''
    Reads an instance file: instance type, demands, distance matrix, capacity, vehicles.
    Capacity and vehicles may be labelled ("Capacity: 5") or plain numbers ("5").
    Returns (instance_type, cust_demands, dist_matrix, vehicle_cap, num_vehicles)
    '''
    with open(path) as instance_file:
        lines = [line.strip() for line in instance_file if line.strip()]

    instance_type = lines[0].lower()
    if instance_type not in TIME_LIMITS:
        raise ValueError(f"{path}: first line should be one of {list(TIME_LIMITS)}, got '{lines[0]}'")

    vehicle_cap = num_vehicles = None
    number_rows = []
    for line in lines[1:]:
        label = line.lower()
        if label.startswith("capacity"):
            vehicle_cap = to_number(line.split(":")[1])
        elif label.startswith("vehicles"):
            num_vehicles = int(line.split(":")[1])
        else:
            number_rows.append([to_number(value) for value in line.split()])

    # unlabelled format: vehicles is the last line, capacity the one before
    if num_vehicles is None:
        num_vehicles = int(number_rows.pop()[0])
    if vehicle_cap is None:
        vehicle_cap = number_rows.pop()[0]

    cust_demands, dist_matrix = number_rows[0], number_rows[1:]
    if len(dist_matrix) != len(cust_demands) or any(len(row) != len(cust_demands) for row in dist_matrix):
        raise ValueError(f"{path}: {len(cust_demands)} demands, so the distance matrix should be "
                         f"{len(cust_demands)} x {len(cust_demands)}")

    return instance_type, cust_demands, dist_matrix, vehicle_cap, num_vehicles


def format_cost(cost):
    '''Total cost to 6 decimal places, e.g. 21.8 -> "21.800000", 24000000 -> "24000000.000000"'''
    # if cost == int(cost):
    #     return str(int(cost))
    return f"{cost:.6f}"


def format_solution(cost, routes):
    '''The total cost on the first line, then one route per line, e.g. "0 2 3 0"'''
    lines = [format_cost(cost)]
    lines += [" ".join(str(stop) for stop in route) for route in routes]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Solve one instance file")
    parser.add_argument("instance", help="instance file, e.g. grp5_small.txt")
    parser.add_argument("--output", help="also write the solution to this file")
    args = parser.parse_args()

    instance_type, cust_demands, dist_matrix, vehicle_cap, num_vehicles = read_instance(args.instance)

    best_cost, best_routes = run_algo(cust_demands, dist_matrix, vehicle_cap, num_vehicles, instance_type)

    if best_routes is None:
        print(f"no feasible solution found in {TIME_LIMIT_FRACTION * TIME_LIMITS[instance_type]:g} s")
        return

    output = format_solution(best_cost, best_routes)
    print(output)

    if args.output:
        with open(args.output, "w") as output_file:
            output_file.write(output + "\n")


if __name__ == "__main__":
    main()
