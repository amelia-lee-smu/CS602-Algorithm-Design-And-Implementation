import multiprocessing

from route_first_cluster_second import run_rfcs

# Time limit in seconds for each instance type
TIME_LIMITS = { 
    "small": 30,
    "medium": 60,
    "large": 180,
}
# Stop at 95% of the time limit, leaving time to write the output
TIME_LIMIT_FRACTION = 0.95


def find_solution(cust_demands, dist_matrix, vehicle_cap, num_vehicles, best, log_info):
    # 1. Clarke-Wright savings: a fast first feasible solution
    # TODO: call the CW algo here and save its solution: best["solution"] = (cost, routes)

    # 2. Route first, cluster second (RFCS): search for a better solution with the remaining time
    run_rfcs(cust_demands, dist_matrix, vehicle_cap, num_vehicles, best, log_info)

    best_cost, best_routes = best["solution"]
    return best_cost, best_routes


def run_algo(cust_demands, dist_matrix, vehicle_cap, num_vehicles, instance_type, log_info=None):
    # log_info: only passed by the batch tester, to log each run to the CSV file
    TIMEOUT = TIME_LIMIT_FRACTION * TIME_LIMITS[instance_type]

    with multiprocessing.Manager() as manager:
        # Saved best solution, in the format (cost, list of routes)
        best = manager.dict(solution=(float("inf"), None))

        # Run the algo
        process = multiprocessing.Process(
            target=find_solution, 
            args=(cust_demands, dist_matrix, vehicle_cap, num_vehicles, best, log_info)
        )
        process.start()
        process.join(timeout=TIMEOUT) # set the time limit for the run

        # Terminate the process when the time limit is hit
        if process.is_alive():
            print(f"Code terminated at {TIMEOUT} seconds.")
            process.terminate()  # Forces the process to close
            process.join()       # Cleans up the process resources

        best_cost, best_routes = best["solution"]

    return best_cost, best_routes


# if __name__ == "__main__":
#     from test_cases.sample_from_instructions import (
#         cust_demands,
#         dist_matrix,
#         vehicle_capacity,
#         num_vehicles,
#     )

#     best_cost, best_routes = run_algo(cust_demands, dist_matrix, vehicle_capacity, num_vehicles, "small")
#     print(best_cost)
#     print(best_routes)
