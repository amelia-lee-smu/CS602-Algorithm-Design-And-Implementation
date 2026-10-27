'''
Runs one algorithm on its own, for evaluate_algorithms.ipynb.

The functions live in a file (not in the notebook) because the solver runs in a separate process,
and on macOS / Windows that process can only start functions it can import.

ALGORITHMS:
    packing         final_solution.run_packing only
    clarke_wright   clark_wright_savings.run_clarke_wright only
    rfcs            route_first_cluster_second.run_rfcs only, starting with no solution
    full            final_solution.find_solution: packing -> Clarke-Wright -> RFCS, as run_algo runs it
'''

import multiprocessing
import random
import time

import final_solution
from clark_wright_savings import run_clarke_wright
from route_first_cluster_second import run_rfcs
from utilities import SEED

ALGORITHMS = ("packing", "clarke_wright", "rfcs", "full")


def run_algorithm(algorithm, cust_demands, dist_matrix, vehicle_cap, num_vehicles, best):
    '''
    Process target: runs one algorithm, saving its best solution to best["solution"].
    The setup is the same as final_solution.find_solution, so results match the full solver.
    '''
    if algorithm == "full":
        final_solution.find_solution(cust_demands, dist_matrix, vehicle_cap, num_vehicles, best)
        return

    state = {
        "best": best,
        "best_cost": best["solution"][0],
    }
    rng = random.Random(SEED)
    nearest_neighbours = final_solution.build_nearest_neighbours(dist_matrix)
    overload_penalty = final_solution.starting_overload_penalty(cust_demands, dist_matrix)

    run = {
        "packing": final_solution.run_packing,
        "clarke_wright": run_clarke_wright,
        "rfcs": run_rfcs,
    }[algorithm]
    run(cust_demands, dist_matrix, vehicle_cap, num_vehicles, nearest_neighbours, overload_penalty, rng, state)


def solve(algorithm, cust_demands, dist_matrix, vehicle_cap, num_vehicles, time_limit):
    '''
    Runs algorithm in a separate process, stopping it after time_limit seconds
    (packing and Clarke-Wright usually finish well before that; RFCS and full always run until stopped).
    Returns (cost, routes, time_taken); routes is None if no feasible solution was found.
    '''
    with multiprocessing.Manager() as manager:
        best = manager.dict(solution=(float("inf"), None))
        process = multiprocessing.Process(
            target=run_algorithm,
            args=(algorithm, cust_demands, dist_matrix, vehicle_cap, num_vehicles, best),
        )
        start = time.perf_counter()
        process.start()
        process.join(timeout=time_limit)
        time_taken = time.perf_counter() - start

        if process.is_alive():
            process.terminate()
            process.join()

        cost, routes = best["solution"]
    return cost, routes, time_taken
