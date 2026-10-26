'''
Logging for testing only: one CSV row per solver step, so a run can be checked afterwards.

None of the solver files refer to this file. attach() wraps a few solver functions from the
outside (only in the process it is called in), and every wrapper calls the real function
unchanged, so the solver behaves exactly the same with or without logging.
To remove logging completely: delete this file and the lines in run_benchmarks.py that use it.

Used by run_benchmarks.py:
    process target = find_solution_with_log(cust_demands, dist_matrix, vehicle_cap, num_vehicles, best, log_info)
'''

import csv
import inspect
import os
import time

import final_solution
import route_first_cluster_second as rfcs
from utilities import SEED, calculate_cost, is_feasible


# ---------------------------------------------------------------------------
# CSV log (one file per batch of test cases)
# ---------------------------------------------------------------------------

# Columns of the log file, in order
LOG_COLUMNS = [
    "instance",       # test case name, e.g. A-n32-k5
    "time_s",         # seconds since the run started
    "step_time_s",    # seconds since the previous row, i.e. how long this step took
    "iteration",      # 0 = setup, then 1, 2, 3, ... (one giant tour, or one ruin and recreate)
    "step",           # what just finished (see LOG_STEPS)
    "cost",           # cost right after this step (empty if the step failed)
    "best_cost",      # best feasible cost so far (empty if there is none yet)
    "new_best",       # yes if this step's routes were saved as the new best
    "routes",         # routes used / vehicles available, e.g. 5/5
    "detail",         # why a step failed, or facts about the instance
]

# Values of the step column. Each row is written when its step FINISHES, so if the process
# was terminated, the step after the last row is the one that was still running.
# An iteration runs: giant_tour -> split / split_soft (or a failure) -> relocate_swap -> two_opt
#              or:   ruin_recreate (or recreate_failed) -> relocate_swap -> two_opt
LOG_STEPS = {
    "start":                "setup finished; detail has the seed, size and tightness of the instance",
    "instance_infeasible":  "the instance can never be solved (detail says why); run_benchmarks.py skips it",
    "giant_tour":           "giant tour built and 2-opted; cost = its length as a round trip from the depot",
    "split":                "tour cut into routes within capacity",
    "split_soft":           "split found no cut, so routes were allowed to overload, then repaired",
    "split_soft_no_cut":    "FAILED: no cut into num_vehicles routes, even with overloaded routes allowed",
    "repair_failed":        "FAILED: routes still overloaded after repair",
    "partition_infeasible": "FAILED: routes broke a rule, which is a bug",
    "ruin_recreate":        "customers removed from the best solution and put back",
    "recreate_failed":      "FAILED: a removed customer fit in no route",
    "relocate_swap":        "customers moved between routes",
    "two_opt":              "customers reordered inside each route",
}


def open_log(log_info):
    '''
    log_info: None (logging off), or a dict with
        log_path, instance (from the batch tester)
    If log_info is None, return None, so that nothing is logged.
    Otherwise:
    1. Record the start time (time.monotonic), used for time_s and step_time_s
    2. Create the logs folder if it does not exist
    3. Open log_path in append mode
    4. Write the header row (LOG_COLUMNS) only if the file is new or empty
    Returns a log handle holding the start time, the time of the last row, the open file, the CSV writer and log_info
    '''
    if log_info is None:
        return None

    start_time = time.monotonic()
    log_path = log_info["log_path"]

    folder = os.path.dirname(log_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    is_new_file = not os.path.exists(log_path) or os.path.getsize(log_path) == 0

    log_file = open(log_path, "a", newline="")
    writer = csv.writer(log_file)
    if is_new_file:
        writer.writerow(LOG_COLUMNS)
        log_file.flush()

    return {"start_time": start_time, "last_time": start_time, "file": log_file, "writer": writer,
            "info": log_info}


def write_row(log, iteration, step, cost, best_cost, new_best, num_routes, num_vehicles, detail=""):
    '''
    Write one row to the CSV log. Does nothing if log is None.
    cost: None if the step failed; best_cost: infinity if there is no feasible solution yet;
    num_routes: None if there are no routes to count.
    Flushes straight away, so the row is kept if the process is terminated.
    '''
    if log is None:
        return

    now = time.monotonic()
    step_time = now - log["last_time"]
    log["last_time"] = now

    info = log["info"]
    log["writer"].writerow([
        info["instance"],
        f"{now - log['start_time']:.3f}",
        f"{step_time:.3f}",
        iteration,
        step,
        "" if cost is None else f"{cost:.2f}",
        "" if best_cost == float("inf") else f"{best_cost:.2f}",
        "yes" if new_best else "",
        "" if num_routes is None else f"{num_routes}/{num_vehicles}",
        detail,
    ])
    log["file"].flush()     # written to the OS straight away, so it survives the process being terminated


# ---------------------------------------------------------------------------
# Wrapping the solver
# ---------------------------------------------------------------------------

def attach(log_info, cust_demands, dist_matrix, vehicle_cap, num_vehicles):
    '''
    Opens the log and wraps the solver functions that mark the end of each step, so each one writes a row:
        final_solution.run_rfcs         -> start (after Clarke-Wright, as RFCS begins)
        rfcs.build_giant_tour           -> giant_tour (and starts a new iteration)
        rfcs.partition_giant_tour       -> split_soft_no_cut / repair_failed / partition_infeasible (on failure)
        rfcs.ruin_and_recreate          -> recreate_failed (on failure; also starts a new iteration)
        rfcs.check_and_update_new_best  -> split / split_soft / ruin_recreate / relocate_swap / two_opt
    Each loop of run_rfcs calls exactly one of build_giant_tour or ruin_and_recreate,
    so counting those calls gives the same iteration numbers as run_rfcs.
    check_and_update_new_best is not told which step it is saving, so the name comes from the function
    that produced the routes just before it (run_rfcs always saves straight after each step):
        partition_giant_tour -> its method ("split" / "split_soft"), ruin_and_recreate -> "ruin_recreate",
        improve_between_routes -> "relocate_swap", improve_within_routes -> "two_opt"
    Only changes the modules in this process, so call it inside the solver process.
    '''
    log = open_log(log_info)
    tracker = {"iteration": 0, "best_cost": float("inf"), "step": ""}    # step: the last step that made routes

    def row(step, cost=None, num_routes=None, detail="", new_best=False):
        write_row(log, tracker["iteration"], step, cost, tracker["best_cost"], new_best,
                  num_routes, num_vehicles, detail)

    original_run_rfcs = final_solution.run_rfcs
    def run_rfcs_logged(cust_demands, dist_matrix, vehicle_cap, num_vehicles, best, *args, **kwargs):
        tracker["best_cost"] = best["solution"][0]      # Clarke-Wright's cost, if it saved one
        total_demand = sum(cust_demands)
        row("start", detail=(f"seed {SEED}, {len(cust_demands) - 1} customers, {num_vehicles} vehicles, "
                             f"capacity {vehicle_cap}, total demand {total_demand}, "
                             f"tightness {total_demand / (num_vehicles * vehicle_cap):.3f}"))
        return original_run_rfcs(cust_demands, dist_matrix, vehicle_cap, num_vehicles, best, *args, **kwargs)
    final_solution.run_rfcs = run_rfcs_logged

    original_build_giant_tour = rfcs.build_giant_tour
    def build_giant_tour_logged(*args, **kwargs):
        tracker["iteration"] += 1
        tour = original_build_giant_tour(*args, **kwargs)
        row("giant_tour", cost=calculate_cost(dist_matrix, [[0, *tour, 0]]))
        return tour
    rfcs.build_giant_tour = build_giant_tour_logged

    original_partition = rfcs.partition_giant_tour
    def partition_giant_tour_logged(*args, **kwargs):
        routes, method = original_partition(*args, **kwargs)
        if routes is None:
            row(method)     # the step name says how it failed
        elif not is_feasible(routes, cust_demands, vehicle_cap, num_vehicles):
            row("partition_infeasible")     # run_rfcs skips these routes too (log only: the solver checks itself)
        else:
            tracker["step"] = method        # "split" or "split_soft", for the save that follows
        return routes, method
    rfcs.partition_giant_tour = partition_giant_tour_logged

    original_ruin_and_recreate = rfcs.ruin_and_recreate
    ruin_signature = inspect.signature(original_ruin_and_recreate)
    def ruin_and_recreate_logged(*args, **kwargs):
        tracker["iteration"] += 1
        result = original_ruin_and_recreate(*args, **kwargs)
        if result[0] is None:
            num_remove = ruin_signature.bind(*args, **kwargs).arguments.get("num_remove")
            row("recreate_failed", detail=f"{num_remove} customers removed")
        else:
            tracker["step"] = "ruin_recreate"
        return result
    rfcs.ruin_and_recreate = ruin_and_recreate_logged

    original_improve_between_routes = rfcs.improve_between_routes
    def improve_between_routes_logged(*args, **kwargs):
        tracker["step"] = "relocate_swap"
        return original_improve_between_routes(*args, **kwargs)
    rfcs.improve_between_routes = improve_between_routes_logged

    original_improve_within_routes = rfcs.improve_within_routes
    def improve_within_routes_logged(*args, **kwargs):
        tracker["step"] = "two_opt"
        return original_improve_within_routes(*args, **kwargs)
    rfcs.improve_within_routes = improve_within_routes_logged

    original_check_and_update_new_best = rfcs.check_and_update_new_best
    def check_and_update_new_best_logged(routes, state):
        best_before = state["best_cost"]
        cost = original_check_and_update_new_best(routes, state)
        tracker["iteration"] = state["iteration"]
        tracker["best_cost"] = state["best_cost"]
        row(tracker["step"], cost, len(routes), new_best=state["best_cost"] < best_before)
        return cost
    rfcs.check_and_update_new_best = check_and_update_new_best_logged


def find_solution_with_log(cust_demands, dist_matrix, vehicle_cap, num_vehicles, best, log_info):
    '''
    Process target for run_benchmarks.py: attaches the log inside the solver process,
    then runs final_solution.find_solution exactly as run_algo would.
    '''
    attach(log_info, cust_demands, dist_matrix, vehicle_cap, num_vehicles)
    return final_solution.find_solution(cust_demands, dist_matrix, vehicle_cap, num_vehicles, best)
