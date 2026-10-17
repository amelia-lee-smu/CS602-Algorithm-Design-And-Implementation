# CS602 Algorithm Design and Implementation (G2 - Group 5)

## Setup

```bash
conda env create -f environment.yml
conda activate algo-design
```


## Input and output format

### Output: 

The total cost on the first line, then one route per line, each starting and ending at the depot (`0`):

```
21.8
0 1 0
0 2 3 0
```


## Algorithms

1. `route_first_cluster_second_solution.py` implements the **route first, cluster second** method from Beasley (1983): 
   - Build a random giant tour through all customers (excluding the depot) and improve it with 2-opt.
   - Partition the giant tour
   - optimise each route using 3-opt
   - (Note: multiple giant tours are created, to reduce the effect of having a bad random start)
