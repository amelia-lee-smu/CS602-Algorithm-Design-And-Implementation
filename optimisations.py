EPS = 1e-9

def two_opt(cycle, dist_matrix):
    '''
    Repeatedly replaces edges (a,b) and (c,d) with (a,c) and (b,d) whenever it shortens the tour
    (e.g. uncrossing two crossed edges), until no further improvement can be made.
    '''
    num_customers = len(cycle)
    improved = True

    while improved:
        improved = False
        for i in range(num_customers - 1):
            for j in range(i + 2, num_customers):

                # do nothing if edges are adjacent
                if i == 0 and j == num_customers - 1:
                    continue

                a, b = cycle[i], cycle[i + 1]
                c, d = cycle[j], cycle[(j + 1) % num_customers]         # j might go past the end of the list, so a wrap aroung might be needed
                delta = dist_matrix[a][c] + dist_matrix[b][d] - dist_matrix[a][b] - dist_matrix[c][d]

                # if it reduces the distance, swap the nodes
                if delta < -EPS:
                    cycle[i + 1:j + 1] = reversed(cycle[i + 1:j + 1])
                    improved = True

    return cycle

def three_opt(cycle, dist_matrix):
    '''
    Repeatedly removes three edges (a,b), (c,d) and (e,f) and tries reconnecting the segments
    (reversing a segment, or swapping two segments), keeping any change that shortens the tour,
    until no further improvement can be made.
    '''
    num_customers = len(cycle)
    dist = dist_matrix
    improved = True

    while improved:
        improved = False
        for i in range(num_customers):
            for j in range(i + 2, num_customers):
                for k in range(j + 2, num_customers + (i > 0)):
                    a, b = cycle[i - 1], cycle[i]
                    c, d = cycle[j - 1], cycle[j]
                    e, f = cycle[k - 1], cycle[k % num_customers]
                    d0 = dist[a][b] + dist[c][d] + dist[e][f]
                    d1 = dist[a][c] + dist[b][d] + dist[e][f]
                    d2 = dist[a][b] + dist[c][e] + dist[d][f]
                    d3 = dist[a][d] + dist[e][b] + dist[c][f]
                    d4 = dist[f][b] + dist[c][d] + dist[e][a]
                    if d1 < d0 - EPS:
                        cycle[i:j] = reversed(cycle[i:j])
                    elif d2 < d0 - EPS:
                        cycle[j:k] = reversed(cycle[j:k])
                    elif d4 < d0 - EPS:
                        cycle[i:k] = reversed(cycle[i:k])
                    elif d3 < d0 - EPS:
                        cycle[i:k] = cycle[j:k] + cycle[i:j]
                    else:
                        continue
                    improved = True

    return cycle
