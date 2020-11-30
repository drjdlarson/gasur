""" Implements helper functions for graph search algorithms.

This module contains the longer helper functions for the graph search
algorithms.
"""
import numpy as np


def assign_opt(dist_mat_in):
    """ Optimal assignment used by Murty's m-best assignment algorithm.

    This ports the implementation from
    `here <https://ba-tuong.vo-au.com/codes.html>`_.
    to python.

    Args:
        dist_mat_in (numpy array): Distance matrix to calculate assignment from

    Returns:
        tuple containing

            - assign (numpy array): Optimal assignment
            - cost (float): Cost of the assignment
    """

    dist_mat = dist_mat_in.copy()
    if len(dist_mat.shape) == 1:
        dist_mat = dist_mat.reshape((1, dist_mat.shape[0]))
    elif len(dist_mat.shape) > 2:
        raise RuntimeError('Distance matrix must have at most 2 dimensions')
    dist_mat_orig = dist_mat.copy()

    (n_rows, n_cols) = dist_mat.shape
    assign = -np.ones(n_rows, dtype=int)
    if (np.isfinite(dist_mat) & (dist_mat < 0)).any():
        raise RuntimeError('All matrix values must be non-negative')

    if np.isinf(dist_mat).all():
        return (assign, 0)

    # replace any infite values with a large number
    if np.isinf(dist_mat).any():
        max_finite = dist_mat[np.nonzero(np.isfinite(dist_mat))]
        if len(max_finite) == 0:
            return (assign, 0)

        max_finite = max_finite.max()
        if max_finite > 0:
            inf_val = 10 * max_finite * n_rows * n_cols
        else:
            inf_val = 10
        dist_mat[np.nonzero(np.isinf(dist_mat))] = inf_val

    covered_cols = [False] * n_cols
    covered_rows = [False] * n_rows
    star_mat = np.zeros((n_rows, n_cols), dtype=bool)
    prime_mat = np.zeros((n_rows, n_cols), dtype=bool)

    if n_rows <= n_cols:
        min_dim = n_rows

        # find min of each row and subract from each element in the row
        dist_mat = dist_mat - dist_mat.min(axis=1).reshape((n_rows, 1))

        # steps 1 and 2a
        for row in range(0, n_rows):
            for col in range(0, n_cols):
                if dist_mat[row, col] == 0:
                    if not covered_cols[col]:
                        star_mat[row, col] = True
                        covered_cols[col] = True
                        break
    else:
        min_dim = n_cols

        # find min of each column and subract from each element in the column
        dist_mat = dist_mat - dist_mat.min(axis=0)

        # steps 1 and 2a
        for col in range(0, n_cols):
            for row in range(0, n_rows):
                if dist_mat[row, col].squeeze() == 0:
                    if not covered_rows[row]:
                        star_mat[row, col] = True
                        covered_cols[col] = True
                        covered_rows[row] = True
                        break

        for row in range(0, n_rows):
            covered_rows[row] = False

    __aop_step2b(assign, dist_mat, star_mat, prime_mat, covered_cols,
                 covered_rows, n_rows, n_cols, min_dim)
    cost = __aop_comp_assign(assign, dist_mat_orig, n_rows)

    return (assign, cost)


def __aop_step2a(assign, dist_mat, star_mat, prime_mat, covered_cols,
                 covered_rows, n_rows, n_cols, min_dim):
    for col in range(0, n_cols):
        for row in range(0, n_rows):
            if star_mat[row, col]:
                covered_cols[col] = True
                break

    __aop_step2b(assign, dist_mat, star_mat, prime_mat, covered_cols,
                 covered_rows, n_rows, n_cols, min_dim)


def __aop_step2b(assign, dist_mat, star_mat, prime_mat, covered_cols,
                 covered_rows, n_rows, n_cols, min_dim):
    n_covered_cols = 0
    for v in covered_cols:
        if v:
            n_covered_cols += 1

    if n_covered_cols == min_dim:
        __aop_build_assign(assign, star_mat, n_rows, n_cols)

    else:
        __aop_step3(assign, dist_mat, star_mat,
                    prime_mat, covered_cols, covered_rows, n_rows,
                    n_cols, min_dim)


def __aop_step3(assign, dist_mat, star_mat, prime_mat, covered_cols,
                covered_rows, n_rows, n_cols, min_dim):
    z_found = True
    while z_found:
        z_found = False
        for col in range(0, n_cols):
            if not covered_cols[col]:
                for row in range(0, n_rows):
                    cond1 = not covered_rows[row]
                    cond2 = dist_mat[row, col] == 0
                    if cond1 and cond2:
                        prime_mat[row, col] = True

                        star_col = 0
                        for ii in range(0, n_cols):
                            if star_mat[row, ii]:
                                star_col = ii
                                break
                            else:
                                star_col = ii + 1

                        if star_col == n_cols:
                            __aop_step4(assign, dist_mat, star_mat,
                                        prime_mat, covered_cols, covered_rows,
                                        n_rows, n_cols, min_dim, row, col)
                            return
                        else:
                            covered_rows[row] = True
                            covered_cols[star_col] = False
                            z_found = True
                            break

    __aop_step5(assign, dist_mat, star_mat, prime_mat, covered_cols,
                covered_rows, n_rows, n_cols, min_dim)


def __aop_step4(assign, dist_mat, star_mat, prime_mat, covered_cols,
                covered_rows, n_rows, n_cols, min_dim, row, col):
    new_star_mat = star_mat.copy()

    new_star_mat[row, col] = True
    star_col = col
    star_row = 0
    for ii in range(0, n_rows):
        star_row = ii
        if star_mat[ii, star_col]:
            break
        else:
            star_row = ii + 1


    while star_row < n_rows:
        new_star_mat[star_row, star_col] = False

        prime_row = star_row
        prime_col = 0
        for ii in range(0, n_cols):
            prime_col = ii
            if prime_mat[prime_row, ii]:
                break
            else:
                prime_col = ii + 1

        new_star_mat[prime_row, prime_col] = True

        star_col = prime_col
        for ii in range(0, n_rows):
            if star_mat[ii, star_col]:
                star_row = ii
                break
            else:
                star_row = ii + 1

    for row in range(0, n_rows):
        for col in range(0, n_cols):
            prime_mat[row, col] = False
            star_mat[row, col] = new_star_mat[row, col]

    for row in range(0, n_rows):
        covered_rows[row] = False

    __aop_step2a(assign, dist_mat, star_mat, prime_mat, covered_cols,
                 covered_rows, n_rows, n_cols, min_dim)


def __aop_step5(assign, dist_mat, star_mat, prime_mat, covered_cols,
                covered_rows, n_rows, n_cols, min_dim):
    h = np.inf
    for row in range(0, n_rows):
        if not covered_rows[row]:
            for col in range(0, n_cols):
                if not covered_cols[col]:
                    v = dist_mat[row, col]
                    if v < h:
                        h = v

    for row in range(0, n_rows):
        if covered_rows[row]:
            for col in range(0, n_cols):
                dist_mat[row, col] += h

    for col in range(0, n_cols):
        if not covered_cols[col]:
            for row in range(0, n_rows):
                dist_mat[row, col] -= h

    __aop_step3(assign, dist_mat, star_mat, prime_mat, covered_cols,
                covered_rows, n_rows, n_cols, min_dim)


def __aop_build_assign(assign, star_mat, n_rows, n_cols):
    for row in range(0, n_rows):
        for col in range(0, n_cols):
            if star_mat[row, col]:
                assign[row] = col
                break


def __aop_comp_assign(assign, dist_mat, n_rows):
    cost = 0
    for row in range(0, n_rows):
        col = assign[row]
        if col >= 0:
            v = dist_mat[row, col]
            if np.isfinite(v):
                cost += v
            else:
                assign[row] = -1
    return cost


class AStarNode:
    """
        A node class for A* Pathfinding
        parent is parent of the current Node
        position is current position of the Node in the maze
        g is cost from start to current Node
        h is heuristic based estimated cost for current Node to end Node
        f is total cost of present node i.e. :  f = g + h
        x and y are Cartesian coordinates of the node
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0
        self.x = 0
        self.y = 0

    def __eq__(self, other):
        return self.position == other.position


def astar_return_path(current_node, maze):
    path = []
    no_rows, no_columns = np.shape(maze)

    # here we create the initialized result maze with -1 in every position
    result = [[-1 for i in range(no_columns)] for j in range(no_rows)]
    current = current_node
    total_cost = []
    while current is not None:
        path.append(current.position)
        total_cost.append(current.h)
        current = current.parent

    # Return reversed path as we need to show from start to end path
    path = path[::-1]
    start_value = 0

    # we update the path of start to end found by A-star serch with every step
    # incremented by 1
    for i in range(len(path)):
        result[path[i][0]][path[i][1]] = start_value
        start_value += 1

    inds = np.argwhere(np.array(result) >= 0)
    vals = [result[r][c] for r, c in inds]
    s_inds = np.argsort(vals)
    vals = [vals[ii] for ii in s_inds]

    return (tuple([(inds[ii][0], inds[ii][1]) for ii in s_inds]), max(total_cost))
