import numpy as np


def assign_opt(dist_mat_in):
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
        max_finite = dist_mat[np.nonzero(np.isfinite(dist_mat))].max()
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

    while star_row < n_rows:
        new_star_mat[star_row, star_col] = False

        prime_row = star_row
        prime_col = 0
        for ii in range(0, n_cols):
            if prime_mat[prime_row, ii]:
                prime_col = ii
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
