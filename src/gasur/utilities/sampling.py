import numpy as np


def gibbs(in_costs, m):
    # Determine size of cost matrix
    cost_size = np.shape(in_costs)

    # Initialize assignment and cost matrices
    assignments = np.zeros((m, cost_size[0]))
    costs = np.zeros((m, 1))

    # Initialize current solution
    cur_soln = np.arange(cost_size[0]+1, 2*cost_size[0])

    assignments[0, ] = cur_soln
    sub_ind = np.ravel_multi_index((np.arange(0, cost_size[0]), cur_soln),
                                   dims=cost_size)
    costs[0] = np.sum(in_costs[sub_ind])

    # Loop over all possible assignments and determine costs
    for sol in range(1, m):
        for var in range(0, cost_size[0]):
            temp_samp = np.exp(-in_costs[var, ])
            temp_samp_ind = np.concatenate(np.arange((1, var-1)),
                                           np.arange((var+1, len(curr_soln))))
            temp_samp[temp_samp_ind] = 0
            old_ind_temp = temp_samp > 0
            old_ind = np.array([])
            for ii in range(0, len(old_ind_temp)):
                if old_ind_temp[ii] is True:
                    old_ind.append(ii)

            temp_samp = temp_samp[old_ind]
            [hist_n, cur_soln[var]] = np.histogram()
            cur_soln[var] = old_ind[cur_soln[var]]

        assignments[sol, ] = cur_soln
        sub_ind = np.ravel_multi_index((np.arange(0, cost_size[0]), cur_soln),
                                       dims=cost_size)
        costs(sol) = sum(in_costs[sub_ind])

    [C, I] = np.unique(assignments)  #  may need to use axis = 1 or axis = 0
    assignments = C
    costs = costs[I]
