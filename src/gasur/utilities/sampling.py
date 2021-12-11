import numpy as np


def gibbs(in_costs, m):
    # Determine size of cost matrix
    cost_size = np.shape(in_costs)

    # Initialize assignment and cost matrices
    assignments = np.zeros((m, cost_size[0]))
    costs = np.zeros((m, 1))

    # Initialize current solution
    # cur_soln = np.arange(cost_size[0]+1, 2*cost_size[0]+1)  # Replicates Matlab syntax
    cur_soln = np.arange(cost_size[0], 2*cost_size[0])

    assignments[0, ] = cur_soln
    sub_ind = np.ravel_multi_index((np.arange(0, cost_size[0]), cur_soln),
                                   dims=cost_size)
    flat_in_costs = in_costs.flatten()
    costs[0] = np.sum(flat_in_costs[sub_ind])

    # Loop over all possible assignments and determine costs
    for sol in range(1, m):
        for var in range(0, cost_size[0]):
            temp_samp = np.exp(-in_costs[var, ])
            temp_samp_ind = cur_soln[np.concatenate((np.arange(0, var),
                                                     np.arange(var+1, len(cur_soln))))]
            temp_samp[temp_samp_ind] = 0
            old_ind_temp = temp_samp > 0
            old_ind = np.array([])
            for ii in range(0, len(old_ind_temp)):
                if old_ind_temp[ii] == True:
                    old_ind = np.append(old_ind, int(ii))

            temp_samp = temp_samp[old_ind.astype(int)]
            #finish this, use the histc function in matlab to help
            hist_in_array = np.concatenate((np.array([0]),
                                            np.cumsum(temp_samp[:])/np.sum(temp_samp)))

            # cur_soln[var] = np.digitize(np.random.rand(1,1), hist_in_array)

            # digitize is correct function to use, but think i need to
            # subtract by 1 to keep indexes consistent between matlab and python
            cur_soln[var] = np.digitize(np.random.rand(1,1), hist_in_array)-1
            if old_ind.size != 0:
                cur_soln[var] = old_ind[cur_soln[var]]

        assignments[sol, ] = cur_soln
        sub_ind = np.ravel_multi_index((np.arange(0, cost_size[0]), cur_soln),
                                       dims=cost_size)
        costs[sol] = np.sum(in_costs.flatten()[sub_ind])

    [C, I] = np.unique(assignments, return_index=True, axis=0)  #  may need to use axis = 1 or axis = 0
    assignments = C
    # assignments = np.array([C]).T
    costs = costs[I]
    # costs = costs[int(I)]

    return assignments, costs
