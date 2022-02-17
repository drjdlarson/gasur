"""Implements general sampling methods."""
import numpy as np


def gibbs(in_costs, num_iters, rng=None):
    """Implements a Gibbs sampler.

    Notes
    -----
    Computationally efficient form of the Metropolis Hastings Markov Chain
    Monte Carlo algorithm, useful when direct sampling of the target
    multivariate distribution is difficult. Generates samples by selecting
    from a histogram constructed from the distribution and randomly sampling
    from the most likely bins in the histogram. This is based on the sampler in
    :cite:`Vo2017_AnEfficientImplementationoftheGeneralizedLabeledMultiBernoulliFilter`.

    Parameters
    ----------
    in_costs : N x M numpy array
        Cost matrix.
    num_iters : int
        Number of iterations to run.
    rng : numpy random generator, optional
        Random number generator to be used when sampling. The default is None
        which implies :code:`default_rng()`

    Returns
    -------
    assignments : M (max size) x N numpy array
        The unique entries from the sampling.
    costs : M x 1 numpy array
        Cost of each assignment.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Determine size of cost matrix
    cost_size = np.shape(in_costs)

    # Initialize assignment and cost matrices
    assignments = np.zeros((num_iters, cost_size[0]))
    costs = np.zeros((num_iters, 1))

    # Initialize current solution
    cur_soln = np.arange(cost_size[0], 2 * cost_size[0])

    assignments[0, ] = cur_soln
    sub_ind = np.ravel_multi_index((np.arange(0, cost_size[0]), cur_soln),
                                   dims=cost_size)
    flat_in_costs = in_costs.ravel()
    costs[0] = np.sum(flat_in_costs[sub_ind])

    # Loop over all possible assignments and determine costs
    for sol in range(1, num_iters):
        for var in range(0, cost_size[0]):
            temp_samp = np.exp(-in_costs[var, ])
            temp_samp_ind = cur_soln[np.concatenate((np.arange(0, var),
                                                     np.arange(var + 1, len(cur_soln))))]
            temp_samp[temp_samp_ind] = 0
            old_ind_temp = temp_samp > 0
            old_ind = np.array([])
            for ii in range(0, len(old_ind_temp)):
                if old_ind_temp[ii]:
                    old_ind = np.append(old_ind, int(ii))

            temp_samp = temp_samp[old_ind.astype(int)]
            hist_in_array = np.concatenate((np.array([0]),
                                            np.cumsum(temp_samp[:]) / np.sum(temp_samp)))

            cur_soln[var] = np.digitize(rng.uniform(size=(1, 1)),
                                        hist_in_array) - 1
            if old_ind.size != 0:
                cur_soln[var] = old_ind[cur_soln[var]]

        assignments[sol, ] = cur_soln
        sub_ind = np.ravel_multi_index((np.arange(0, cost_size[0]), cur_soln),
                                       dims=cost_size)
        costs[sol] = np.sum(in_costs.flatten()[sub_ind])

    [assignments, I] = np.unique(assignments, return_index=True, axis=0)
    costs = costs[I]

    return assignments, costs
