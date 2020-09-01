# -*- coding: utf-8 -*-
import numpy as np
from warnings import warn
from scipy.linalg import block_diag

from gasur.utilities.graphs_subroutines import assign_opt

def k_shortest(log_cost_in, k):
    if k == 0:
        paths = []
        costs = []
        return (paths, costs)
    log_cost = np.squeeze(log_cost_in)
    num_states = log_cost.size
    cost_mat = np.zeros((num_states, num_states))

    # sort in descending order and save index
    sort_inds = np.argsort(log_cost)
    sort_inds = sort_inds[::-1]
    log_cost = [log_cost[ii] for ii in sort_inds]

    for ii in range(0, num_states):
        if ii >= 1:
            cost_mat[0:ii, ii] = log_cost[ii]

    cost_pad = np.zeros((num_states + 2, num_states + 2))
    cost_pad[0, 1:-1] = log_cost
    cost_pad[0, -1] = np.finfo(float).eps
    cost_pad[1:-1, -1] = np.finfo(float).eps
    cost_pad[1:-1, 1:-1] = cost_mat

    (paths, costs) = __k_short_helper(cost_pad, 0, num_states + 1, k)

    for p in range(0, len(paths)):
        if np.array_equal(np.array(paths[p]), np.array([0, num_states + 1])):
            paths[p] = []
        else:
            sub = paths[p][1:-1]
            for ii in range(0, len(sub)):
                sub[ii] = sub[ii] - 1
            paths[p] = [sort_inds[ii] for ii in sub]

    return (paths, costs)


def __k_short_helper(net_cost_mat, src, dst, k_paths):
    if src > net_cost_mat.shape[0] or dst > net_cost_mat.shape[0]:
        msg = 'Source or destination nodes not part of cost matrix'
        warn(msg, RuntimeWarning)
        return ([], [])

    (cost, path, _) = bfm_shortest_path(net_cost_mat, src, dst)
    if len(path) == 0:
        return ([], [])

    P = []
    P.append((path, cost))
    cur_p = 0

    X = []
    X.append((len(P) - 1, path, cost))

    S = []
    S.append(path[0])

    shortest_paths = []
    shortest_paths.append(path)
    tot_costs = []
    tot_costs.append(cost)

    while (len(shortest_paths) < k_paths) and (len(X) != 0):
        for ii in range(0, len(X)):
            if X[ii][0] == cur_p:
                del X[ii]
                break
        P_ = P[cur_p][0].copy()

        w = S[cur_p]
        for ii in range(0, len(P_)):
            if w == P_[ii]:
                w_ind_in_path = ii

        for ind_dev_vert in range(w_ind_in_path, len(P_) - 1):
            temp_cost_mat = net_cost_mat.copy()
            for ii in range(0, ind_dev_vert - 1):
                v = P_(ii)
                temp_cost_mat[v, :] = np.inf
                temp_cost_mat[:, v] = np.inf

            sp_same_sub_p = []
            sp_same_sub_p.append(P_)
            for sp in shortest_paths:
                if len(sp) > ind_dev_vert:
                    if np.array_equal(np.array(P_[0:ind_dev_vert + 1]),
                                      np.array(sp[0:ind_dev_vert + 1])):
                        sp_same_sub_p.append(sp)
            v_ = P_[ind_dev_vert]
            for sp in sp_same_sub_p:
                nxt = sp[ind_dev_vert + 1]
                temp_cost_mat[v_, nxt] = np.inf

            sub_P = P_[0:ind_dev_vert+1]
            cost_sub_P = 0
            for ii in range(0, len(sub_P) - 1):
                cost_sub_P += net_cost_mat[sub_P[ii], sub_P[ii + 1]]

            (c, dev_p, _) = bfm_shortest_path(temp_cost_mat, P_[ind_dev_vert],
                                              dst)
            if len(dev_p) > 0:
                tmp_path = sub_P[0:-2] + dev_p
                P.append((tmp_path, cost_sub_P + c))
                S.append(P_[ind_dev_vert])
                X.append((len(P) - 1, P[-1][0], P[-1][1]))

        if len(X) > 0:
            shortest_x_cost = X[0][2]
            shortest_x = X[0][0]
            for ii in range(1, len(X)):
                if X[ii][2] < shortest_x_cost:
                    shortest_x = X[ii][0]
                    shortest_x_cost = X[ii][2]

            cur_p = shortest_x
            shortest_paths.append(P[cur_p][0])
            tot_costs.append(P[cur_p][1])

    return (shortest_paths, tot_costs)


def bfm_shortest_path(ncm, src, dst):
    # Basic form of the Bellman-Ford-Moore Shortest Path algorithm
    # Assumes G(N,A) is in sparse adjacency matrix form, with |N| = n,
    # |A| = m = nnz(G). It constructs a shortest path tree with root r whichs
    # is represented by an vector of parent 'pointers' p, along with a vector
    # of shortest path lengths D.
    # Complexity: O(mn)
    # Derek O'Connor, 19 Jan, 11 Sep 2012.  derekroconnor@eircom.net
    (pred, D, _) = __bfm_helper(ncm, src)
    dist = D[dst]

    if dist == np.inf:
        path = []
    else:
        path = [dst]
        while not(path[0] == src):
            path = [pred[path[0]]] + path

    return (dist, path, pred)


def __bfm_helper(G, r):
    # Copyright (c) 2012, Derek O'Connor
    # All rights reserved.
    #
    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions are
    # met:
    #
    #     * Redistributions of source code must retain the above copyright
    #       notice, this list of conditions and the following disclaimer.
    #     * Redistributions in binary form must reproduce the above copyright
    #       notice, this list of conditions and the following disclaimer in
    #       the documentation and/or other materials provided with the distribution
    #
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    # POSSIBILITY OF SUCH DAMAGE.
    def init(G):
        # Transforms the sparse matrix G into the list-of-arcs form
        # and intializes the shortest path parent-pointer and distance
        # arrays, p and D.
        # Derek O'Connor, 21 Jan 2012
        inds = np.argwhere(G.T >= np.finfo(float).eps)
        tail = inds[:, 1]
        head = inds[:, 0]
        W = G[tail, head]
        sort_inds = np.lexsort((tail, head))
        tail = [tail[ii] for ii in sort_inds]
        head = [head[ii] for ii in sort_inds]
        W = [W[ii] for ii in sort_inds]
        n = G.shape[1]
        m = np.count_nonzero(G)
        p = [0] * n
        D = np.inf * np.ones((n, 1))
        return (m, n, p, D, tail, head, W)

    (m, n, p, D, tail, head, W) = init(G)
    p[r] = int(0)
    D[r] = 0
    for ii in range(0, n - 1):
        optimal = True
        for arc in range(0, m):
            u = tail[arc]
            v = head[arc]
            duv = W[arc]
            if D[v] > D[u] + duv:
                D[v] = D[u] + duv
                p[v] = u
                optimal = False

        n_iters = ii + 1
        if optimal:
            break

    return (p, D, n_iters)


def murty_m_best(cost_mat, m):
    if len(cost_mat.shape) == 1 or len(cost_mat.shape) > 2:
        raise RuntimeError('Cost matrix must be 2D array')

    if m == 0:
        return ([], [])
    blk = -np.log(np.ones((cost_mat.shape[0], cost_mat.shape[0])))

    cm = np.hstack((cost_mat, blk))
    x = cm.min()
    cm = cm - x

    (assigns, costs) = __murty_helper(cm, m)

    for (ii, a) in enumerate(assigns):
        costs[ii] += x * np.count_nonzero(a >= 0)
    assigns += 1

    # remove extra entries
    assigns = assigns[:, 0:cost_mat.shape[0]]
    # dummy assignmets are clutter
    assigns[np.where(assigns > cost_mat.shape[1])] = 0
    assigns = assigns.astype(int)
    return (assigns, costs)


def __murty_helper(p0, m):
    # MURTY   Murty's Algorithm for m-best ranked optimal assignment problem
    # Port of Ba Tuong Vo's 2015 Matlab code
    # NOTE: the assignment is zero based indexing
    (s0, c0) = assign_opt(p0)
    s0 = s0.T

    if m == 1:
        return (s0, c0)

    (n_rows, n_cols) = p0.shape
    # preallocate arrays
    blk_sz = 1000
    ans_lst_P = np.zeros((n_rows, n_cols, blk_sz))
    ans_lst_S = np.zeros((n_rows, blk_sz), dtype=int)
    ans_lst_C = np.nan * np.ones(blk_sz)

    ans_lst_P[:, :, 0] = p0
    ans_lst_S[:, 0] = s0.T
    ans_lst_C[0] = c0
    ans_nxt_ind = 1

    assigns = np.nan * np.ones((n_rows, m))
    costs = np.zeros(m)

    for ii in range(0, m):
        # if cleared break
        if np.isnan(ans_lst_C).all():
            assigns = assigns[:, 0:ans_nxt_ind]
            costs = costs[0:ans_nxt_ind]
            break

        # find lowest cost solution
        idx_top = np.nanargmin(ans_lst_C[0:ans_nxt_ind])
        assigns[:, ii] = ans_lst_S[:, idx_top]
        costs[ii] = ans_lst_C[idx_top]

        P_now = ans_lst_P[:, :, idx_top]
        S_now = ans_lst_S[:, idx_top]

        ans_lst_C[idx_top] = np.nan

        for (aw, aj) in enumerate(S_now):
            if aj >= 0:
                P_tmp = P_now.copy()
                if aj <= n_cols - n_rows - 1:
                    P_tmp[aw, aj] = np.inf
                else:
                    P_tmp[aw, (n_cols - n_rows):] = np.inf
                (S_tmp, C_tmp) = assign_opt(P_tmp)

                S_tmp = S_tmp.T
                if (S_tmp >= 0).all():
                    # allocate more space as needed
                    if ans_nxt_ind >= len(ans_lst_C):
                        ans_lst_P = np.concatenate((ans_lst_P,
                                                    np.zeros((n_rows, n_cols,
                                                              blk_sz))),
                                                   axis=2)
                        ans_lst_S = np.concatenate((ans_lst_S,
                                                    np.zeros((n_rows, blk_sz),
                                                             dtype=int)),
                                                   axis=1)
                        ans_lst_C = np.hstack((ans_lst_C,
                                               np.nan * np.ones(blk_sz)))

                    ans_lst_P[:, :, ans_nxt_ind] = P_tmp
                    ans_lst_S[:, ans_nxt_ind] = S_tmp
                    ans_lst_C[ans_nxt_ind] = C_tmp
                    ans_nxt_ind += 1

                    v_tmp = P_now[aw, aj]
                    P_now[aw, :] = np.inf
                    P_now[:, aj] = np.inf
                    P_now[aw, aj] = v_tmp

    return (assigns.T, costs)
