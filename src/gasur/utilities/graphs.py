# -*- coding: utf-8 -*-
import numpy as np
from warnings import warn


def k_shortest(self, log_cost, k):
    if k == 0:
        paths = []
        costs = []
        return (paths, costs)
    num_states = log_cost.size
    cost_mat = np.zeros((num_states, num_states))

    # sort in descending order and save index
    sort_inds = np.argsort(log_cost)
    np.sort(log_cost)
    log_cost = log_cost[::-1]

    for ii in range(0, num_states):
        if ii >= 1:
            cost_mat[0:ii-1, ii] = log_cost[ii]

    cost_pad = np.zeros((num_states + 2, num_states + 2))
    cost_pad[0, 1:-2] = log_cost
    cost_pad[0, -1] = np.finfo(float).eps
    cost_pad[1:-2, -1] = np.finfo(float).eps
    cost_pad[1:-2, 1:-2] = cost_mat

    (paths, costs) = __k_short_helper(cost_pad, 1, num_states + 2, k)

    for p in range(0, len(paths)):
        if (paths[p] == np.array([1, num_states + 2])).all():
            paths[p] = []
        else:
            paths[p] = paths[p][1:-2] - 1
            paths[p] = sort_inds(paths[p])

    return (paths, costs)


def __k_short_helper(net_cost_mat, src, dst, k_paths):
    if src > net_cost_mat.shape[0] or dst > net_cost_mat.shape[0]:
        msg = 'Source or destination nodes not part of cost matrix'
        warn(msg, RuntimeWarning)
        return ([], [])
    k = 1
    (cost, path, _) = bfm_shortest_path(net_cost_mat, src, dst)
    if len(path) == 0:
        return ([], [])

    P = []
    P.append((path, cost))
    cur_p = 1

    X = []
    X.append((len(P), path, cost))

    S = []
    S.append(path[0])

    shortest_paths = []
    shortest_paths.append(path)
    tot_costs = []
    tot_costs.append(cost)

    while k < k_paths and not(len(X) == 0):
        new_X = X
        for ii in range(0, len(X)):
            if X[ii][0] == cur_p:
                del new_X[ii]
        X = new_X
        P_ = P[cur_p][0]

        w = S[cur_p]
        for ii in range(0, len(P_)):
            if w == P_[ii]:
                w_ind_in_path = ii

        for ind_dev_vert in range(w_ind_in_path, len(P_) - 1):
            temp_cost_mat = net_cost_mat
            for ii in range(0, ind_dev_vert - 2):
                v = P_(ii)
                temp_cost_mat[v, :] = np.inf
                temp_cost_mat[:, v] = np.inf

            sp_same_sub_p = []
            sp_same_sub_p.append(P_)
            for sp in shortest_paths:
                if len(sp) >= ind_dev_vert:
                    if P_[0:ind_dev_vert] == sp[0:ind_dev_vert]:
                        sp_same_sub_p.append(sp)
            v_ = P_[ind_dev_vert]
            for sp in sp_same_sub_p:
                nxt = sp[ind_dev_vert + 1]
                temp_cost_mat[v_, nxt] = np.inf

            sub_P = P_[0:ind_dev_vert]
            cost_sub_P = 0
            for ii in range(0, len(sub_P) - 2):
                cost_sub_P += net_cost_mat[sub_P[ii], sub_P[ii + 1]]

            (c, dev_p, _) = bfm_shortest_path(temp_cost_mat, P_[ind_dev_vert],
                                              dst)
            if len(dev_p) > 0:
                tmp_path = sub_P[0:-2]
                tmp_path.append(dev_p)
                P.append((tmp_path, cost_sub_P + c))
                S.append(P_[ind_dev_vert])
                X.append((len(P), P[-1][0], P[-1][1]))

        if len(X) > 0:
            shortest_x_cost = X[0][2]
            shortest_x = X[0][0]
            for ii in range(1, len(X)):
                if X[ii][2] < shortest_x_cost:
                    shortest_x = X[ii][0]
                    shortest_x_cost = X[ii][2]

            cur_p = shortest_x
            k += 1
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
        (tail, head) = np.nonzero(G >= np.finfo(float).eps)
        W = G[tail, head]
        sort_inds = np.argsort(W)
        sort_inds = sort_inds[::-1]
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
