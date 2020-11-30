""" Implements graph search algorithms.

This module contains the functions for graph search algorithms.
"""
import numpy as np
from warnings import warn

import gasur.utilities.graphs_subroutines as subs


def k_shortest(log_cost_in, k):
    """ This implents k-shortest path algorithm.

    This ports the implementation from
    `here <https://ba-tuong.vo-au.com/codes.html>`_.
    to python. The following are comments from the original implementation.

    Meral Shirazipour
    This function is based on Yen's k-Shortest Path algorithm (1971)
    This function calls a slightly modified function dijkstra()
    by Xiaodong Wang 2004.
    * log_cost_in must have positive weights/costs
    Modified by BT VO
    Replaces dijkstra's algorithm with Derek O'Connor's
    Bellman-Ford-Moore implementation which allows negative entries in cost
    matrices provided there are no negative cycles and used in GLMB filter
    codes for prediction * netCostMatrix can have negative weights/costs

    Args:
        log_cost_in (Numpy array): Input cost matrix, inf represents absence
            of a link
        k (int): Maximum number of paths to find

    Returns:
        tuple containing:

            - paths (list): List with each element being a list of
              indices into the cost matrix for the shortest path
            - costs (list): List of costs associated with each path
    """

    if k == 0:
        paths = []
        costs = []
        return (paths, costs)

    if log_cost_in.size > 1:
        log_cost = np.squeeze(log_cost_in.copy())
    else:
        log_cost = log_cost_in.copy()

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
                v = P_[ii]
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
    """ This implements the Bellman-Ford-Moore shortest path algorithm.

    This ports the implementation from
    `here <https://ba-tuong.vo-au.com/codes.html>`_.
    to python. The following are comments from the original implementation.

    Copyright (c) 2012, Derek O'Connor
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    Basic form of the Bellman-Ford-Moore Shortest Path algorithm
    Assumes G(N,A) is in sparse adjacency matrix form, with abs(N) = n,
    abs(A) = m = nnz(G). It constructs a shortest path tree with root r whichs
    is represented by an vector of parent 'pointers' p, along with a vector
    of shortest path lengths D.
    Complexity: O(mn)
    Derek O'Connor, 19 Jan, 11 Sep 2012.  derekroconnor@eircom.net

    Unlike the original BFM algorithm, this does an optimality test on the
    SP Tree p which may greatly reduce the number of iters to convergence.
    USE:
    n=10^6; G=sprand(n,n,5/n); r=1; format long g;
    tic; [p,D,iter] = BFMSpathOT(G,r);toc, disp([(1:10)' p(1:10) D(1:10)]);
    WARNING:
    This algorithm performs well on random graphs but may perform
    badly on real problems.

    Args:
        ncm (numpy array): cost matrix
        src (int): source index
        dst (int): destination index

    Returns:
        tuple containing

            - dist (float): Path distance
            - path (list): Indices of the path
            - pred (int): number of iterations
    """
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
    """
    Copyright (c) 2012, Derek O'Connor
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in
          the documentation and/or other materials provided with the distribution

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    """
    def init(G):
        # Transforms the sparse matrix G into the list-of-arcs form
        # and intializes the shortest path parent-pointer and distance
        # arrays, p and D.
        # Derek O'Connor, 21 Jan 2012
        inds = np.argwhere(np.abs(G.T) >= np.finfo(float).eps)
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
    """ This implements Murty's m-best ranked optimal assignment.

    This ports the implementation from
    `here <https://ba-tuong.vo-au.com/codes.html>`_.
    to python. The following are comments from the original implementation.

    MURTY   Murty's Algorithm for m-best ranked optimal assignment problem
    Port of Ba Tuong Vo's 2015 Matlab code
    NOTE: the assignment is zero based indexing

    Args:
        cost_mat (numpy array): Cost matrix
        m (int): Number of best ranked assignments to find

    Returns:
        tuple containing

            - assigns (numpy array): Array of best paths
            - costs (numpy array): Cost of each path
    """
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
    (s0, c0) = subs.assign_opt(p0)
    s0 = s0.T

    if m == 1:
        return (s0.reshape((1, s0.size)), np.array([c0]))

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

                (S_tmp, C_tmp) = subs.assign_opt(P_tmp)

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


def a_star_search(maze, start, end, cost=1):
    """
        Returns a list of tuples as a path from the given start to the given end in the given maze
        :param maze:
        :param cost
        :param start:
        :param end:
        :return:
    """

    # Create start and end node with initized values for g, h and f
    start_node = subs.AStarNode(None, tuple(start))
    start_node.g = start_node.h = start_node.f = 0
    end_node = subs.AStarNode(None, tuple(end))
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both yet_to_visit and visited list
    # in this list we will put all node that are yet_to_visit for exploration.
    # From here we will find the lowest cost node to expand next
    yet_to_visit_list = []

    # in this list we will put all node those already explored so that we don't explore it again
    visited_list = []

    # Add the start node
    yet_to_visit_list.append(start_node)

    # Adding a stop condition. This is to avoid any infinite loop and stop
    # execution after some reasonable number of steps
    outer_iterations = 0
    max_iterations = (len(maze) // 2) ** 10

    # what squares do we search . search movement is left-right-top-bottom
    # (4 movements) from every positon
    move = [[-1, 0],  # go up
            [0, -1],  # go left
            [1, 0],  # go down
            [0, 1],  # go right
            [1, 1],  # diagonals
            [1, -1],
            [-1, 1],
            [-1, -1]]

    """
        1) We first get the current node by comparing all f cost and selecting the lowest cost node for further expansion
        2) Check max iteration reached or not . Set a message and stop execution
        3) Remove the selected node from yet_to_visit list and add this node to visited list
        4) Perofmr Goal test and return the path else perform below steps
        5) For selected node find out all children (use move to find children)
            a) get the current postion for the selected node (this becomes parent node for the children)
            b) check if a valid position exist (boundary will make few nodes invalid)
            c) if any node is a wall then ignore that
            d) add to valid children node list for the selected parent

            For all the children node
                a) if child in visited list then ignore it and try next node
                b) calculate child node g, h and f values
                c) if child in yet_to_visit list then ignore it
                d) else move the child to yet_to_visit list
    """
    #find maze has got how many rows and columns
    no_rows, no_columns = np.shape(maze)

    # Loop until you find the end

    while len(yet_to_visit_list) > 0:

        # Every time any node is referred from yet_to_visit list, counter of limit operation incremented
        outer_iterations += 1


        # Get the current node
        current_node = yet_to_visit_list[0]
        current_index = 0
        for index, item in enumerate(yet_to_visit_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # if we hit this point return the path such as it may be no solution or
        # computation cost is too high
        if outer_iterations > max_iterations:
            print ("giving up on pathfinding too many iterations")
            return subs.astar_return_path(current_node,maze)

        # Pop current node out off yet_to_visit list, add to visited list
        yet_to_visit_list.pop(current_index)
        visited_list.append(current_node)

        # test if goal is reached or not, if yes then return the path
        if current_node == end_node:
            return subs.astar_return_path(current_node, maze)

        # Generate children from all adjacent squares
        children = []

        for new_position in move:

            # Get node position
            node_position = (current_node.position[0] + new_position[0],
                             current_node.position[1] + new_position[1])

            # Make sure within range (check if within maze boundary)
            if (node_position[0] > (no_rows - 1) or
                node_position[0] < 0 or
                node_position[1] > (no_columns -1) or
                node_position[1] < 0 or
                node_position[0] < 0 and node_position[1] < 0 or
                node_position[0] > (no_rows-1) and node_position[1] < 0 or
                node_position[0] < 0 and node_position[1] < (no_columns-1) or
                node_position[0] > (no_rows-1) and node_position[1] > (no_columns-1)):
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = subs.AStarNode(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the visited list (search entire visited list)
            if len([visited_child for visited_child in visited_list
                    if visited_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + cost

            ## Heuristic costs calculated here, this is using eucledian distance
            child.h = (((child.position[0] - end_node.position[0]) ** 2) +
                       ((child.position[1] - end_node.position[1]) ** 2))

            child.f = child.g + child.h

            # Child is already in the yet_to_visit list and g cost is already lower
            if len([i for i in yet_to_visit_list
                    if child == i and child.g > i.g]) > 0:
                continue

            # Add the child to the yet_to_visit list
            yet_to_visit_list.append(child)
