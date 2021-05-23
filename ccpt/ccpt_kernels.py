import numba as n
import numba.cuda as cuda


# init cc_ids with sequential numbers; each node is its own group
@cuda.jit()
def init_sequential(array, size):
    a = cuda.grid(1)

    if a < size:
        array[a] = a

# initialize each edge-mark to 0
@cuda.jit()
def init_zero(array, size):
    a = cuda.grid(1)

    if a < size:
        array[a] = 0



# initial cc_ids are replaced by a lower-id neighbor, if there exists an edge to one. process is random.
@cuda.jit()
def select_winner_init(d_an, d_an_writeonce, d_edges, num_e):
    a = cuda.grid(1)

    if a < num_e:
        x = d_edges[(0, a*2 + 0)]
        y = d_edges[(0, a*2 + 1)]

        if x > y:
            mx = x
            mn = y
        else:
            mx = y
            mn = x

        write = cuda.atomic.add(d_an_writeonce, mx, 1)
        if write == 0:
            d_an[mx] = mn



# Following greener's algorithm, there are two iterations, one hooking from lower-valued edges to higher-valued edges,
# and the second hooking vice-versa.

# over all edges not marked done (d_mark[i] == 0): hook from lower-valued trees to higher-valued trees.
@cuda.jit()
def connect_low2hi(d_an, d_edges, num_e, flag, d_mark):
    a = cuda.grid(1)
    b = cuda.threadIdx.x

    s_flag = cuda.shared.array(1, n.i4)
    if b == 0: s_flag[0] = 0
    cuda.syncthreads()

    if a < num_e and d_mark[a] == 0:
        x = d_edges[(0, a*2 + 0)]
        y = d_edges[(0, a*2 + 1)]

        a_x = d_an[x]
        a_y = d_an[y]

        if a_x == a_y:
            d_mark[a] = -1

        else:
            if a_x > a_y:
                mx = a_x
                mn = a_y
            else:
                mx = a_y
                mn = a_x

            d_an[mx] = mn
            cuda.atomic.add(s_flag, 0, 1)

    cuda.syncthreads()
    if b == 0 and s_flag[0] >= 1:
        cuda.atomic.add(flag, 0, s_flag[0])

# over all edges not marked done (d_mark[i] == 0): hook from higher-valued trees to lower-valued trees.
@cuda.jit()
def connect_hi2low(d_an, d_edges, num_e, flag, d_mark):
    a = cuda.grid(1)
    b = cuda.threadIdx.x

    s_flag = cuda.shared.array(1, n.i4)
    if b == 0: s_flag[0] = 0
    cuda.syncthreads()

    if a < num_e and d_mark[a] == 0:
        x = d_edges[(0, a*2 + 0)]
        y = d_edges[(0, a*2 + 1)]

        a_x = d_an[x]
        a_y = d_an[y]

        if a_x == a_y:
            d_mark[a] = -1

        else:
            if a_x > a_y:
                mx = a_x
                mn = a_y
            else:
                mx = a_y
                mn = a_x

            d_an[mn] = mx
            cuda.atomic.add(s_flag, 0, 1)

    cuda.syncthreads()
    if b == 0 and s_flag[0] >= 1:
        cuda.atomic.add(flag, 0, s_flag[0])







@cuda.jit()
def pointer_jump(num_n, d_an, flag):
    a = cuda.grid(1)
    b = cuda.threadIdx.x

    s_flag = cuda.shared.array(1, n.i4)
    if b == 0: s_flag[0] = 0
    cuda.syncthreads()

    if a < num_n:
        y = d_an[a]
        x = d_an[y]

        if x != y:
            cuda.atomic.add(s_flag, 0, 1)
            d_an[a] = x

    cuda.syncthreads()
    if b == 0 and s_flag[0] >= 1:
        cuda.atomic.add(flag, 0, 1)


# Nodes are either root nodes or leaf nodes. Leaf nodes are directly connected to the root nodes, hence do not
# need to jump iteratively. Once root nodes have reascertained the new root nodes, the leaf nodes can just jump once
@cuda.jit()
def pointer_jump_masked(num_n, d_an, flag, d_mask):
    a = cuda.grid(1)
    b = cuda.threadIdx.x

    s_flag = cuda.shared.array(1, n.i4)
    if b == 0: s_flag[0] = 0
    cuda.syncthreads()

    if a < num_n and d_mask[a] == 0:
        y = d_an[a]
        x = d_an[y]

        if x != y:
            cuda.atomic.add(s_flag, 0, 1)
            d_an[a] = x
        else:
            d_mask[a] = -1

    cuda.syncthreads()
    if b == 0 and s_flag[0] >= 1:
        cuda.atomic.add(flag, 0, 1)


# The assumption is that all the nodes are root nodes, or not known whether they are leaf nodes.
@cuda.jit()
def pointer_jump_unmasked(num_n, d_an, d_mask):
    a = cuda.grid(1)

    if a < num_n and d_mask[a] == 1:
        y = d_an[a]
        x = d_an[y]
        d_an[a] = x


# check if each node is the parent of itself, and update it as a leaf or root node
@cuda.jit()
def update_mask(d_mask, num_n, d_an):
    a = cuda.grid(1)

    if a < num_n:
        if d_an[a] == a: val = 0
        else: val = 1

        d_mask[a] = val





