import numpy as np
from itertools import product
from collections import deque, defaultdict

TOL = 1e-9


def balance(s, d, C):
    S, D = s.sum(), d.sum()
    if abs(S - D) < TOL:
        return s.copy(), d.copy(), C.copy()
    if S < D:
        add = D - S
        s2 = np.append(s, add)
        C2 = np.vstack([C, np.zeros((1, C.shape[1]))])
        return s2, d.copy(), C2
    else:
        add = S - D
        d2 = np.append(d, add)
        C2 = np.hstack([C, np.zeros((C.shape[0], 1))])
        return s.copy(), d2, C2


def northwest_corner(s, d):
    s = s.copy()
    d = d.copy()
    m, n = len(s), len(d)
    X = np.zeros((m, n))
    i = j = 0
    while i < m and j < n:
        x = min(s[i], d[j])
        X[i, j] = x
        s[i] -= x
        d[j] -= x
        if s[i] <= TOL:
            i += 1
        elif d[j] <= TOL:
            j += 1
    return X


def basic_cells(X):
    return {(i, j)
            for i in range(X.shape[0])
            for j in range(X.shape[1]) if X[i, j] > TOL}


def compute_uv(C, B):
    m, n = C.shape
    u = np.full(m, np.nan)
    v = np.full(n, np.nan)
    rows = defaultdict(list)
    cols = defaultdict(list)
    for (i, j) in B:
        rows[i].append(j)
        cols[j].append(i)

    seen_r = [False] * m
    seen_c = [False] * n
    for i0 in range(m):
        if seen_r[i0]:
            continue
        u[i0] = 0.0
        q = deque([('r', i0)])
        seen_r[i0] = True
        while q:
            kind, idx = q.popleft()
            if kind == 'r':
                i = idx
                for j in rows[i]:
                    if np.isnan(v[j]):
                        v[j] = C[i, j] - u[i]
                    if not seen_c[j]:
                        seen_c[j] = True
                        q.append(('c', j))
            else:
                j = idx
                for i in cols[j]:
                    if np.isnan(u[i]):
                        u[i] = C[i, j] - v[j]
                    if not seen_r[i]:
                        seen_r[i] = True
                        q.append(('r', i))
    return np.nan_to_num(u, nan=0.0), np.nan_to_num(v, nan=0.0)


def reduced_costs(C, u, v):
    return C - u[:, None] - v[None, :]


def find_cycle_through(B, start):
    ext = sorted(set(B) | {start})
    rows = defaultdict(list)
    cols = defaultdict(list)
    for (i, j) in ext:
        rows[i].append((i, j))
        cols[j].append((i, j))

    def dfs(node, use_row, path, used):
        if node == start and len(path) >= 5 and len(path) % 2 == 1:
            return path
        neighbors = rows[node[0]] if use_row else cols[node[1]]
        neighbors = list(reversed(neighbors))
        for nxt in neighbors:
            if nxt == node:
                continue
            if path and nxt == path[-1]:
                continue
            if nxt in used and nxt != start:
                continue
            path.append(nxt)
            used.add(nxt)
            res = dfs(nxt, not use_row, path, used)
            if res is not None:
                return res
            used.discard(nxt)
            path.pop()
        return None

    p = [start]
    u = {start}
    res = dfs(start, True, p, u)
    if res is not None:
        return res
    p = [start]
    u = {start}
    return dfs(start, False, p, u)


def modi_optimize(s, d, C, max_iter=5000):
    X = northwest_corner(s, d)
    m, n = X.shape
    for _ in range(max_iter):
        B = basic_cells(X)
        u, v = compute_uv(C, B)
        R = reduced_costs(C, u, v)
        enter, best = None, 0.0
        for i, j in product(range(m), range(n)):
            if (i, j) not in B and not np.isnan(R[i, j]) and R[i,
                                                               j] < best - TOL:
                best = R[i, j]
                enter = (i, j)
        if enter is None:
            return X
        cyc = find_cycle_through(B, enter)
        if cyc is None:
            X[enter] = 0.0
            continue
        cyc = cyc[:-1]
        minus = cyc[1::2]
        theta = min(X[i, j] for (i, j) in minus)
        add = True
        for (i, j) in cyc:
            if add:
                X[i, j] += theta
            else:
                X[i, j] -= theta
            add = not add
        X[X < TOL] = 0.0
    return X


def total_cost(X, C):
    return float((X * C).sum())


DEMAND = np.array([300, 150, 300, 150, 250], dtype=float)
SUPPLY = np.array([150, 250, 250, 150, 150], dtype=float)
COST = np.array([[2, 1, 3, 1, 5], [8, 3, 7, 4, 6], [6, 4, 9, 3, 4],
                 [5, 2, 1, 2, 3], [4, 6, 2, 3, 4]],
                dtype=float)

s_bal, d_bal, C_bal = balance(SUPPLY, DEMAND, COST)
X = modi_optimize(s_bal, d_bal, C_bal)
print(total_cost(X, C_bal))
print(X)
