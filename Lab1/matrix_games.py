import numpy as np
from itertools import combinations


def saddle_point(A: np.ndarray):
    row_mins = A.min(axis=1)
    col_maxs = A.max(axis=0)
    v_lower = row_mins.max()
    v_upper = col_maxs.min()
    if abs(v_lower - v_upper) <= 1e-12:
        pos = np.argwhere(A == v_lower)
        i, j = pos[0]
        return True, int(i), int(j), float(v_lower)
    return False, None, None, None


def solve_support(A: np.ndarray, R, C):
    k = len(R)
    A_RC = A[np.ix_(R, C)]

    M_p = np.zeros((k + 1, k + 1))
    M_p[:k, :k] = A_RC.T
    M_p[:k, k] = -1
    M_p[k, :k] = 1
    b_p = np.zeros(k + 1)
    b_p[k] = 1
    try:
        sol_p = np.linalg.solve(M_p, b_p)
    except np.linalg.LinAlgError:
        return None
    p_R, v_p = sol_p[:k], sol_p[k]
    if (p_R < -1e-10).any():
        return None

    M_q = np.zeros((k + 1, k + 1))
    M_q[:k, :k] = A_RC
    M_q[:k, k] = -1
    M_q[k, :k] = 1
    b_q = np.zeros(k + 1)
    b_q[k] = 1
    try:
        sol_q = np.linalg.solve(M_q, b_q)
    except np.linalg.LinAlgError:
        return None
    q_C, v_q = sol_q[:k], sol_q[k]
    if (q_C < -1e-10).any():
        return None

    if abs(v_p - v_q) > 1e-7:
        return None
    v = 0.5 * (v_p + v_q)

    for j in range(A.shape[1]):
        if j in C:
            continue
        if (A[np.ix_(R, [j])].T @ p_R).item() < v - 1e-7:
            return None
    for i in range(A.shape[0]):
        if i in R:
            continue
        if (A[np.ix_([i], C)] @ q_C).item() > v + 1e-7:
            return None

    p = np.zeros(A.shape[0])
    q = np.zeros(A.shape[1])
    p[R] = p_R
    q[C] = q_C
    return dict(p=p, q=q, v=float(v))


def solve_game(A: np.ndarray):
    exists, i, j, v = saddle_point(A)
    if exists:
        p = np.zeros(A.shape[0])
        p[i] = 1
        q = np.zeros(A.shape[1])
        q[j] = 1
        return dict(p=p, q=q, v=v)

    m, n = A.shape
    for k in range(1, min(m, n) + 1):
        for R in combinations(range(m), k):
            for C in combinations(range(n), k):
                ans = solve_support(A, list(R), list(C))
                if ans is not None:
                    return ans


A = np.array([[3, 4, 4, 0, 6], [5, 10, 6, 6, 11], [8, 5, 12, 1, 6],
              [11, 12, 10, 6, 6], [2, 9, 7, 9, 3]],
             dtype=float)

sol = solve_game(A)
print("v =", sol["v"])
print("P =", np.round(sol["p"], 2))
print("Q =", np.round(sol["q"], 2))
