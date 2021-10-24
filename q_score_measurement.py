""" Q-Score approximation. Author: Yukai Qiao. """

import numpy as np

def Q_divide(B, g, init):
    Bg = B.copy()
    if (not init):
        for i in range(len(g)):
            Bg[i, i] -= sum(B[i, :])
    w, v = np.linalg.eigh(Bg)
    signs = v[:,np.argmax(w)]
    g1 = []
    g2 = []
    s = np.zeros(len(g))
    for i in range(len(g)):
        if (signs[i] > 0):
            g1.append(g[i])
            s[i] = 1
        else:
            g2.append(g[i])
            s[i] = -1
    Q = s@Bg@s
    if (Q <= 0 or g1 == [] or g2 == []):
        return 0, [g]
    Q1, groups1 = Q_divide(Bg[g1][:,g1], list(range(len(g1))), False)
    Q2, groups2 = Q_divide(Bg[g2][:,g2], list(range(len(g2))), False)
    groups = []
    for group in groups1:
        tmp = []
        for x in group:
            tmp.append(g1[x])
        groups.append(tmp)
    for group in groups2:
        tmp = []
        for x in group:
            tmp.append(g2[x])
        groups.append(tmp)
    return Q + Q1 + Q2, groups

# Modify usages of 'graph' to your implementation of graph representation.
def compute_B(graph):
    n = graph.node_count
    m = graph.edge_count
    B = np.zeros((n, n))
    k = np.zeros(n)
    for edge in graph:
        B[edge.source, edge.target] += 1
        B[edge.target, edge.source] += 1
        k[edge.source] += 1
        k[edge.target] += 1
    for i in range(n):
        for j in range(n):
            B[i, j] -= k[i] * k[j] / (2 * m)
    return B

# Modify usages of 'graph' to your implementation of graph representation.
def Q(graph):
    n = graph.node_count
    m = graph.edge_count
    if (m < 1):
        return 0, []
    B = compute_B(graph)
    Q, groups = Q_divide(B, list(range(n)), True)
    Q = Q / (4 * m)
    return (Q, groups)