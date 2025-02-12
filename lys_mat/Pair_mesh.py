import numpy as np


def makePair(crys, Elem1, Elem2, max_dist, allow_same=False):
    m, size = _makeMesh(crys, max(1, max_dist))
    for i1 in range(1, size[0] + 1):
        for j1 in range(1, size[1] + 1):
            for k1 in range(1, size[2] + 1):
                atoms1 = [at for at in m[i1][j1][k1] if at.Element in Elem1]
                if len(atoms1) == 0:
                    continue
                for i2 in [i1 - 1, i1, i1 + 1]:
                    for j2 in [j1 - 1, j1, j1 + 1]:
                        for k2 in [k1 - 1, k1, k1 + 1]:
                            atoms2 = [at for at in m[i2][j2][k2] if at.Element in Elem2]
                            _connectAtoms(crys, atoms1, atoms2, max_dist, _calcShift(i2, j2, k2, size), allow_same)


def _calcShift(i, j, k, size):
    s = np.array([0, 0, 0])
    if i == 0:
        s = s - np.array([1, 0, 0])
    if j == 0:
        s = s - np.array([0, 1, 0])
    if k == 0:
        s = s - np.array([0, 0, 1])
    if i == size[0] + 1:
        s = s + np.array([1, 0, 0])
    if j == size[1] + 1:
        s = s + np.array([0, 1, 0])
    if k == size[2] + 1:
        s = s + np.array([0, 0, 1])
    return s


def _connectAtoms(crys, atoms1, atoms2, max_dist, shift, allow_same):
    for at1 in atoms1:
        p1 = np.array(at1.Position)
        for at2 in atoms2:
            p2 = np.array(at2.Position) + shift
            r_ij = crys.unit.T.dot(p1 - p2)
            d = np.linalg.norm(r_ij)
            if 0 < d <= max_dist:
                if hasattr(at1, "pair"):
                    if (not at2 in at1.pair) or allow_same:
                        at1.pair.append(at2)
                        at1.r_ij.append(r_ij)
                else:
                    at1.pair = [at2]
                    at1.r_ij = [r_ij]


def _makeMesh(crys, max_dist):
    size = (max(1, int(crys.a // max_dist)),
            max(1, int(crys.b // max_dist)), max(1, int(crys.c // max_dist)))
    mesh = [[[[] for k in range(size[2] + 2)] for j in range(size[1] + 2)] for i in range(size[0] + 2)]
    for at in crys.atoms:
        index = int(at.Position[0] * size[0]) + 1, int(at.Position[1] * size[1]) + 1, int(at.Position[2] * size[2]) + 1
        mesh[index[0]][index[1]][index[2]].append(at)
    if crys.periodic:
        # Corner
        for f1, t1 in [[1, size[0] + 1], [size[0], 0]]:
            for f2, t2 in [[1, size[1] + 1], [size[1], 0]]:
                for f3, t3 in [[1, size[2] + 1], [size[2], 0]]:
                    for at in mesh[f1][f2][f3]:
                        mesh[t1][t2][t3].append(at)
        # Edge
        for i in range(1, size[0] + 1):
            for f2, t2 in [[1, size[1] + 1], [size[1], 0]]:
                for f3, t3 in [[1, size[2] + 1], [size[2], 0]]:
                    for at in mesh[i][f2][f3]:
                        mesh[i][t2][t3].append(at)
        for j in range(1, size[1] + 1):
            for f1, t1 in [[1, size[0] + 1], [size[0], 0]]:
                for f3, t3 in [[1, size[2] + 1], [size[2], 0]]:
                    for at in mesh[f1][j][f3]:
                        mesh[t1][j][t3].append(at)
        for k in range(1, size[2] + 1):
            for f1, t1 in [[1, size[0] + 1], [size[0], 0]]:
                for f2, t2 in [[1, size[1] + 1], [size[1], 0]]:
                    for at in mesh[f1][f2][k]:
                        mesh[t1][t2][k].append(at)
        # Plane
        for i in range(1, size[0] + 1):
            for j in range(1, size[1] + 1):
                for f3, t3 in [[1, size[2] + 1], [size[2], 0]]:
                    for at in mesh[i][j][f3]:
                        mesh[i][j][t3].append(at)
        for j in range(1, size[1] + 1):
            for k in range(1, size[2] + 1):
                for f1, t1 in [[1, size[0] + 1], [size[0], 0]]:
                    for at in mesh[f1][j][k]:
                        mesh[t1][j][k].append(at)
        for k in range(1, size[2] + 1):
            for i in range(1, size[0] + 1):
                for f2, t2 in [[1, size[1] + 1], [size[1], 0]]:
                    for at in mesh[i][f2][k]:
                        mesh[i][t2][k].append(at)
    return mesh, size
