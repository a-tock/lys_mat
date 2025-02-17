import numpy as np
from joblib import Parallel, delayed
from lys import Wave


def MakePCF(crystal, element1, element2, sig=0.1, dim=3):
    if dim == 3:
        return _MakePCF_3d(crystal, element1, element2, sig=sig)
    else:
        return _MakePCF_2d(crystal, element1, element2, sig=sig)


def _MakePCF_3d(crystal, element1, element2, sig=0.1):
    rlist = _MakeRlist(crystal, element1, element2, supercell=(1, 1, 1))
    N = len(rlist)
    rho = N / crystal.Volume()
    xdata = np.linspace(1, 10, 200)
    res = Parallel()([delayed(_MakeHist)(N, r, xdata, sig, rho) for r in rlist])
    return Wave(np.sum(res, axis=0), xdata)
#    return {"data": np.sum(res, axis=0), "axes": [xdata]}


def _MakeRlist(crys, element1, element2, supercell=(3, 3, 3)):
    pos1list = _MakePoslist(crys, atype=element1)
    pos2list = _MakePoslist(crys, atype=element2, supercell=supercell)
    rlist = [[np.linalg.norm(p2 - p1, ord=2) for p2 in pos2list]for p1 in pos1list]
    return rlist


def _MakePoslist(crys, atype, supercell=(0, 0, 0)):
    atoms = [a for a in crys.atoms if a.element in atype]
    poslist = []
    for i in range(-supercell[0], supercell[0] + 1):
        for j in range(-supercell[1], supercell[1] + 1):
            for k in range(-supercell[2], supercell[2] + 1):
                s = np.array([i, j, k]).dot(crys.unit)
                for a in atoms:
                    poslist.append(np.dot(np.array(a.Position), crys.unit) + s)
    return poslist


def _gauss(xdata, r, sig):
    return np.exp(-(xdata - r)**2 / (2 * sig**2)) / np.sqrt(2 * np.pi) / sig


def _MakeHist(N, rlist, xdata, sig, rho):
    data = [_gauss(xdata, r, sig) for r in rlist]
    data = np.sum(data, axis=0) / N / (4 * np.pi * xdata * xdata) / rho
    return data


def _MakePCF_2d(crys, atom1="Si", atom2="Si", sig=0.01):
    rlist = _MakeRlist_R(crys, atom1, atom2, supercell=(1, 1, 1))
    N = len(rlist)
    rho = N / crys.Volume()
    xdata = np.linspace(1, 10, 200)
    res = Parallel()([delayed(_MakeHist_R)(
        N, r, xdata, sig, rho, crys) for r in rlist])
    return Wave(np.sum(res, axis=0), xdata)
#    return {"data": np.sum(res, axis=0), "axes": [xdata]}


def _MakeHist_R(N, rlist, xdata, sig, rho, crys):
    data = [_gauss(xdata, r, sig) for r in rlist]
    data = np.sum(data, axis=0) / N / (2 * np.pi * xdata * crys.c) / rho
    return data


def _MakeRlist_R(crys, atom1="Si", atom2="Si", supercell=(2, 2, 2)):
    pos1list = _MakePoslist(crys, atype=atom1)
    pos2list = _MakePoslist(crys, atype=atom2, supercell=supercell)
    rlist = [[np.linalg.norm((p2 - p1)[:1], ord=2) for p2 in pos2list]for p1 in pos1list]
    return rlist
