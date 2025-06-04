import numpy as np
from joblib import Parallel, delayed
from lys import Wave


def MakePCF(crystal, element1, element2, sig=0.1, dim=3):
    """
    Create a Pair Correlation Function (PCF) for a given crystal structure.

    Args:
        crystal (CrystalStructure): The crystal structure.
        element1 (str): The first element type.
        element2 (str): The second element type.
        sig (float, optional): The standard deviation for Gaussian smoothing. Defaults to 0.1.
        dim (int, optional): The dimensionality of the PCF (2 or 3). Defaults to 3.

    Returns:
        Wave: The PCF as a Wave object.
    """
    if dim == 3:
        return _MakePCF_3d(crystal, element1, element2, sig=sig)
    else:
        return _MakePCF_2d(crystal, element1, element2, sig=sig)


def _MakePCF_3d(crystal, element1, element2, sig=0.1):
    """
    Create a 3D Pair Correlation Function (PCF) for a given crystal structure.

    Args:
        crystal (CrystalStructure): The crystal structure.
        element1 (str): The first element type.
        element2 (str): The second element type.
        sig (float, optional): The standard deviation for Gaussian smoothing. Defaults to 0.1.

    Returns:
        Wave: The 3D PCF as a Wave object.
    """
    rlist = _MakeRlist(crystal, element1, element2, supercell=(1, 1, 1))
    N = len(rlist)
    rho = N / crystal.Volume()
    xdata = np.linspace(1, 10, 200)
    res = Parallel()([delayed(_MakeHist)(N, r, xdata, sig, rho) for r in rlist])
    return Wave(np.sum(res, axis=0), xdata)


def _MakeRlist(crys, element1, element2, supercell=(3, 3, 3)):
    """
    Create a list of distances between atoms of two elements in a crystal structure.

    Args:
        crys (CrystalStructure): The crystal structure.
        element1 (str): The first element type.
        element2 (str): The second element type.
        supercell (tuple, optional): The dimensions of the supercell. Defaults to (3, 3, 3).

    Returns:
        list: A list of distances between atoms.
    """
    pos1list = _MakePoslist(crys, atype=element1)
    pos2list = _MakePoslist(crys, atype=element2, supercell=supercell)
    rlist = [[np.linalg.norm(p2 - p1, ord=2) for p2 in pos2list]for p1 in pos1list]
    return rlist


def _MakePoslist(crys, atype, supercell=(0, 0, 0)):
    """
    Create a list of atomic positions for a given element type in a crystal structure.

    Args:
        crys (CrystalStructure): The crystal structure.
        atype (str): The element type.
        supercell (tuple, optional): The dimensions of the supercell. Defaults to (0, 0, 0).

    Returns:
        list: A list of atomic positions.
    """
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
    """
    Calculate the Gaussian function for a given distance.

    Args:
        xdata (numpy.array): The x-axis data.
        r (float): The distance.
        sig (float): The standard deviation for Gaussian smoothing.

    Returns:
        numpy.array: The Gaussian function values.
    """
    return np.exp(-(xdata - r)**2 / (2 * sig**2)) / np.sqrt(2 * np.pi) / sig


def _MakeHist(N, rlist, xdata, sig, rho):
    """
    Create a histogram for the Pair Correlation Function (PCF).

    Args:
        N (int): The number of distances.
        rlist (list): The list of distances.
        xdata (numpy.array): The x-axis data.
        sig (float): The standard deviation for Gaussian smoothing.
        rho (float): The density.

    Returns:
        numpy.array: The histogram values.
    """
    data = [_gauss(xdata, r, sig) for r in rlist]
    data = np.sum(data, axis=0) / N / (4 * np.pi * xdata * xdata) / rho
    return data


def _MakePCF_2d(crys, atom1="Si", atom2="Si", sig=0.01):
    """
    Create a 2D Pair Correlation Function (PCF) for a given crystal structure.

    Args:
        crys (CrystalStructure): The crystal structure.
        atom1 (str, optional): The first element type. Defaults to "Si".
        atom2 (str, optional): The second element type. Defaults to "Si".
        sig (float, optional): The standard deviation for Gaussian smoothing. Defaults to 0.01.

    Returns:
        Wave: The 2D PCF as a Wave object.
    """
    rlist = _MakeRlist_R(crys, atom1, atom2, supercell=(1, 1, 1))
    N = len(rlist)
    rho = N / crys.Volume()
    xdata = np.linspace(1, 10, 200)
    res = Parallel()([delayed(_MakeHist_R)(N, r, xdata, sig, rho, crys) for r in rlist])
    return Wave(np.sum(res, axis=0), xdata)


def _MakeHist_R(N, rlist, xdata, sig, rho, crys):
    """
    Create a histogram for the 2D Pair Correlation Function (PCF).

    Args:
        N (int): The number of distances.
        rlist (list): The list of distances.
        xdata (numpy.array): The x-axis data.
        sig (float): The standard deviation for Gaussian smoothing.
        rho (float): The density.
        crys (CrystalStructure): The crystal structure.

    Returns:
        numpy.array: The histogram values.
    """
    data = [_gauss(xdata, r, sig) for r in rlist]
    data = np.sum(data, axis=0) / N / (2 * np.pi * xdata * crys.c) / rho
    return data


def _MakeRlist_R(crys, atom1="Si", atom2="Si", supercell=(2, 2, 2)):
    """
    Create a list of distances between atoms of two elements in a 2D crystal structure.

    Args:
        crys (CrystalStructure): The crystal structure.
        atom1 (str, optional): The first element type. Defaults to "Si".
        atom2 (str, optional): The second element type. Defaults to "Si".
        supercell (tuple, optional): The dimensions of the supercell. Defaults to (2, 2, 2).

    Returns:
        list: A list of distances between atoms.
    """
    pos1list = _MakePoslist(crys, atype=atom1)
    pos2list = _MakePoslist(crys, atype=atom2, supercell=supercell)
    rlist = [[np.linalg.norm((p2 - p1)[:1], ord=2) for p2 in pos2list]for p1 in pos1list]
    return rlist
