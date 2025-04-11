import numpy as np
import scipy.linalg
from . import CrystalStructure


class Strain(object):
    def __init__(self, crys):
        self._crys = crys

    def createStrainedCrystal(self, eps):
        R = np.array([[1 + eps[0], eps[3], eps[5]], [eps[3], 1 + eps[1], eps[4]], [eps[5], eps[4], 1 + eps[2]]])
        return CrystalStructure(R.dot(self._crys.unit.T).T, self._crys.atoms)

    def calculateStrain(self, ref):
        R = self._crys.unit.T.dot(np.linalg.inv(ref.unit.T))
        U, P = scipy.linalg.polar(R)
        return (P[0, 0] - 1, P[1, 1] - 1, P[2, 2] - 1, P[0, 1], P[1, 2], P[0, 2])
