import numpy as np
import sympy as sp

from . import sympyFuncs as spf


class Lattice(object):
    def __init__(self, cell):
        super().__init__()
        if len(cell) == 6:
            self._cell = np.array(cell)
        else:
            self._cell = self.__unitToCell(cell)

    def __unitToCell(self, unit):
        """
        Private method to convert unit cell vectors to cell parameters.

        Parameters:
            unit (array-like, shape (3,3)): Unit cell vectors.

        Returns:
            Cell parameters (a, b, c, alpha, beta, gamma).

        Return type:
            numpy.ndarray (shape (6, ))
        """
        if spf.isSympyObject(unit):
            unit_pow = unit.dot(unit.T)
            a, b, c = sp.sqrt(unit_pow[0, 0]), sp.sqrt(unit_pow[1, 1]), sp.sqrt(unit_pow[2, 2])
            gamma = sp.re(sp.acos(unit_pow[0, 1] / (a * b))) / sp.pi * 180
            alpha = sp.re(sp.acos(unit_pow[1, 2] / (b * c))) / sp.pi * 180
            beta = sp.re(sp.acos(unit_pow[2, 0] / (c * a))) / sp.pi * 180
            return np.array([a, b, c, alpha, beta, gamma])  # a,b,c type is sympy.Add, alpha,beta,gamma type is sympy.Mul
        else:
            def angle(v1, v2): return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
            a, b, c = np.linalg.norm(unit, axis=1)
            gamma, alpha, beta = angle(unit[0], unit[1]), angle(unit[1], unit[2]), angle(unit[2], unit[0])
            return np.array([a, b, c, alpha, beta, gamma])

    def InverseLatticeVectors(self):
        """
        Calculates the reciprocal lattice vectors from the real lattice vectors.

        Returns:
            The reciprocal lattice vectors.

        Return type:
            numpy.ndarray (shape (3,3))

        Note:
            InverseLatticeVectors[0], InverseLatticeVectors[1], InverseLatticeVectors[2] are a*, b*, c* respectively.
            InverseLatticeVectors[index][0], InverseLatticeVectors[index][1], InverseLatticeVectors[index][2] are kx, ky, kz of the vector of the index.
        """

        lib = sp if spf.isSympyObject(self.unit) else np
        res = []
        a, b, c = self.unit
        res.append(np.cross(b, c) / np.dot(a, np.cross(b, c)) * 2 * lib.pi)
        res.append(np.cross(c, a) / np.dot(b, np.cross(c, a)) * 2 * lib.pi)
        res.append(np.cross(a, b) / np.dot(c, np.cross(a, b)) * 2 * lib.pi)
        return np.array(res)

    @property
    def unit(self):
        """
        Returns the unit cell vectors of the lattice.

        Returns:
            The unit cell vectors in Angstrom in real space.

        Return type:
            numpy.ndarray (shape (3,3))

        Note:
            unit[0], unit[1], unit[2] are a, b, c respectively.
            unit[index][0], unit[index][1], unit[index][2] are x, y, z of the vector of the index.
        """
        lib = sp if spf.isSympyObject(self._cell) else np
        unit = []
        a, b, c = self._cell[0:3]
        al, be, ga = self._cell[3:6] / 180 * lib.pi
        unit.append([a, 0, 0])
        unit.append([b * lib.cos(ga), b * lib.sin(ga), 0])
        unit.append([c * lib.cos(be), c * (lib.cos(al) - lib.cos(be) * lib.cos(ga)) / lib.sin(ga), c * lib.sqrt(1 + 2 * lib.cos(al) * lib.cos(be) * lib.cos(ga) - lib.cos(al) * lib.cos(al) - lib.cos(be) * lib.cos(be) - lib.cos(ga) * lib.cos(ga)) / lib.sin(ga)])
        return np.array(unit)

    @property
    def inv(self):
        """
        Returns the reciprocal lattice vectors of the lattice.

        Returns:
            The reciprocal lattice vectors in 1/Angstrom in reciprocal space.

        Return type:
            numpy.ndarray (shape (3,3))

        Note:
            inv[0], inv[1], inv[2] are a*, b*, c* respectively.
            inv[index][0], inv[index][1], inv[index][2] are kx, ky, kz of the vector of the index.
        """
        return self.InverseLatticeVectors()

    @property
    def a(self):
        """
        The lattice parameter a in Angstrom.

        Returns:
            The lattice parameter a in Angstrom.

        Return type:
            float or sympy expression
        """
        return self._cell[0]

    @property
    def b(self):
        """
        The lattice parameter b in Angstrom.

        Returns:
            The lattice parameter b in Angstrom.

        Return type:
            float or sympy expression
        """
        return self._cell[1]

    @property
    def c(self):
        """
        The lattice parameter c in Angstrom.

        Returns:
            The lattice parameter c in Angstrom.

        Return type:
            float or sympy expression
        """
        return self._cell[2]

    @property
    def alpha(self):
        """
        The lattice angle alpha in degree.

        Returns:
            The lattice angle alpha in degree.

        Return type:
            float or sympy expression
        """
        return self._cell[3]

    @property
    def beta(self):
        """
        The lattice angle beta in degree.

        Returns:
            The lattice angle beta in degree.

        Return type:
            float or sympy expression
        """

        return self._cell[4]

    @property
    def gamma(self):
        """
        The lattice angle gamma in degree.

        Returns:
            The lattice angle gamma in degree.

        Return type:
            float or sympy expression
        """

        return self._cell[5]

    @property
    def cell(self):
        """
        Returns the complete cell array consisting of lattice parameters and angles.

        Returns:
            The complete cell array consisting of lattice parameters and angles, i.e. numpy.array([a, b, c, alpha, beta, gamma]). Each element is a float or sympy expression.

        Return type:
            numpy.ndarray (shape (6, ))
        """
        return self._cell

    def latticeInfo(self):
        """
        Returns a string containing the lattice parameters and angles.

        Returns:
            a string containing the lattice parameters and angles.

        Return type:
            str

        Note:
            The format is "a = x, b = y, c = z, alpha = u, beta = v, gamma = w\\n" where x, y, z, u, v, and w are the actual values.
            If the values are sympy expressions, they are converted to strings using str(). If the values are numbers, they are formatted as "{:.5f}".
        """
        if spf.isSympyObject(self._cell):
            res = "a = {:}, b = {:}, c = {:}".format(self.a, self.b, self.c)
            res += ", alpha = {:}, beta = {:}, gamma = {:}\n".format(self.alpha, self.beta, self.gamma)
        else:
            res = "a = {:.5f}, b = {:.5f}, c = {:.5f}".format(self.a, self.b, self.c)
            res += ", alpha = {:.5f}, beta = {:.5f}, gamma = {:.5f}\n".format(self.alpha, self.beta, self.gamma)
        return res

    def Volume(self):
        """
        Returns the volume of the unit cell in A^3.

        Returns:
            The volume of the unit cell in A^3.

        Return type:
            float or sympy expression
        """
        a, b, c = self.unit
        return np.dot(a, np.cross(b, c))
