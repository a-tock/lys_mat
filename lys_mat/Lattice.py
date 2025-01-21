import numpy as np
import sympy as sp

from . import sympyFuncs as spf


class Lattice(object):
    """
    Lattice class is a basic lattice class used for crystal structures.

    Args:
        cell (array-like): An array of length 6 representing the cell parameters (a, b, c, alpha, beta, gamma) in Angstrom and degrees.

    Example::

        from lys_mat import Lattice
        cell = [4.0, 5.0, 10.0, 90.0, 90.5, 60.0]
        lattice = Lattice(cell)
        print(lattice.latticeInfo())    #a = 4.00000, b = 5.00000, c = 10.00000, alpha = 90.00000, beta = 90.50000, gamma = 60.00000

    """

    def __init__(self, cell):

        super().__init__()
        self._cell = np.array(cell)

    @property
    def a(self):
        """
        The lattice parameter a in Angstrom.

        Returns:
            float or sympy expression: The lattice parameter a in Angstrom.
        """
        return self._cell[0]

    @property
    def b(self):
        """
        The lattice parameter b in Angstrom.

        Returns:
            float or sympy expression: The lattice parameter b in Angstrom.
        """
        return self._cell[1]

    @property
    def c(self):
        """
        The lattice parameter c in Angstrom.

        Returns:
            float or sympy expression: The lattice parameter c in Angstrom.
        """
        return self._cell[2]

    @property
    def alpha(self):
        """
        The lattice angle alpha in degree.

        Returns:
            float or sympy expression: The lattice angle alpha in degree.
        """
        return self._cell[3]

    @property
    def beta(self):
        """
        The lattice angle beta in degree.

        Returns:
            float or sympy expression: The lattice angle beta in degree.
        """

        return self._cell[4]

    @property
    def gamma(self):
        """
        The lattice angle gamma in degree.

        Returns:
            float or sympy expression: The lattice angle gamma in degree.
        """

        return self._cell[5]

    @property
    def cell(self):
        """
        Returns the complete cell array consisting of lattice parameters and angles.

        Returns:
            numpy.ndarray (shape (6, )): The complete cell array consisting of lattice parameters and angles, i.e. numpy.array([a, b, c, alpha, beta, gamma]).
                                        Each element is a float or sympy expression.

        """
        return self._cell

    def latticeInfo(self):
        """
        Returns a string containing the lattice parameters and angles.

        Returns:
            str: a string containing the lattice parameters and angles.

        Note:
            The format is "a = x, b = y, c = z, alpha = u, beta = v, gamma = w\\n" where x, y, z, u, v, and w are the actual values.
            If the values contain sympy expressions, they are converted to strings using str(). If the values are numbers, they are formatted as "{:.5f}".
        """

        return ", ".join([key + " = " + ("{:}" if spf.isSympyObject(value) else "{:.5f}").format(value) for key, value in zip(["a", "b", "c", "alpha", "beta", "gamma"], self._cell)]) + "\n"

    def Volume(self):
        """
        Returns the volume of the unit cell in A^3.

        Returns:
            float or sympy expression: The volume of the unit cell in A^3.
        """
        angle = self.cell[3:].copy()
        lib = sp if spf.isSympyObject(self.cell) else np
        for i in range(3):
            angle[i] = angle[i] * lib.pi / 180.0

        return self.cell[0] * self.cell[1] * self.cell[2] * lib.sqrt(1 + 2 * lib.cos(angle[0]) * lib.cos(angle[1]) * lib.cos(angle[2]) - lib.cos(angle[0]) ** 2 - lib.cos(angle[1]) ** 2 - lib.cos(angle[2]) ** 2)


class CartesianLattice(Lattice):
    """
    CartesianLattice class is a basic lattice class with Cartesian coordinates used for crystal structures.

    Parameters:
        cell (array-like): An array of length 6 representing the cell parameters (a, b, c, alpha, beta, gamma) in Angstrom and degrees.
                           Alternatively, a 3x3 array representing unit cell vectors.
        basis (numpy.ndarray, shape (3,3), optional): The basis of the lattice vectors in Cartesian coordinates like [e1, e2, e3]. Defaults to None.
                Each of e1, e2, e3 has to be [x, y, z] and parallel to the a, b, c vector, and the norm of them is 1.

    Note:
        If basis is None, the basis will be calculated automatically from the lattice parameters as below.
        e1 vector: pallarell to the x-axis
        e2 vector: parallel to the xy plane
        e3 vector: determined by a vector, b vector, alpha, beta, gamma

    Example::

        import numpy as np
        from lys_mat import CartesianLattice

        cell = [4.0, 5.0, 10.0, 90.0, 90.0, 60.0]
        lattice = CartesianLattice(cell)
        print(lattice.unit)     #[[4. 0. 0.], [2.5 4.33012702 0.], [0. 0. 10.]]

        lattice = CartesianLattice(cell, basis = [[0, 1, 0], [0, 1/2, np.sqrt(3)/2], [1, 0, 0]])
        print(lattice.unit)     #[[0. 4. 0.], [0. 2.5 4.33012702], [10. 0. 0.]]

        cell = [[0.0, 4.0, 0.0], [0.0, 2.5, 5*np.sqrt(3)/2], [10.0, 0.0, 0.0]]
        lattice = CartesianLattice(cell)
        print(lattice.latticeInfo())    #a = 4.00000, b = 5.00000, c = 10.00000, alpha = 90.00000, beta = 90.00000, gamma = 60.00000
        print(lattice.unit)     #[[4. 0. 0.], [2.5 4.33012702 0.], [0. 0. 10.]]

    """

    def __init__(self, cell, basis=None):

        if len(cell) == 6:
            super().__init__(cell)
        else:
            super().__init__(self.__unitToCell(cell))
        self.basis = basis

    def __unitToCell(self, unit):
        """
        Private method to convert unit cell vectors to cell parameters.

        Parameters:
            unit (array-like, shape (3,3)): Unit cell vectors.

        Returns:
            numpy.ndarray (shape (6, )): Cell parameters (a, b, c, alpha, beta, gamma).
        """
        if spf.isSympyObject(unit):
            unit = np.array(unit)
            unit_pow = unit.dot(unit.T)
            a, b, c = sp.sqrt(unit_pow[0, 0]), sp.sqrt(unit_pow[1, 1]), sp.sqrt(unit_pow[2, 2])
            gamma = sp.deg(sp.re(sp.acos(unit_pow[0, 1] / (a * b))))
            alpha = sp.deg(sp.re(sp.acos(unit_pow[1, 2] / (b * c))))
            beta = sp.deg(sp.re(sp.acos(unit_pow[2, 0] / (c * a))))
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
            numpy.ndarray (shape (3,3)): The reciprocal lattice vectors.

        Note:
            InverseLatticeVectors[0], InverseLatticeVectors[1], InverseLatticeVectors[2] are a*, b*, c* respectively.
            InverseLatticeVectors[index][0], InverseLatticeVectors[index][1], InverseLatticeVectors[index][2] are kx, ky, kz of the vector of the index.
        """

        lib = sp if spf.isSympyObject(self.unit) else np
        return np.array(2 * lib.pi * (sp.Matrix(self.unit.T).inv()))

    @property
    def unit(self):
        """
        Returns the unit cell vectors of the lattice.

        Returns:
            numpy.ndarray (shape (3,3)): The unit cell vectors in Angstrom in real space.

        Note:
            unit[0], unit[1], unit[2] are a, b, c respectively.
            unit[index][0], unit[index][1], unit[index][2] are x, y, z of the vector of the index.
        """
        return np.array([e * norm for e, norm in zip(self._basis, self.cell[0:3])])

    @property
    def inv(self):
        """
        Returns the reciprocal lattice vectors of the lattice.

        Returns:
            numpy.ndarray (shape (3,3)): The reciprocal lattice vectors in 1/Angstrom in reciprocal space.

        Note:
            inv[0], inv[1], inv[2] are a*, b*, c* respectively.
            inv[index][0], inv[index][1], inv[index][2] are kx, ky, kz of the vector of the index.
        """
        return self.InverseLatticeVectors()

    @property
    def basis(self):
        """
        Returns the basis of the lattice vectors to Cartesian coordinates.

        Returns:
            numpy.ndarray (shape (3,3)): The basis vectors in Angstrom in real space.
        """

        return self._basis

    @basis.setter
    def basis(self, basis):
        """
        Sets the basis of the lattice vectors to Cartesian coordinates.

        Parameters:
            basis (numpy.ndarray, shape (3,3)): The basis of the lattice vectors like [e1, e2, e3].
                Each of e1, e2, e3 has to be [x, y, z] and parallel to the a, b, c vector, and the norm of each of them has to be 1.

        Note:
            If basis is None, the basis will be calculated automatically from the lattice parameters as below.
            e1 vector: pallarell to the x-axis
            e2 vector: parallel to the xy plane
            e3 vector: determined by a vector, b vector, alpha, beta, gamma

        """
        if basis is None:
            lib = sp if spf.isSympyObject(self.cell) else np
            [al, be, ga] = [sp.rad(deg) if lib is sp else np.deg2rad(deg) for deg in [self.alpha, self.beta, self.gamma]]

            basis = []
            basis.append([1, 0, 0])
            basis.append([lib.cos(ga), lib.sin(ga), 0])
            basis.append([lib.cos(be), (lib.cos(al) - lib.cos(be) * lib.cos(ga)) / lib.sin(ga), lib.sqrt(1 + 2 * lib.cos(al) * lib.cos(be) * lib.cos(ga) - lib.cos(al) * lib.cos(al) - lib.cos(be) * lib.cos(be) - lib.cos(ga) * lib.cos(ga)) / lib.sin(ga)])
        else:
            basis = np.array(basis)

            # check if basis is consistent with cell
            if np.shape(basis) != (3, 3):
                raise ValueError("basis must be a 3x3 array.")
            else:
                if not spf.isSympyObject(self.cell):
                    if np.allclose(np.array([np.linalg.norm(v) for v in basis]), np.ones(3)) is False:
                        raise ValueError("Norm of basis must be 1.")
                    al, be, ga = [np.rad2deg(np.arccos(cos)) for cos in [np.dot(basis[1], basis[2]), np.dot(basis[2], basis[0]), np.dot(basis[0], basis[1])]]
                    if np.allclose([al, be, ga], [self.alpha, self.beta, self.gamma]) is False:
                        raise ValueError("Angles of basis must be alpha, beta, gamma.")

        self._basis = np.array(basis)
