import numpy as np
import sympy as sp
import warnings
import unittest
import numpy as np

from lys_mat import Symmetry, Atom, Atoms, CartesianLattice


class TestSymmetry(unittest.TestCase):
    def test_cubic(self):

        at1 = Atom("Au", (0, 0, 0))
        at2 = Atom("Au", (0.5, 0.5, 0))
        at3 = Atom("Au", (0, 0.5, 0.5))
        at4 = Atom("Au", (0.5, 0, 0.5))
        atoms = Atoms([at1, at2, at3, at4])
        cell = [4.07825, 4.07825, 4.07825, 90, 90, 90]
        lattice = CartesianLattice(cell)
        sym = Symmetry(atoms, lattice)  # Space group: Fm-3m, Space group number: 225  -> extended Bravais lattice in seekpath: cF2

        self.assertEqual(sym.crystalSystem(), "cubic")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            np.testing.assert_array_equal(sym.standardPath(), ['GAMMA', 'X', 'U', 'K', 'GAMMA', 'L', 'W', 'X'])
            self.assert_dictionary_of_array_almost_equal(sym.symmetryPoints(), {'GAMMA': [0., 0., 0.], 'X': [0.5, 0., 0.5], 'L': [0.5, 0.5, 0.5], 'W': [0.5, 0.25, 0.75], 'W_2': [0.75, 0.25, 0.5], 'K': [0.375, 0.375, 0.75], 'U': [0.625, 0.25, 0.625]})
        self.assertEqual(len(sym.irreducibleAtoms()), 1)
        for v in ["Fm-3m", "225"]:
            self.assertIn(v, sym.symmetryInfo())
        symop = sym.getSymmetryOperations()
        self.assertEqual(len(symop), 2)
        self.assertEqual(np.shape(symop[0]), (4 * 4 * 4 * 3, 3, 3))
        self.assertEqual(np.shape(symop[1]), (4 * 4 * 4 * 3, 3))
        np.testing.assert_array_equal(symop[0][0], np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
        np.testing.assert_array_equal(symop[1][0], np.array([0., 0., 0.]))
        symop = sym.getSymmetryOperations(pointGroup=True)
        self.assertEqual(np.shape(symop), (4 * 4 * 3, 3, 3))
        np.testing.assert_array_equal(symop[0], np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))

    def test_hexagonal(self):
        at1 = Atom("Ga", (1 / 3, 2 / 3, 0))
        at2 = Atom("N", (1 / 3, 2 / 3, 0.377))
        symop = []
        symop.append((np.eye(3), np.zeros(3)))
        symop.append(([[0, -1, 0], [1, -1, 0], [0, 0, 1]], np.zeros(3)))
        symop.append(([[-1, 1, 0], [-1, 0, 0], [0, 0, 1]], np.zeros(3)))
        symop.append(([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], (0, 0, 0.5)))
        symop.append(([[1, -1, 0], [1, 0, 0], [0, 0, 1]], (0, 0, 0.5)))
        symop.append(([[0, 1, 0], [-1, 1, 0], [0, 0, 1]], (0, 0, 0.5)))
        symop.append(([[0, -1, 0], [-1, 0, 0], [0, 0, 1]], np.zeros(3)))
        symop.append(([[-1, 1, 0], [0, 1, 0], [0, 0, 1]], np.zeros(3)))
        symop.append(([[1, 0, 0], [1, -1, 0], [0, 0, 1]], np.zeros(3)))
        symop.append(([[0, 1, 0], [1, 0, 0], [0, 0, 1]], (0, 0, 0.5)))
        symop.append(([[1, -1, 0], [0, -1, 0], [0, 0, 1]], (0, 0, 0.5)))
        symop.append(([[-1, 0, 0], [-1, 1, 0], [0, 0, 1]], (0, 0, 0.5)))
        atoms = Atoms([at1, at2], symop)
        cell = [3.19, 3.19, 5.189, 90, 90, 120]
        lattice = CartesianLattice(cell)
        sym = Symmetry(atoms, lattice)  # Space group: P6(3)mc, Space group number: 186  -> extended Bravais lattice in seekpath: hP2

        self.assertEqual(sym.crystalSystem(), "hexagonal")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            np.testing.assert_array_equal(sym.standardPath(), ['GAMMA', 'M', 'K', 'GAMMA', 'A', 'L', 'H', 'A', 'L', 'M', 'H', 'K'])
            self.assert_dictionary_of_array_almost_equal(sym.symmetryPoints(), {'GAMMA': [0., 0., 0.], 'A': [0., 0., 0.5], 'K': [1 / 3, 1 / 3, 0], 'H': [1 / 3, 1 / 3, 0.5], 'H_2': [1 / 3, 1 / 3, -0.5], 'M': [0.5, 0., 0.], 'L': [0.5, 0., 0.5]})
        self.assertEqual(len(sym.irreducibleAtoms()), 2)
        for v in ["P6_3mc", "186"]:
            self.assertIn(v, sym.symmetryInfo())
        symop = sym.getSymmetryOperations()
        self.assertEqual(len(symop), 2)
        self.assertEqual(np.shape(symop[0]), (12, 3, 3))
        self.assertEqual(np.shape(symop[1]), (12, 3))
        symop = sym.getSymmetryOperations(pointGroup=True)
        self.assertEqual(np.shape(symop), (6, 3, 3))

    def test_tetragonal(self):

        at1 = Atom("O", (0.241, 0.0999, 0.176))
        at2 = Atom("Si", (0.2977, 0.2977, 0))
        symop = []
        symop.append((np.eye(3), np.zeros(3)))
        symop.append(([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], (0, 0, 0.5)))
        symop.append(([[0, -1, 0], [1, 0, 0], [0, 0, 1]], (0.5, 0.5, 0.25)))
        symop.append(([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], (0.5, 0.5, 0.75)))
        symop.append(([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], (0.5, 0.5, 0.25)))
        symop.append(([[1, 0, 0], [0, -1, 0], [0, 0, -1]], (0.5, 0.5, 0.75)))
        symop.append(([[0, 1, 0], [1, 0, 0], [0, 0, -1]], np.zeros(3)))
        symop.append(([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], (0, 0, 0.5)))
        atoms = Atoms([at1, at2], symop)
        cell = [4.9898, 4.9898, 6.992, 90, 90, 90]
        lattice = CartesianLattice(cell)    # Space grup: P4(1)2(1)2, Space group number: 92  -> extended Bravais lattice in seekpath: tP1
        sym = Symmetry(atoms, lattice)

        self.assertEqual(sym.crystalSystem(), "tetragonal")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            np.testing.assert_array_equal(sym.standardPath(), ['GAMMA', 'X', 'M', 'GAMMA', 'Z', 'R', 'A', 'Z', 'X', 'R', 'M', 'A'])
            self.assert_dictionary_of_array_almost_equal(sym.symmetryPoints(), {'GAMMA': [0, 0, 0], 'Z': [0, 0, 0.5], 'M': [0.5, 0.5, 0], 'A': [0.5, 0.5, 0.5], 'R': [0, 0.5, 0.5], 'X': [0, 0.5, 0]})
        self.assertEqual(len(sym.irreducibleAtoms()), 2)
        for v in ["P4_12_12", "92"]:
            self.assertIn(v, sym.symmetryInfo())
        symop = sym.getSymmetryOperations()
        self.assertEqual(len(symop), 2)
        self.assertEqual(np.shape(symop[0]), (8, 3, 3))
        self.assertEqual(np.shape(symop[1]), (8, 3))
        symop = sym.getSymmetryOperations(pointGroup=True)
        self.assertEqual(np.shape(symop), (2, 3, 3))

    def test_monoclinic(self):

        at1 = Atom("V", (0.2597, 0.018, 0.2915))
        at2 = Atom("O", (0.084, 0.265, 0.4))
        at3 = Atom("O", (0.606, 0.21, 0.403))
        symop = []
        symop.append((np.eye(3), np.zeros(3)))
        symop.append(([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], (0, 0.5, 0.5)))
        symop.append((-np.eye(3), np.zeros(3)))
        symop.append(([[1, 0, 0], [0, -1, 0], [0, 0, 1]], (0, 0.5, 0.5)))
        atoms = Atoms([at1, at2, at3], symop)
        cell = [5.3572, 4.5263, 5.3825, 90, 115.222, 90]
        lattice = CartesianLattice(cell)    # Space grup: P2(1)/c, Space group number: 14  -> extended Bravais lattice in seekpath: mP1
        sym = Symmetry(atoms, lattice)

        self.assertEqual(sym.crystalSystem(), "monoclinic")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            np.testing.assert_array_equal(sym.standardPath(), ['GAMMA', 'Z', 'D', 'B', 'GAMMA', 'A', 'E', 'Z', 'C_2', 'Y_2', 'GAMMA'])
            Y, N = 0.35182367709814505, 0.34937051990826495
            self.assert_dictionary_of_array_almost_equal(sym.symmetryPoints(), {'GAMMA': [0, 0, 0], 'Z': [0, 0.5, 0], 'B': [0, 0, 0.5], 'B_2': [0, 0, -0.5], 'Y': [0.5, 0, 0], 'Y_2': [-0.5, 0, 0], 'C': [0.5, 0.5, 0], 'C_2': [-0.5, 0.5, 0], 'D': [0, 0.5, 0.5], 'D_2': [0, 0.5, -0.5], 'A': [-0.5, 0, 0.5], 'E': [-0.5, 0.5, 0.5], 'H': [-Y, 0, 1 - N], 'H_2': [-1 + Y, 0, N], 'H_4': [-Y, 0, -N], 'M': [-Y, 0.5, 1 - N], 'M_2': [-1 + Y, 0.5, N], 'M_4': [-Y, 0.5, -N]})
        for v in ["P2_1/c", "14"]:
            self.assertIn(v, sym.symmetryInfo())
        symop = sym.getSymmetryOperations()
        self.assertEqual(len(sym.irreducibleAtoms()), 3)
        self.assertEqual(len(symop), 2)
        self.assertEqual(np.shape(symop[0]), (4, 3, 3))
        self.assertEqual(np.shape(symop[1]), (4, 3))
        symop = sym.getSymmetryOperations(pointGroup=True)
        self.assertEqual(np.shape(symop), (2, 3, 3))

    def test_supercell_of_cubic(self):

        # create 2x2x1 supercell
        at1 = Atom("Au", (0, 0, 0))
        at2 = Atom("Au", (0.25, 0.25, 0))
        at3 = Atom("Au", (0, 0.25, 0.5))
        at4 = Atom("Au", (0.25, 0, 0.5))
        symop = []
        symop.append((np.eye(3), np.zeros(3)))
        symop.append((np.eye(3), (0.5, 0, 0)))
        symop.append((np.eye(3), (0, 0.5, 0)))
        symop.append((np.eye(3), (0.5, 0.5, 0)))
        atoms = Atoms([at1, at2, at3, at4], symop)
        cell = [8.1565, 8.1565, 4.07825, 90, 90, 90]
        lattice = CartesianLattice(cell)
        sym = Symmetry(atoms, lattice)

        self.assertEqual(sym.crystalSystem(), "cubic")  # same as primitive cubic cell
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            np.testing.assert_array_equal(sym.standardPath(), ['GAMMA', 'X', 'U', 'K', 'GAMMA', 'L', 'W', 'X'])
            self.assert_dictionary_of_array_almost_equal(sym.symmetryPoints(), {'GAMMA': [0., 0., 0.], 'X': [0.5, 0., 0.5], 'L': [0.5, 0.5, 0.5], 'W': [0.5, 0.25, 0.75], 'W_2': [0.75, 0.25, 0.5], 'K': [0.375, 0.375, 0.75], 'U': [0.625, 0.25, 0.625]})
        self.assertEqual(len(sym.irreducibleAtoms()), 1)    # same as primitive cubic cell
        for v in ["Fm-3m", "225"]:  # same as primitive cubic cell
            self.assertIn(v, sym.symmetryInfo())
        symop = sym.getSymmetryOperations()
        self.assertEqual(len(symop), 2)
        self.assertEqual(np.shape(symop[0]), (4 * 4 * 4 * 4, 3, 3))  # differ from primitive cubic cell
        self.assertEqual(np.shape(symop[1]), (4 * 4 * 4 * 4, 3))
        symop = sym.getSymmetryOperations(pointGroup=True)
        self.assertEqual(np.shape(symop), (4 * 4, 3, 3))

    def test_orthorhombic_from_hexagonal(self):
        at1 = Atom("Ga", (1 / 3, 0, 0))
        at2 = Atom("N", (1 / 3, 0, 0.377))
        symop = []
        symop.append((np.eye(3), np.zeros(3)))
        symop.append((np.eye(3), (0.5, 0.5, 0)))
        symop.append((np.eye(3), (1 / 3, 0, 0.5)))
        symop.append((np.eye(3), (-1 / 6, 0.5, 0.5)))
        atoms = Atoms([at1, at2], symop)
        cell = [5.525, 3.19, 5.189, 90, 90, 90]
        lattice = CartesianLattice(cell)
        sym = Symmetry(atoms, lattice)

        self.assertEqual(sym.crystalSystem(), "orthorhombic")  # differ from primitive hexagonal cell
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            np.testing.assert_array_equal(sym.standardPath(), ['GAMMA', 'Y', 'C_0', 'SIGMA_0', 'GAMMA', 'Z', 'A_0', 'E_0', 'T', 'Y', 'GAMMA', 'S', 'R', 'Z', 'T'])
            X = 0.33334063594111507
            self.assert_dictionary_of_array_almost_equal(sym.symmetryPoints(), {'GAMMA': [0., 0., 0.], 'Y': [-0.5, 0.5, 0.], 'T': [-0.5, 0.5, 0.5], 'Z': [0., 0., 0.5], 'S': [0., 0.5, 0.], 'R': [0., 0.5, 0.5], 'SIGMA_0': [X, X, 0.], 'C_0': [-X, 1 - X, 0.], 'A_0': [X, X, 0.5], 'E_0': [-X, 1 - X, 0.5]})
        self.assertEqual(len(sym.irreducibleAtoms()), 2)    # same as primitive hexagonal cell
        for v in ["Cmc2_1", "36"]:  # differ from primitive hexagonal cell
            self.assertIn(v, sym.symmetryInfo())
        symop = sym.getSymmetryOperations()
        self.assertEqual(len(symop), 2)
        self.assertEqual(np.shape(symop[0]), (8, 3, 3))  # differ from primitive hexagonal cell
        self.assertEqual(np.shape(symop[1]), (8, 3))
        symop = sym.getSymmetryOperations(pointGroup=True)
        self.assertEqual(np.shape(symop), (2, 3, 3))

    def assert_dictionary_of_array_almost_equal(self, dict1, dict2):
        self.assertEqual(dict1.keys(), dict2.keys())
        for key in dict1:
            np.testing.assert_array_almost_equal(dict1[key], dict2[key])
