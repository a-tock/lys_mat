import unittest
import numpy as np
import sympy as sp
import pickle
from lys_mat import Atom, Atoms, CartesianLattice, Symmetry, CrystalStructure
from lys_mat import sympyFuncs as spf


class TestCrystalStructure(unittest.TestCase):
    def test_float_basic(self):
        at1 = Atom("Au", (0, 0, 0))
        at2 = Atom("Au", (0.5, 0.5, 0))
        at3 = Atom("Au", (0, 0.5, 0.5))
        at4 = Atom("Au", (0.5, 0, 0.5))
        cell = [4.0773, 4.0773, 4.0773, 90, 90, 90]
        crys = CrystalStructure(cell, [at1, at2, at3, at4])

        # atoms property
        for atom in crys.atoms:
            self.assertEqual(atom.element, "Au")
            self.assertTrue(any([np.allclose(pos, atom.Position) for pos in [(0, 0, 0), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5)]]))

        # atoms setter
        crys.atoms = [at1, at2]
        for atom in crys.atoms:
            self.assertEqual(atom.element, "Au")
            self.assertTrue(any([np.allclose(pos, atom.Position) for pos in [(0, 0, 0), (0.5, 0.5, 0)]]))

        crys.atoms = [at1, at2, at3, at4]

        # Atoms class method
        np.testing.assert_array_almost_equal(crys.getAtomicPositions(), [(0, 0, 0), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5)])

        # Lattice class property and method
        np.testing.assert_array_almost_equal(crys.cell, [4.0773, 4.0773, 4.0773, 90, 90, 90])
        self.assertAlmostEqual(crys.volume(), 4.0773 ** 3)

        # Symmetry class method
        self.assertEqual(crys.crystalSystem(), "cubic")

        # density funciton
        self.assertAlmostEqual(crys.density(), 19.30, delta=0.01)

        # pickle
        pickled_crys = pickle.loads(pickle.dumps(crys))
        for atom in pickled_crys.atoms:
            self.assertEqual(atom.element, "Au")
            self.assertTrue(any([np.allclose(pos, atom.Position) for pos in [(0, 0, 0), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5)]]))

    def test_sympy_basic(self):
        a, b, c, r = sp.symbols("a b c r")
        at1 = Atom("Au", (0, 0, 0))
        at2 = Atom("Au", (r, r, 0))
        at3 = Atom("Au", (0, r, r))
        at4 = Atom("Au", (r, 0, r))
        cell = [a, b, c, 90, 90, 90]
        crys = CrystalStructure(cell, [at1, at2, at3, at4])

        # pickle
        pickled_crys = pickle.loads(pickle.dumps(crys))
        np.testing.assert_array_equal(pickled_crys.cell, cell)
        for atom in pickled_crys.atoms:
            self.assertEqual(atom.element, "Au")
            self.assertTrue(any([all(pos == atom.position) for pos in [(0, 0, 0), (r, r, 0), (0, r, r), (r, 0, r)]]))

        # sympy method
        self.assertEqual(crys.isSympyObject(), True)
        self.assertEqual(crys.free_symbols, {a, b, c, r})
        np.testing.assert_array_equal(crys.symbolNames(), ["a", "b", "c", "r"])
        crys_subs1 = crys.subs(a, 2)
        np.testing.assert_array_almost_equal(crys_subs1.cell, [2, b, c, 90, 90, 90])
        crys_subs2 = crys.subs({a: 3.0, b: 4.0, c: 5.0})
        np.testing.assert_array_almost_equal(crys_subs2.cell, [3, 4, 5, 90, 90, 90])
        crys_subs2 = crys.subs([(a, 6), (b, 7), (c, 8)])
        np.testing.assert_array_almost_equal(crys_subs2.cell, [6, 7, 8, 90, 90, 90])

        # sympy method to not sympy object
        crys_subs = crys.subs({a: 3.0, b: 4.0, c: 5.0, r: 0.5})
        self.assertEqual(crys_subs.isSympyObject(), False)
        self.assertEqual(crys_subs.free_symbols, set())
        np.testing.assert_array_equal(crys_subs.symbolNames(), [])
