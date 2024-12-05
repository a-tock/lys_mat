import unittest
import numpy as np
import sympy as sp
from lys_mat import Atom
from lys_mat import sympyFuncs as spf


class TestAtom(unittest.TestCase):
    def test_getAtomicNumber(self):
        self.assertEqual(Atom.getAtomicNumber("H"), 1)
        self.assertEqual(Atom.getAtomicNumber("La"), 57)
        self.assertEqual(Atom.getAtomicNumber("Ra"), 88)
        self.assertEqual(Atom.getAtomicNumber("Pb"), 82)
        self.assertEqual(Atom.getAtomicNumber("Ar"), 18)

    def test_getAtomicName(self):
        self.assertEqual(Atom.getAtomicName(1), "H")
        self.assertEqual(Atom.getAtomicName(57), "La")
        self.assertEqual(Atom.getAtomicName(88), "Ra")
        self.assertEqual(Atom.getAtomicName(82), "Pb")
        self.assertEqual(Atom.getAtomicName(18), "Ar")

    def test_getAtomicMass(self):
        self.assertAlmostEqual(Atom.getAtomicMass("H"), 1.00794, delta=1e-4)
        self.assertAlmostEqual(Atom.getAtomicMass("La"), 138.9055, delta=1e-4)
        self.assertAlmostEqual(Atom.getAtomicMass("Ra"), 226.0, delta=1e-4)
        self.assertAlmostEqual(Atom.getAtomicMass("Pb"), 207.2, delta=1e-4)
        self.assertAlmostEqual(Atom.getAtomicMass("Ar"), 39.948, delta=1e-4)

    def test_AtomBasic(self):
        at = Atom("La", (0.1, 0.2, 0.3), occupancy=0.2)

        self.assertFalse(spf.isSympyObject(at))
        self.assertEqual(at.element, "La")
        self.assertEqual(at.Z, 57)
        np.testing.assert_array_equal(at.position, (0.1, 0.2, 0.3))
        self.assertEqual(at.occupancy, 0.2)

        at2 = Atom("La", (0.1, 0.2, 0.3))
        self.assertEqual(at2.occupancy, 1)

        # check duplicate
        at_dup = at.duplicate()
        self.assertEqual(at_dup.element, "La")
        self.assertEqual(at_dup.Z, 57)
        np.testing.assert_array_equal(at_dup.position, (0.1, 0.2, 0.3))
        self.assertEqual(at_dup.occupancy, 0.2)

        at2_dup = at2.duplicate()
        self.assertEqual(at2_dup.occupancy, 1)

        # check save and load
        d = at.saveAsDictionary()
        at_ld = Atom.loadFromDictionary(d)
        self.assertEqual(at_ld.element, "La")
        self.assertEqual(at_ld.Z, 57)
        np.testing.assert_array_equal(at_ld.position, (0.1, 0.2, 0.3))
        self.assertEqual(at_ld.occupancy, 0.2)

    def test_atomicDisplacementParams(self):
        at = Atom("H", U=1.1)
        np.testing.assert_array_equal(at.Uani, 1.1 * np.eye(3))

        at = Atom("H", U=[1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(at.Uani, [[1, 4, 6], [4, 2, 5], [6, 5, 3]])

        at = Atom("H", U=[[1, 4, 6], [4, 2, 5], [6, 5, 3]])
        np.testing.assert_array_equal(at.Uani, [[1, 4, 6], [4, 2, 5], [6, 5, 3]])

        # check duplicate
        at_dup = at.duplicate()
        np.testing.assert_array_equal(at_dup.Uani, [[1, 4, 6], [4, 2, 5], [6, 5, 3]])

        # check save and load
        d = at.saveAsDictionary()
        at_ld = Atom.loadFromDictionary(d)
        np.testing.assert_array_equal(at_ld.Uani, [[1, 4, 6], [4, 2, 5], [6, 5, 3]])

        at = Atom("H", U=1.1)
        d = at.saveAsDictionary()
        at_ld = Atom.loadFromDictionary(d)
        np.testing.assert_array_equal(at_ld.Uani, 1.1 * np.eye(3))

    def test_arbitraryParams(self):
        at = Atom("H", spin=1.1, st="name", lis=[1, 2, 3], tup=(3, 4, 5), arr=np.array([5, 6, 7]))
        self.assertEqual(at.spin, 1.1)
        self.assertEqual(at.st, "name")
        np.testing.assert_array_equal(at.lis, [1, 2, 3])
        np.testing.assert_array_equal(at.tup, [3, 4, 5])
        np.testing.assert_array_equal(at.arr, [5, 6, 7])

        at_dup = at.duplicate()
        self.assertEqual(at_dup.spin, 1.1)
        self.assertEqual(at_dup.st, "name")
        np.testing.assert_array_equal(at_dup.lis, [1, 2, 3])
        np.testing.assert_array_equal(at_dup.tup, [3, 4, 5])
        np.testing.assert_array_equal(at_dup.arr, [5, 6, 7])

        d = at.saveAsDictionary()
        at_ld = Atom.loadFromDictionary(d)
        self.assertEqual(at_ld.spin, 1.1)
        self.assertEqual(at_ld.st, "name")
        np.testing.assert_array_equal(at_ld.lis, [1, 2, 3])
        np.testing.assert_array_equal(at_ld.tup, [3, 4, 5])
        np.testing.assert_array_equal(at_ld.arr, [5, 6, 7])

    def test_sympy(self):
        x, y, z = sp.symbols("x y z")
        s = sp.symbols("s")
        oc = sp.symbols("oc")

        at = Atom("H", (x, y, z))
        self.assertTrue(spf.isSympyObject(at))
        self.assertEqual(at.free_symbols, set({x, y, z}))
        at2 = at.subs(x, 0.1)
        self.assertEqual(at2.position[0], 0.1)
        at2 = at.subs({x: 0.1, y: 0.2})
        self.assertEqual(at2.position[0], 0.1)
        self.assertEqual(at2.position[1], 0.2)

        at = Atom("H", spin=s, occupancy=oc)
        self.assertTrue(spf.isSympyObject(at))
        self.assertEqual(at.free_symbols, set({s, oc}))
        at2 = at.subs(s, 0.1)
        self.assertEqual(at2.spin, 0.1)
        at3 = at2.duplicate()
        self.assertEqual(at3.spin, 0.1)
        self.assertEqual(at3.occupancy, oc)

        at = Atom("H", (x, y, z), spin=s, occupancy=oc)
        d = at.saveAsDictionary()
        at2 = Atom.loadFromDictionary(d)
        self.assertEqual(at.free_symbols, set({x, y, z, s, oc}))
        self.assertEqual(at2.position[1], y)
        self.assertEqual(at2.spin, s)
