import unittest
import numpy as np
import sympy as sp
from lys_mat import Atom
from lys_mat import Atoms
from lys_mat import sympyFuncs as spf


class TestAtoms(unittest.TestCase):
    def test_Atoms_without_sym(self):
        at1 = Atom("Si", (0, 0, 0), occupancy=1)
        at2 = Atom("Au", (0.5, 0.5, 0), occupancy=0.5)
        at3 = Atom("C", (0.5, 0.5, 0.5), occupancy=0.25)
        at4 = Atom("Au", (0, 0.5, 0.5), occupancy=0.4)
        atlist = [at1, at2, at3, at4]
        atoms = Atoms(atlist)
        atlist = sorted(atlist, key=lambda at: at.element)

        np.testing.assert_array_equal(atoms.getElements(), ["Au", "C", "Si"])

        for at01, at02 in zip(atoms.getAtoms(), atlist):
            np.testing.assert_array_equal(at01.element, at02.element)
            np.testing.assert_array_equal(at01.Position, at02.Position)
            np.testing.assert_array_equal(at01.occupancy, at02.occupancy)

        for pos, at in zip(atoms.getAtomicPositions(external=False), atlist):
            np.testing.assert_array_equal(pos, at.Position)

        atlist = [at1, at2]
        atoms.setAtoms(atlist)
        atlist = sorted(atlist, key=lambda at: at.element)

        for at01, at02 in zip(atoms.getAtoms(), atlist):
            np.testing.assert_array_equal(at01.element, at02.element)
            np.testing.assert_array_equal(at01.Position, at02.Position)
            np.testing.assert_array_equal(at01.occupancy, at02.occupancy)

        self.assertEqual(type(atoms.atomInfo()), str)

    def test_Atoms_with_sym(self):
        at1 = Atom("Au", (0, 0, 0), occupancy=1)
        at2 = Atom("Au", (0.25, 0.25, 0.25), occupancy=1)
        atlist = [at1, at2]

        sym = []
        sym.append((np.eye(3), np.zeros(3)))
        sym.append((np.eye(3), (0.5, 0.5, 0.5)))
        atoms = Atoms(atlist, sym=sym)
        self.assertEqual(len(atoms.getAtoms()), 4)

        sym = []
        sym.append((np.eye(3), np.zeros(3)))
        sym.append((np.eye(3), (0.25, 0.25, 0.25)))
        sym.append((np.eye(3), (0.5, 0.5, 0.5)))
        sym.append((np.eye(3), (-0.25, -0.25, -0.25)))
        atoms.setAtoms(atlist, sym=sym)
        self.assertEqual(len(atoms.getAtoms()), 4)

        sym = []
        sym.append((np.eye(3), np.zeros(3)))
        sym.append((-np.eye(3), np.zeros(3)))
        atoms.setAtoms(atlist, sym=sym)
        self.assertEqual(len(atoms.getAtoms()), 3)

        sym = []
        sym.append((np.eye(3), np.zeros(3)))
        sym.append((-np.eye(3), np.zeros(3)))
        sym.append((np.eye(3), (0.5, 0.5, 0.5)))
        sym.append((-np.eye(3), (0.25, 0.25, 0.25)))
        atoms.setAtoms(atlist, sym=sym)
        self.assertEqual(len(atoms.getAtoms()), 4)
