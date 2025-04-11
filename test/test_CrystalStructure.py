import unittest
import os
import numpy as np
import sympy as sp
import pickle
from lys_mat import Atom, Atoms, CartesianLattice, Symmetry, CrystalStructure, CrystalStructureIO
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

        # getAtomicPositions method
        np.testing.assert_array_almost_equal(crys.getAtomicPositions(), [[0, 0, 0], [2.03865, 2.03865, 0], [0, 2.03865, 2.03865], [2.03865, 0, 2.03865]])
        np.testing.assert_array_almost_equal(crys.getAtomicPositions(external=False), [[0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])

        # Atoms class method
        np.testing.assert_array_equal(crys.getElements(), ["Au"])

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

        # sympy method for not sympy object
        crys_subs = crys.subs({a: 3.0, b: 4.0, c: 5.0, r: 0.5})
        self.assertEqual(crys_subs.isSympyObject(), False)
        self.assertEqual(crys_subs.free_symbols, set())
        np.testing.assert_array_equal(crys_subs.symbolNames(), [])


class TestStrain(unittest.TestCase):
    def test_strain(self):
        cell = [4.0, 5.0, 6.0, 90.0, 90.0, 90.0]
        atoms = []
        crys = CrystalStructure(cell, atoms)
        eps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        strained_crys = crys.createStrainedCrystal(eps)
        expected_unit = np.array([
            [4.04, 0.16, 0.24],
            [0.2, 5.1, 0.25],
            [0.36, 0.3, 6.18]
        ])
        expected_crys = CrystalStructure(expected_unit, atoms)
        np.testing.assert_array_almost_equal(strained_crys.unit, expected_crys.unit)
        np.testing.assert_array_almost_equal(strained_crys.calculateStrain(crys), eps)

        cell = [4.0, 5.0, 6.0, 70, 80, 90.5]
        atoms = []
        crys = CrystalStructure(cell, atoms)
        eps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        strained_crys = crys.createStrainedCrystal(eps)
        R = [[1 + eps[0], eps[3], eps[5]], [eps[3], 1 + eps[1], eps[4]], [eps[5], eps[4], 1 + eps[2]]]
        expected_unit = np.array(R).dot(crys.unit.T).T
        expected_crys = CrystalStructure(expected_unit, atoms)
        np.testing.assert_array_almost_equal(strained_crys.unit, expected_crys.unit)
        np.testing.assert_array_almost_equal(strained_crys.calculateStrain(crys), eps)


class TestCrystalStructureIO(unittest.TestCase):
    path = "test/DataFiles/"
    iopath = "test/IOtestFiles/"

    def test_io(self):
        crys = CrystalStructureIO.loadFrom(self.path + "VO2_monoclinic")
        self.assertEqual(crys.getElements(), ["O", "V"])
        self.assertEqual(len(crys.getAtoms()), 12)

        os.makedirs(self.iopath, exist_ok=True)

        # save and load without extension
        CrystalStructureIO.saveAs(crys, self.iopath + "test_io_VO2_monoclinic")
        crys2 = CrystalStructureIO.loadFrom(self.iopath + "test_io_VO2_monoclinic")
        self.assertEqual(crys2.getElements(), ["O", "V"])
        self.assertEqual(len(crys2.getAtoms()), 12)

        # save as a cif file and load it with extension
        CrystalStructureIO.saveAs(crys, self.iopath + "test_io2_VO2_monoclinic.cif")
        crys3 = CrystalStructureIO.loadFrom(self.iopath + "test_io2_VO2_monoclinic.cif")
        self.assertEqual(crys3.getElements(), ["O", "V"])
        self.assertEqual(len(crys3.getAtoms()), 12)

        # save as a pcs file and load
        # Note: The extension ".pcs" is not a standard extension for crystal structure files.
        CrystalStructureIO.saveAs(crys, self.iopath + "test_io3_VO2_monoclinic", ext=".pcs")
        crys4 = CrystalStructureIO.loadFrom(self.iopath + "test_io3_VO2_monoclinic", ext=".pcs")
        self.assertEqual(crys4.getElements(), ["O", "V"])
        self.assertEqual(len(crys4.getAtoms()), 12)

        # save using a method of CrystalStructure and load it
        crys.saveAs(self.iopath + "test_io4_VO2_monoclinic")
        crys5 = CrystalStructureIO.loadFrom(self.iopath + "test_io4_VO2_monoclinic")
        self.assertEqual(crys5.getElements(), ["O", "V"])
        self.assertEqual(len(crys5.getAtoms()), 12)
