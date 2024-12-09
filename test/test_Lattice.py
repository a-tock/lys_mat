import numpy as np
import sympy as sp
import unittest

import lys_mat.Lattice as Lattice


class TestLattice(unittest.TestCase):
    def test_LatticeParameters(self):
        cell = [5.12, 6.88, 7.532159, 70.11, 80.88888, 90]
        lat = Lattice.Lattice(cell)
        self.assertEqual(lat.a, 5.12)
        self.assertEqual(lat.b, 6.88)
        self.assertEqual(lat.c, 7.532159)
        self.assertEqual(lat.alpha, 70.11)
        self.assertEqual(lat.beta, 80.88888)
        self.assertEqual(lat.gamma, 90)
        np.testing.assert_array_equal(lat.cell, cell)
        self.assertEqual(lat.latticeInfo(), "a = 5.12000, b = 6.88000, c = 7.53216, alpha = 70.11000, beta = 80.88888, gamma = 90.00000\n")

        sa, sb, sc = sp.symbols("sa sb sc")
        salpha, sbeta, sgamma = sp.symbols("salpha sbeta sgamma")
        cell = [sa, sb, sc, salpha, sbeta, sgamma]
        lat = Lattice.Lattice(cell)
        self.assertEqual(lat.a, sa)
        self.assertEqual(lat.b, sb)
        self.assertEqual(lat.c, sc)
        self.assertEqual(lat.alpha, salpha)
        self.assertEqual(lat.beta, sbeta)
        self.assertEqual(lat.gamma, sgamma)
        np.testing.assert_array_equal(lat.cell, cell)
        self.assertEqual(lat.latticeInfo(), "a = sa, b = sb, c = sc, alpha = salpha, beta = sbeta, gamma = sgamma\n")
