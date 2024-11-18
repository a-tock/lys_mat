import numpy as np
import sympy as sp
import unittest

import lys_mat.sympyFuncs as spf


class TestSympyFuncs(unittest.TestCase):

    def test_isSympyObject(self):
        x = sp.Symbol('x')
        self.assertTrue(spf.isSympyObject(x))
        self.assertTrue(spf.isSympyObject([1, x**2]))
        self.assertTrue(spf.isSympyObject([[[[x]]]]))
        self.assertTrue(spf.isSympyObject([[[1,x],[2,2],[3,3]],[[1,1],[2,2],[3,3]]]))   #check a tensor
        self.assertFalse(spf.isSympyObject(1))
        self.assertFalse(spf.isSympyObject([]))
        self.assertFalse(spf.isSympyObject("x"))


    def test_free_symbols(self):
        x = sp.Symbol('x')
        y = sp.Symbol('y')

        self.assertEqual(spf.free_symbols(x), {x})
        self.assertEqual(spf.free_symbols([x, y]), {x, y})
        self.assertEqual(spf.free_symbols([[[2*x,1],[2,2],[3,3]],[[1,1],[2,2],[y**x,3]]]), {x, y})
        self.assertEqual(spf.free_symbols("x"), set())
        self.assertEqual(spf.free_symbols([]), set())
        self.assertEqual(spf.free_symbols([1,2,3]), set())


    def test_subs(self):
        x = sp.Symbol('x')
        y = sp.Symbol('y')
        z = sp.Symbol('z')
        self.assertAlmostEqual(spf.subs(3*x, x, 0.3), 0.9)
        np.testing.assert_array_equal(spf.subs([x, y], {x: 0.3, y: 0.5}), [0.3, 0.5])
        np.testing.assert_array_equal(spf.subs([2*x, x, y], [(x, z), (y, 0.8)]), [2*z, z, 0.8])
        np.testing.assert_array_equal(spf.subs([[[2*x],[1]],[[y+1],[2]]], [(x, 2), (y, 0.8)]), [[[4],[1]],[[1.8],[2]]])
        self.assertEqual(spf.subs([], x, 0), [])