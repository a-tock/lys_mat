import numpy as np
import sympy as sp
import unittest

import lys_mat.sympyFuncs as spf

"""
class TestSubsFunction(unittest.TestCase):
    def test_subs_single_expr_no_args(self):
        x = sp.Symbol('x')
        expr = 2*x
        result = spf.subs(expr)
        self.assertEqual(result, 2*x)

    def test_subs_array_expr_no_args(self):
        x = sp.Symbol('x')
        arr = np.array([x, x**2])
        result = spf.subs(arr)
        expected = np.array([x, x**2])
        np.testing.assert_array_equal(result, expected)

    def test_subs_single_expr_with_args(self):
        x = sp.Symbol('x')
        y = sp.Symbol('y')
        expr = x + y
        result = spf.subs(expr, y, 2)
        self.assertEqual(result, x + 2)

    def test_subs_array_expr_with_args(self):
        x = sp.Symbol('x')
        y = sp.Symbol('y')
        arr = np.array([x + y, x**2 + y])
        result = spf.subs(arr, y, 2)
        expected = np.array([x + 2, x**2 + 2])
        np.testing.assert_array_equal(result, expected)

    def test_subs_single_expr_with_kwargs(self):
        y = sp.Symbol('y')
        expr = y**2
        result = spf.subs(expr, y, z=3)
        self.assertEqual(result, y**2)

    def test_subs_array_expr_with_kwargs(self):
        x = sp.Symbol('x')
        y = sp.Symbol('y')
        arr = np.array([x**2, y**2])
        result = spf.subs(arr, x=x, y=3)
        expected = np.array([x**2, 9])
        np.testing.assert_array_equal(result, expected)
"""

"""

class TestIsSympyObject(unittest.TestCase):

    def test_empty_array(self):
        self.assertFalse(spf.isSympyObject(np.array([])))

    def test_numpy_array_no_sympy(self):
        x = np.array([1, 2, 3])
        self.assertFalse(spf.isSympyObject(x))

    def test_numpy_array_sympy_object(self):
        x = sp.symbols('x')
        self.assertTrue(spf.isSympyObject(np.array([x])))

    def test_numpy_array_sympy_expression(self):
        x = sp.symbols('x')
        expr = sp.sin(x) + sp.cos(x)
        self.assertTrue(spf.isSympyObject(np.array([expr])))

    def test_numpy_array_mixed(self):
        x = sp.symbols('x')
        expr = sp.sin(x) + sp.cos(x)
        self.assertTrue(spf.isSympyObject(np.array([1, x, expr])))

"""

class TestFreeSymbolsFunction(unittest.TestCase):

    def test_free_symbols_empty_array(self):
        # Test case for an empty array
        x = np.array([])
        expected_result = set()
        self.assertEqual(spf.free_symbols(x), expected_result)

    def test_free_symbols_array_with_no_symbols(self):
        # Test case for an array with no symbols
        x = np.array([1, 2, 3])
        expected_result = set()
        self.assertEqual(spf.free_symbols(x), expected_result)

    def test_free_symbols_array_with_symbols(self):
        # Test case for an array with symbols
        x = np.array([sp.Symbol('x'), sp.Symbol('y'), sp.Symbol('z')])
        expected_result = set(x)
        self.assertEqual(spf.free_symbols(x), expected_result)

    def test_free_symbols_array_with_mixed_elements(self):
        # Test case for an array with mixed elements (symbols and non-symbols)
        x = np.array([sp.Symbol('x'), sp.Symbol('y'), 'z', 1, sp.Symbol('a')])
        expected_result = set([sp.Symbol('x'), sp.Symbol('y'), sp.Symbol('a')])
        self.assertEqual(spf.free_symbols(x), expected_result)

    def test_free_symbols_matrix(self):
        # Test case for an matrix
        x = np.array([[sp.Symbol('x'), sp.Symbol('y')],[sp.Symbol('z'), sp.Symbol('a')]])
        expected_result = set([sp.Symbol('x'), sp.Symbol('y'), sp.Symbol('z'), sp.Symbol('a')])
        self.assertEqual(spf.free_symbols(x), expected_result)
