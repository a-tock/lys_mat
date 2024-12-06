import numpy as np
import sympy as sp
import random
import unittest

import lys_mat.sympyFuncs as spf


class TestSympyFuncs(unittest.TestCase):

    def test_isSympyObject(self):
        x = sp.Symbol('x')
        self.assertTrue(spf.isSympyObject(x))
        self.assertTrue(spf.isSympyObject([1, x**2]))
        self.assertTrue(spf.isSympyObject([[[[x]]]]))
        self.assertTrue(spf.isSympyObject([[[1, x], [2], [3, 3]], [[1, 1, 1, 1], [2, 2, 2]]]))
        self.assertTrue(spf.isSympyObject({"key1": {"key2": x, "key3": 1}}))
        self.assertTrue(spf.isSympyObject((2, x)))
        self.assertTrue(spf.isSympyObject({1, 2, x}))
        self.assertTrue(spf.isSympyObject(np.array([[1, 2], [3, x]])))
        self.assertFalse(spf.isSympyObject(1))
        self.assertFalse(spf.isSympyObject([]))
        self.assertFalse(spf.isSympyObject("x"))

    def test_free_symbols(self):
        x = sp.Symbol('x')
        y = sp.Symbol('y')

        self.assertEqual(spf.free_symbols(x), {x})
        self.assertEqual(spf.free_symbols([x, y]), {x, y})
        self.assertEqual(spf.free_symbols([[[2 * x, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [y**x, 3]]]), {x, y})
        self.assertEqual(spf.free_symbols({"key1": {"key2": x, "key3": 1}}), {x})
        self.assertEqual(spf.free_symbols((2, x)), {x})
        self.assertEqual(spf.free_symbols({1, x, y}), {x, y})
        self.assertEqual(spf.free_symbols(np.array([[1, x], [3, y]])), {x, y})
        self.assertEqual(spf.free_symbols("x"), set())
        self.assertEqual(spf.free_symbols([]), set())
        self.assertEqual(spf.free_symbols([1, 2, 3]), set())

    def test_subs(self):
        x = sp.Symbol('x')
        y = sp.Symbol('y')
        z = sp.Symbol('z')
        self.assertAlmostEqual(spf.subs(3 * x, x, 0.3), 0.9)
        np.testing.assert_array_equal(spf.subs([x, y], {x: 0.3, y: 0.5}), [0.3, 0.5])
        np.testing.assert_array_equal(spf.subs([2 * x, x, y], [(x, z), (y, 0.8)]), [2 * z, z, 0.8])
        np.testing.assert_array_equal(spf.subs([[[2 * x], [1]], [[y + 1], [2]]], [(x, 2), (y, 0.8)]), [[[4], [1]], [[1.8], [2]]])
        np.testing.assert_array_equal(spf.subs({"x": {"a": x, "b": x * 2}, "y": {"a": y, "b": y * 3}}, {x: 0.3, y: 0.5}), {"x": {"a": 0.3, "b": 0.6}, "y": {"a": 0.5, "b": 1.5}})
        np.testing.assert_array_equal(spf.subs((x, y), {x: 0.3, y: 0.5}), (0.3, 0.5))
        np.testing.assert_array_equal(spf.subs({1, x, y}, {x: 0.3, y: 0.5}), {1, 0.3, 0.5})
        np.testing.assert_array_equal(spf.subs(np.array([[1, x], [3, y]]), {x: 0.3, y: 0.5}), np.array([[1, 0.3], [3, 0.5]]))
        self.assertEqual(spf.subs([], x, 0), [])
        self.assertEqual(spf.subs((3), x, 1), (3))
        self.assertEqual(spf.subs("x", x, 1), "x")
        self.assertEqual(spf.subs(1, x, 0), 1)

    def test_einsum_for_number(self):
        self._check_einsum([random.randrange(-100, 100) for i in range(100)])

    def test_einsum_for_sympy(self):
        x, y, z = sp.symbols("x y z")
        self._check_einsum([0, 1, 3, -5, 111, -3333, x, y, z, x + y, y - z, z * x, x / y, y ** z, z % x, x // y, sp.pi, sp.I, sp.sqrt(z**2), sp.exp(x), sp.sin(y), sp.log(z)])

    def _check_einsum(self, arr):

        # views
        a = np.array(random.choices(arr, k=6)).reshape(2, 3)
        np.testing.assert_array_equal(spf.einsum("ij", a), a)
        np.testing.assert_array_equal(spf.einsum("ji", a), a.T)

#        b = np.array(random.choices(arr, k=9)).reshape(3, 3)
#        np.testing.assert_array_equal(spf.einsum("ii -> i", b), np.diag(b))

#        c = np.array(random.choices(arr, k=27)).reshape(3, 3, 3)
#        np.testing.assert_array_equal(spf.einsum("jii -> ji", c), [[x[i, i] for i in range(3)] for x in c])
#        np.testing.assert_array_equal(spf.einsum("iij -> ji", c), [[x[i, i] for i in range(3)] for x in c.transpose(2, 0, 1)])
#        np.testing.assert_array_equal(spf.einsum("jii -> ij", c), [c[:, i, i] for i in range(3)])
#        np.testing.assert_array_equal(spf.einsum("iij -> ij", c), [c.transpose(2, 0, 1)[:, i, i] for i in range(3)])
#        np.testing.assert_array_equal(spf.einsum("iii -> i", c), [c[i, i, i] for i in range(3)])

        d = np.array(random.choices(arr, k=24)).reshape(2, 3, 4)
        np.testing.assert_array_equal(spf.einsum("ijk -> jik", d), d.swapaxes(0, 1))

        # sums
        for n in range(1, 17):
            a = np.array(random.choices(arr, k=n))
            np.testing.assert_array_equal(spf.einsum("i ->", a), np.sum(a, axis=-1))

            a = np.array(random.choices(arr, k=2 * n)).reshape(2, n)
            np.testing.assert_array_equal(spf.einsum("ij -> j", a), np.sum(a, axis=0))

            a = np.array(random.choices(arr, k=2 * 3 * n)).reshape(2, 3, n)
            np.testing.assert_array_equal(spf.einsum("ijk -> ij", a), np.sum(a, axis=-1))
            np.testing.assert_array_equal(spf.einsum("ijk -> jk", a), np.sum(a, axis=0))

#            a = np.array(random.choices(arr, k=n * n)).reshape(n, n)
#            np.testing.assert_array_equal(spf.einsum("ii", a), np.trace(a))

            self.assertEqual(spf.einsum(",", 3, n), 3 * n)
            a = np.array(random.choices(arr, k=3 * n)).reshape(3, n)
            b = np.array(random.choices(arr, k=2 * 3 * n)).reshape(2, 3, n)
            np.testing.assert_array_equal(spf.einsum("ij, kij -> kij", a, b), np.multiply(a, b))

            a = np.array(random.choices(arr, k=2 * 3 * n)).reshape(2, 3, n)
            b = np.array(random.choices(arr, k=n))
            np.testing.assert_array_equal(spf.einsum("ijk, k", a, b), np.inner(a, b))

            a = np.array(random.choices(arr, k=n * 3 * 2)).reshape(n, 3, 2)
            b = np.array(random.choices(arr, k=n))
            np.testing.assert_array_equal(spf.einsum("ijk, i", a, b), np.inner(a.T, b.T).T)

            a = np.array(random.choices(arr, k=3))
            b = np.array(random.choices(arr, k=n))
            np.testing.assert_array_equal(spf.einsum("i, j", a, b), np.outer(a, b))

            a = np.array(random.choices(arr, k=4 * n)).reshape(4, n)
            b = np.array(random.choices(arr, k=n))
            np.testing.assert_array_equal(spf.einsum("ij, j", a, b), np.dot(a, b))
            np.testing.assert_array_equal(spf.einsum("ji, j", a.T, b.T), np.dot(b.T, a.T))

            a = np.array(random.choices(arr, k=4 * n)).reshape(4, n)
            b = np.array(random.choices(arr, k=n * 6)).reshape(n, 6)
            np.testing.assert_array_equal(spf.einsum("ij, jk", a, b), np.dot(a, b))

            a = np.array(random.choices(arr, k=n))
            np.testing.assert_array_equal(spf.einsum("i,i -> i", a, a), np.multiply(a, a))
            np.testing.assert_array_equal(spf.einsum("i,->i", a, 2), 2 * a)
            np.testing.assert_array_equal(spf.einsum(",i->i", 2, a), 2 * a)
            self.assertEqual(spf.einsum("i,i", a, a), np.dot(a, a))
            self.assertEqual(spf.einsum("i,->", a, 2), 2 * np.sum(a))
            self.assertEqual(spf.einsum(",i->", 2, a), 2 * np.sum(a))
            np.testing.assert_array_equal(spf.einsum("i,i -> i", a[1:], a[:-1]), np.multiply(a[1:], a[:-1]))
            np.testing.assert_array_equal(spf.einsum("i,->i", a[1:], 2), 2 * a[1:])
            np.testing.assert_array_equal(spf.einsum(",i->i", 2, a[1:]), 2 * a[1:])
            self.assertEqual(spf.einsum("i,i", a[1:], a[:-1]), np.dot(a[1:], a[:-1]))
            self.assertEqual(spf.einsum("i,->", a[1:], 2), 2 * np.sum(a[1:]))
            self.assertEqual(spf.einsum(",i->", 2, a[1:]), 2 * np.sum(a[1:]))

        a = np.array(random.choices(arr, k=12)).reshape(3, 4)
        b = np.array(random.choices(arr, k=20)).reshape(4, 5)
        c = np.array(random.choices(arr, k=30)).reshape(5, 6)
        np.testing.assert_array_equal([[sp.expand(x) for x in y] for y in spf.einsum("ij, jk, kl", a, b, c)], [[sp.expand(x) for x in y] for y in a.dot(b).dot(c)])

        a = np.array(random.choices(arr, k=60)).reshape(3, 4, 5)
        b = np.array(random.choices(arr, k=24)).reshape(4, 3, 2)
        np.testing.assert_array_equal(spf.einsum("ijk, jil -> kl", a, b), np.tensordot(a, b, axes=([1, 0], [0, 1])))

        a = np.array(random.choices(arr, k=9))
        self.assertEqual(spf.einsum(",i->", 3, a), 3 * np.sum(a))
        self.assertEqual(spf.einsum("i,->", a, 3), 3 * np.sum(a))

        np.testing.assert_array_equal(spf.einsum('ij,ij->j', np.ones((10, 2)), np.ones((1, 2))), [10.] * 2)
