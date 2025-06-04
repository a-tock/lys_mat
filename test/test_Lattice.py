import unittest
import numpy as np
import sympy as sp

from lys_mat.crystal import sympyFuncs as spf
from lys_mat.crystal import Lattice, CartesianLattice


class TestLattice(unittest.TestCase):
    def test_Lattice(self):
        # float and int
        cell = [5.12, 6.88, 7.532159, 70.11, 80.88888, 90]
        lat = Lattice(cell)
        self.assertEqual(lat.a, 5.12)
        self.assertEqual(lat.b, 6.88)
        self.assertEqual(lat.c, 7.532159)
        self.assertEqual(lat.alpha, 70.11)
        self.assertEqual(lat.beta, 80.88888)
        self.assertEqual(lat.gamma, 90)
        np.testing.assert_array_equal(lat.cell, cell)
        self.assertEqual(lat.latticeInfo(), "a = 5.12000, b = 6.88000, c = 7.53216, alpha = 70.11000, beta = 80.88888, gamma = 90.00000\n")
        self.assertAlmostEqual(lat.volume, cell[0] * cell[1] * cell[2] * np.sqrt(1 + 2 * np.cos(cell[3] * np.pi / 180) * np.cos(cell[4] * np.pi / 180) * np.cos(cell[5] * np.pi / 180) - np.cos(cell[3] * np.pi / 180) ** 2 - np.cos(cell[4] * np.pi / 180) ** 2 - np.cos(cell[5] * np.pi / 180) ** 2))

        # sympy.Symbol
        sa, sb, sc = sp.symbols("sa sb sc")
        salpha, sbeta, sgamma = sp.symbols("salpha sbeta sgamma")
        cell = [sa, sb, sc, salpha, sbeta, sgamma]
        lat = Lattice(cell)
        self.assertEqual(lat.a, sa)
        self.assertEqual(lat.b, sb)
        self.assertEqual(lat.c, sc)
        self.assertEqual(lat.alpha, salpha)
        self.assertEqual(lat.beta, sbeta)
        self.assertEqual(lat.gamma, sgamma)
        np.testing.assert_array_equal(lat.cell, cell)
        self.assertEqual(lat.latticeInfo(), "a = sa, b = sb, c = sc, alpha = salpha, beta = sbeta, gamma = sgamma\n")
        self.assertEqual(lat.volume, sa * sb * sc * sp.sqrt(1 + 2 * sp.cos(salpha * sp.pi / 180.0) * sp.cos(sbeta * sp.pi / 180.0) * sp.cos(sgamma * sp.pi / 180.0) - sp.cos(salpha * sp.pi / 180.0) ** 2 - sp.cos(sbeta * sp.pi / 180.0) ** 2 - sp.cos(sgamma * sp.pi / 180.0) ** 2))

        # sympy.Symbol and int
        cell = [sa, sb, sc, 90, 90, 60]
        lat = Lattice(cell)
        np.testing.assert_array_equal(lat.cell, cell)
        self.assertEqual(lat.latticeInfo(), "a = sa, b = sb, c = sc, alpha = 90.00000, beta = 90.00000, gamma = 60.00000\n")
        self.assertEqual(lat.volume, sa * sb * sc * sp.sqrt(1 - sp.cos(60 * sp.pi / 180.0) ** 2))


class TestCartesianLattice(unittest.TestCase):
    def test_CartesianLattice_with_lattice_parameter(self):

        # float without basis
        [a, b, c, al, be, ga] = [5.12, 6.88, 7.532159, 90, 90, 60]
        lat = CartesianLattice([a, b, c, al, be, ga])
        va = [a, 0, 0]
        vb = [b / 2, b * np.sqrt(3) / 2, 0]
        vc = [0, 0, c]
        np.testing.assert_array_almost_equal(lat.unit, np.array([va, vb, vc]))
        np.testing.assert_array_almost_equal(lat.basis, np.array([va, vb, vc]) / (np.multiply(np.ones((3, 3)), np.array([a, b, c])).T))
        np.testing.assert_array_almost_equal(lat.InverseLatticeVectors(), 2 * np.pi * np.array([[1 / a, -1 / (a * np.sqrt(3)), 0], [0, 2 / (b * np.sqrt(3)), 0], [0, 0, 1 / c]]))
        np.testing.assert_array_equal(lat.InverseLatticeVectors(), lat.inv)

        [a, b, c, al, be, ga] = [5.12, 6.88, 7.532159, 80.5, 90.2, 70.8]
        lat = CartesianLattice([a, b, c, al, be, ga])
        va = [a, 0, 0]
        vb = [b * np.cos(np.deg2rad(ga)), b * np.sin(np.deg2rad(ga)), 0]
        vc = [c * np.cos(np.deg2rad(be)), c * (np.cos(np.deg2rad(al)) - np.cos(np.deg2rad(be)) * np.cos(np.deg2rad(ga))) / np.sin(np.deg2rad(ga)), c * np.sqrt(1 + 2 * np.cos(np.deg2rad(al)) * np.cos(np.deg2rad(be)) * np.cos(np.deg2rad(ga)) - np.cos(np.deg2rad(al)) ** 2 - np.cos(np.deg2rad(be)) ** 2 - np.cos(np.deg2rad(ga)) ** 2) / np.sin(np.deg2rad(ga))]
        np.testing.assert_array_almost_equal(lat.unit, np.array([va, vb, vc]))
        np.testing.assert_array_almost_equal(lat.basis, (np.array([va, vb, vc]).T / np.array([a, b, c])).T)

        inv = []
        inv.append(np.cross(vb, vc) / np.dot(va, np.cross(vb, vc)) * 2 * np.pi)
        inv.append(np.cross(vc, va) / np.dot(vb, np.cross(vc, va)) * 2 * np.pi)
        inv.append(np.cross(va, vb) / np.dot(vc, np.cross(va, vb)) * 2 * np.pi)
        np.testing.assert_array_almost_equal(lat.InverseLatticeVectors(), inv)
        np.testing.assert_array_equal(lat.InverseLatticeVectors(), lat.inv)
        np.testing.assert_array_almost_equal(lat.inv, [[1.2271846, -0.4273509, 0.0807516], [0, 0.9670442, -0.1728840], [0, 0, 0.8474120]])  # check whether the result matches that calculated using another software (http://calistry.org/viz/direct-and-reciprocal-lattice-visualizer).

        # sympy simbols without basis
        sa, sb, sc, sal, sbe, sga = sp.symbols("sa sb sc sal sbe sga")
        [a, b, c, al, be, ga] = [sa, sb, sc, sal, sbe, sga]
        lat = CartesianLattice([a, b, c, al, be, ga])
        va = [a, 0, 0]
        vb = [b * sp.cos(sp.rad(ga)), b * sp.sin(sp.rad(ga)), 0]
        vc = [c * sp.cos(sp.rad(be)), c * (sp.cos(sp.rad(al)) - sp.cos(sp.rad(be)) * sp.cos(sp.rad(ga))) / sp.sin(sp.rad(ga)), c * sp.sqrt(1 + 2 * sp.cos(sp.rad(al)) * sp.cos(sp.rad(be)) * sp.cos(sp.rad(ga)) - sp.cos(sp.rad(al)) ** 2 - sp.cos(sp.rad(be)) ** 2 - sp.cos(sp.rad(ga)) ** 2) / sp.sin(sp.rad(ga))]
        np.testing.assert_array_almost_equal(lat.unit, np.array([va, vb, vc]))
        np.testing.assert_array_almost_equal(lat.basis, (np.array([va, vb, vc]).T / np.array([a, b, c])).T)
        inv = []
        inv.append(np.cross(vb, vc) / np.dot(va, np.cross(vb, vc)) * 2 * sp.pi)
        inv.append(np.cross(vc, va) / np.dot(vb, np.cross(vc, va)) * 2 * sp.pi)
        inv.append(np.cross(va, vb) / np.dot(vc, np.cross(va, vb)) * 2 * sp.pi)
        np.testing.assert_array_equal([sp.simplify(v) for v in [v for v in lat.InverseLatticeVectors()]], [sp.simplify(v) for v in [v for v in inv]])
        np.testing.assert_array_equal(lat.InverseLatticeVectors(), lat.inv)

    def test_CartesianLattice_with_vector_without_basis(self):
        # defined by vector
        [a, b, c, al, be, ga] = [5.12, 6.88, 7.532159, 90, 90, 60]
        va = [a, 0, 0]
        vb = [b / 2, b * np.sqrt(3) / 2, 0]
        vc = [0, 0, c]
        lat = CartesianLattice([va, vb, vc])
        np.testing.assert_array_almost_equal(lat.cell, np.array([a, b, c, al, be, ga]))
        np.testing.assert_array_almost_equal(lat.unit, np.array([va, vb, vc]))

        [a, b, c, al, be, ga] = [5.12, 6.88, 7.532159, 80.5, 90.2, 70.8]
        va = [a, 0, 0]
        vb = [b * np.cos(np.deg2rad(ga)), b * np.sin(np.deg2rad(ga)), 0]
        vc = [c * np.cos(np.deg2rad(be)), c * (np.cos(np.deg2rad(al)) - np.cos(np.deg2rad(be)) * np.cos(np.deg2rad(ga))) / np.sin(np.deg2rad(ga)), c * np.sqrt(1 + 2 * np.cos(np.deg2rad(al)) * np.cos(np.deg2rad(be)) * np.cos(np.deg2rad(ga)) - np.cos(np.deg2rad(al)) ** 2 - np.cos(np.deg2rad(be)) ** 2 - np.cos(np.deg2rad(ga)) ** 2) / np.sin(np.deg2rad(ga))]
        lat = CartesianLattice([va, vb, vc])
        np.testing.assert_array_almost_equal(lat.cell, np.array([a, b, c, al, be, ga]))
        np.testing.assert_array_almost_equal(lat.unit, np.array([va, vb, vc]))

        sa, sb, sc, sal, sbe, sga = sp.symbols("sa sb sc sal sbe sga")
        [a, b, c, al, be, ga] = [sa, sb, sc, sal, sbe, sga]
        va = [a, 0, 0]
        vb = [b * sp.cos(sp.rad(ga)), b * sp.sin(sp.rad(ga)), 0]
        vc = [c * sp.cos(sp.rad(be)), c * (sp.cos(sp.rad(al)) - sp.cos(sp.rad(be)) * sp.cos(sp.rad(ga))) / sp.sin(sp.rad(ga)), c * sp.sqrt(1 + 2 * sp.cos(sp.rad(al)) * sp.cos(sp.rad(be)) * sp.cos(sp.rad(ga)) - sp.cos(sp.rad(al)) ** 2 - sp.cos(sp.rad(be)) ** 2 - sp.cos(sp.rad(ga)) ** 2) / sp.sin(sp.rad(ga))]
        lat = CartesianLattice([va, vb, vc])
        aa, bb, cc, aal, bbe, gga = [5.12, 6.88, 7.532159, 80.5, 90.2, 70.8]
        subdic = {sa: aa, sb: bb, sc: cc, sal: aal, sbe: bbe, sga: gga}
        np.testing.assert_array_almost_equal(spf.subs(lat.cell, subdic), np.array([aa, bb, cc, aal, bbe, gga]))
        np.testing.assert_array_almost_equal(spf.subs(lat.unit, subdic), spf.subs(np.array([va, vb, vc]), subdic))

        # defined by rotated vector
        [theta, phi, psi] = [58.1, 20.5, 210.8]
        rtheta = np.array([[np.cos(np.deg2rad(theta)), 0, -np.sin(np.deg2rad(theta))], [0, 1, 0], [np.sin(np.deg2rad(theta)), 0, np.cos(np.deg2rad(theta))]])
        rphi = np.array([[1, 0, 0], [0, np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))], [0, -np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi))]])
        rpsi = np.array([[np.cos(np.deg2rad(psi)), np.sin(np.deg2rad(psi)), 0], [-np.sin(np.deg2rad(psi)), np.cos(np.deg2rad(psi)), 0], [0, 0, 1]])
        mrot = np.dot(np.dot(rphi, rtheta), rpsi)

        [a, b, c, al, be, ga] = [5.12, 6.88, 7.532159, 80.5, 90.2, 70.8]
        va = [a, 0, 0]
        vb = [b * np.cos(np.deg2rad(ga)), b * np.sin(np.deg2rad(ga)), 0]
        vc = [c * np.cos(np.deg2rad(be)), c * (np.cos(np.deg2rad(al)) - np.cos(np.deg2rad(be)) * np.cos(np.deg2rad(ga))) / np.sin(np.deg2rad(ga)), c * np.sqrt(1 + 2 * np.cos(np.deg2rad(al)) * np.cos(np.deg2rad(be)) * np.cos(np.deg2rad(ga)) - np.cos(np.deg2rad(al)) ** 2 - np.cos(np.deg2rad(be)) ** 2 - np.cos(np.deg2rad(ga)) ** 2) / np.sin(np.deg2rad(ga))]
        [vva, vvb, vvc] = np.dot(mrot, np.array([va, vb, vc]).T).T
        lat = CartesianLattice([vva, vvb, vvc])
        np.testing.assert_array_almost_equal(lat.cell, np.array([a, b, c, al, be, ga]))
        np.testing.assert_array_almost_equal(lat.unit, np.array([va, vb, vc]))

        # defined by rotated vector with sympy
        stheta, sphi, spsi = sp.symbols("stheta sphi spsi")
        [theta, phi, psi] = [stheta, sphi, spsi]
        rtheta = np.array([[sp.cos(sp.rad(theta)), 0, -sp.sin(sp.rad(theta))], [0, 1, 0], [sp.sin(sp.rad(theta)), 0, sp.cos(sp.rad(theta))]])
        rphi = np.array([[1, 0, 0], [0, sp.cos(sp.rad(phi)), sp.sin(sp.rad(phi))], [0, -sp.sin(sp.rad(phi)), sp.cos(sp.rad(phi))]])
        rpsi = np.array([[sp.cos(sp.rad(psi)), sp.sin(sp.rad(psi)), 0], [-sp.sin(sp.rad(psi)), sp.cos(sp.rad(psi)), 0], [0, 0, 1]])
        mrot = np.dot(np.dot(rphi, rtheta), rpsi)

        sa, sb, sc, sal, sbe, sga = sp.symbols("sa sb sc sal sbe sga")
        [a, b, c, al, be, ga] = [sa, sb, sc, sal, sbe, sga]
        va = [a, 0, 0]
        vb = [b * sp.cos(sp.rad(ga)), b * sp.sin(sp.rad(ga)), 0]
        vc = [c * sp.cos(sp.rad(be)), c * (sp.cos(sp.rad(al)) - sp.cos(sp.rad(be)) * sp.cos(sp.rad(ga))) / sp.sin(sp.rad(ga)), c * sp.sqrt(1 + 2 * sp.cos(sp.rad(al)) * sp.cos(sp.rad(be)) * sp.cos(sp.rad(ga)) - sp.cos(sp.rad(al)) ** 2 - sp.cos(sp.rad(be)) ** 2 - sp.cos(sp.rad(ga)) ** 2) / sp.sin(sp.rad(ga))]
        [vva, vvb, vvc] = np.dot(mrot, np.array([va, vb, vc]).T).T
        lat = CartesianLattice([vva, vvb, vvc])
        aa, bb, cc, aal, bbe, gga, ttheta, pphi, ppsi = [5.12, 6.88, 7.532159, 80.5, 90.2, 70.8, 58.1, 20.5, 210.8]
        subdic = {sa: aa, sb: bb, sc: cc, sal: aal, sbe: bbe, sga: gga, stheta: ttheta, sphi: pphi, spsi: ppsi}
        np.testing.assert_array_almost_equal(spf.subs(lat.cell, subdic), np.array([aa, bb, cc, aal, bbe, gga]))
        np.testing.assert_array_almost_equal(spf.subs(lat.unit, subdic), spf.subs(np.array([va, vb, vc]), subdic))

    def test_CartesianLattice_with_vector_with_basis(self):
        # float
        [theta, phi, psi] = [58.1, 20.5, 210.8]
        rtheta = np.array([[np.cos(np.deg2rad(theta)), 0, -np.sin(np.deg2rad(theta))], [0, 1, 0], [np.sin(np.deg2rad(theta)), 0, np.cos(np.deg2rad(theta))]])
        rphi = np.array([[1, 0, 0], [0, np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))], [0, -np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi))]])
        rpsi = np.array([[np.cos(np.deg2rad(psi)), np.sin(np.deg2rad(psi)), 0], [-np.sin(np.deg2rad(psi)), np.cos(np.deg2rad(psi)), 0], [0, 0, 1]])
        mrot = np.dot(np.dot(rphi, rtheta), rpsi)

        [a, b, c, al, be, ga] = [5.12, 6.88, 7.532159, 80.5, 90.2, 70.8]
        va = [a, 0, 0]
        vb = [b * np.cos(np.deg2rad(ga)), b * np.sin(np.deg2rad(ga)), 0]
        vc = [c * np.cos(np.deg2rad(be)), c * (np.cos(np.deg2rad(al)) - np.cos(np.deg2rad(be)) * np.cos(np.deg2rad(ga))) / np.sin(np.deg2rad(ga)), c * np.sqrt(1 + 2 * np.cos(np.deg2rad(al)) * np.cos(np.deg2rad(be)) * np.cos(np.deg2rad(ga)) - np.cos(np.deg2rad(al)) ** 2 - np.cos(np.deg2rad(be)) ** 2 - np.cos(np.deg2rad(ga)) ** 2) / np.sin(np.deg2rad(ga))]
        [vva, vvb, vvc] = np.dot(mrot, np.array([va, vb, vc]).T).T
        lat = CartesianLattice([vva, vvb, vvc], basis=(np.array([vva, vvb, vvc]).T / np.array([a, b, c])).T)
        np.testing.assert_array_almost_equal(lat.cell, np.array([a, b, c, al, be, ga]))
        np.testing.assert_array_almost_equal(lat.basis, (np.array([vva, vvb, vvc]).T / np.array([a, b, c])).T)
        np.testing.assert_array_almost_equal(lat.unit, np.array([vva, vvb, vvc]))
        inv = []
        inv.append(np.cross(vvb, vvc) / np.dot(vva, np.cross(vvb, vvc)) * 2 * np.pi)
        inv.append(np.cross(vvc, vva) / np.dot(vvb, np.cross(vvc, vva)) * 2 * np.pi)
        inv.append(np.cross(vva, vvb) / np.dot(vvc, np.cross(vva, vvb)) * 2 * np.pi)
        np.testing.assert_array_almost_equal(lat.InverseLatticeVectors(), inv)
        np.testing.assert_array_equal(lat.InverseLatticeVectors(), lat.inv)

        with self.assertRaises(ValueError):
            lat.basis = [vva, vvb, vvc]

        with self.assertRaises(ValueError):
            lat.basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # set new bases
        [theta, phi, psi] = [100.8, -59.2, -95.3]
        rtheta = np.array([[np.cos(np.deg2rad(theta)), 0, -np.sin(np.deg2rad(theta))], [0, 1, 0], [np.sin(np.deg2rad(theta)), 0, np.cos(np.deg2rad(theta))]])
        rphi = np.array([[1, 0, 0], [0, np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))], [0, -np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi))]])
        rpsi = np.array([[np.cos(np.deg2rad(psi)), np.sin(np.deg2rad(psi)), 0], [-np.sin(np.deg2rad(psi)), np.cos(np.deg2rad(psi)), 0], [0, 0, 1]])
        mrot = np.dot(np.dot(rphi, rtheta), rpsi)

        [vva, vvb, vvc] = np.dot(mrot, np.array([vva, vvb, vvc]).T).T
        lat.basis = (np.array([vva, vvb, vvc]).T / np.array([a, b, c])).T
        np.testing.assert_array_almost_equal(lat.basis, (np.array([vva, vvb, vvc]).T / np.array([a, b, c])).T)
        np.testing.assert_array_almost_equal(lat.unit, np.array([vva, vvb, vvc]))

        # sympy
        stheta, sphi, spsi = sp.symbols("stheta sphi spsi")
        [theta, phi, psi] = [stheta, sphi, spsi]
        rtheta = np.array([[sp.cos(sp.rad(theta)), 0, -sp.sin(sp.rad(theta))], [0, 1, 0], [sp.sin(sp.rad(theta)), 0, sp.cos(sp.rad(theta))]])
        rphi = np.array([[1, 0, 0], [0, sp.cos(sp.rad(phi)), sp.sin(sp.rad(phi))], [0, -sp.sin(sp.rad(phi)), sp.cos(sp.rad(phi))]])
        rpsi = np.array([[sp.cos(sp.rad(psi)), sp.sin(sp.rad(psi)), 0], [-sp.sin(sp.rad(psi)), sp.cos(sp.rad(psi)), 0], [0, 0, 1]])
        mrot = np.dot(np.dot(rphi, rtheta), rpsi)

        sa, sb, sc, sal, sbe, sga = sp.symbols("sa sb sc sal sbe sga")
        [a, b, c, al, be, ga] = [sa, sb, sc, sal, sbe, sga]
        va = [a, 0, 0]
        vb = [b * sp.cos(sp.rad(ga)), b * sp.sin(sp.rad(ga)), 0]
        vc = [c * sp.cos(sp.rad(be)), c * (sp.cos(sp.rad(al)) - sp.cos(sp.rad(be)) * sp.cos(sp.rad(ga))) / sp.sin(sp.rad(ga)), c * sp.sqrt(1 + 2 * sp.cos(sp.rad(al)) * sp.cos(sp.rad(be)) * sp.cos(sp.rad(ga)) - sp.cos(sp.rad(al)) ** 2 - sp.cos(sp.rad(be)) ** 2 - sp.cos(sp.rad(ga)) ** 2) / sp.sin(sp.rad(ga))]
        [vva, vvb, vvc] = np.dot(mrot, np.array([va, vb, vc]).T).T
        lat = CartesianLattice([vva, vvb, vvc], basis=(np.array([vva, vvb, vvc]).T / np.array([a, b, c])).T)
        aa, bb, cc, aal, bbe, gga, ttheta, pphi, ppsi = [5.12, 6.88, 7.532159, 80.5, 90.2, 70.8, 58.1, 20.5, 210.8]
        subdic = {sa: aa, sb: bb, sc: cc, sal: aal, sbe: bbe, sga: gga, stheta: ttheta, sphi: pphi, spsi: ppsi}
        np.testing.assert_array_almost_equal(spf.subs(lat.cell, subdic), np.array([aa, bb, cc, aal, bbe, gga]))
        np.testing.assert_array_almost_equal(spf.subs(lat.unit, subdic), spf.subs(np.array([vva, vvb, vvc]), subdic))

        # set new bases with sympy
        srho, ssigma, stau = sp.symbols("srho ssigma stau")
        [rho, sigma, tau] = [srho, ssigma, stau]
        rrho = np.array([[sp.cos(sp.rad(rho)), 0, -sp.sin(sp.rad(rho))], [0, 1, 0], [sp.sin(sp.rad(rho)), 0, sp.cos(sp.rad(rho))]])
        rsigma = np.array([[1, 0, 0], [0, sp.cos(sp.rad(sigma)), sp.sin(sp.rad(sigma))], [0, -sp.sin(sp.rad(sigma)), sp.cos(sp.rad(sigma))]])
        rtau = np.array([[sp.cos(sp.rad(tau)), sp.sin(sp.rad(tau)), 0], [-sp.sin(sp.rad(tau)), sp.cos(sp.rad(tau)), 0], [0, 0, 1]])
        mrot = np.dot(np.dot(rrho, rsigma), rtau)

        [vva, vvb, vvc] = np.dot(mrot, np.array([vva, vvb, vvc]).T).T
        lat.basis = (np.array([vva, vvb, vvc]).T / np.array([a, b, c])).T
        aa, bb, cc, aal, bbe, gga, ttheta, pphi, ppsi, rhoo, sigmaa, ttau = [5.12, 6.88, 7.532159, 80.5, 90.2, 70.8, 58.1, 20.5, 210.8, 100.8, -59.2, -95.3]
        subdic = {sa: aa, sb: bb, sc: cc, sal: aal, sbe: bbe, sga: gga, stheta: ttheta, sphi: pphi, spsi: ppsi, srho: rhoo, ssigma: sigmaa, stau: ttau}
        np.testing.assert_array_almost_equal(spf.subs(lat.basis, subdic), spf.subs((np.array([vva, vvb, vvc]).T / np.array([a, b, c])).T, subdic))
        np.testing.assert_array_almost_equal(spf.subs(lat.unit, subdic), spf.subs(np.array([vva, vvb, vvc]), subdic))
