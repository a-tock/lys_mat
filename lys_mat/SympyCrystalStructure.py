import numpy as np
import sympy as sp
import spglib
import copy
from . import sympyFuncs as spf
from .Atom import Atom
from .Crystal import CrystalStructure


class SympyCS(object):
    def __init__(self, crys):
        self._crys = crys

    def isSympyObject(self):
        return spf.isSympyObject(self._crys.atoms) or spf.isSympyObject(self._crys.cell)

    @property
    def free_symbols(self):
        res = [spf.free_symbols(self._crys.atoms), spf.free_symbols(self._crys.cell)]
        return set().union(*res)

    def symbolNames(self):
        cellName = ["a", "b", "c", "alpha", "beta", "gamma"]
        res = sorted([s.name for s in self.free_symbols if s.name not in cellName])
        cell = [c for c in cellName if c in [s.name for s in self.free_symbols]]
        return cell + res

    def subs(self, *args, **kwargs):
        return CrystalStructure(spf.subs(self._crys.cell, *args, **kwargs), spf.subs(self._crys.atoms, *args, **kwargs))


# To be implemented in the future
    # def createParametrizedCrystal(self, cell=True, atoms=True, U=False):
    #     conv = self._crys.createConventionalCell()
    #     if cell:
    #         cell = self.__reduceCellBySymmetry(self._crys.cell)
    #     else:
    #         cell = self._crys.cell
    #     if atoms:
    #         atm = [Atom(at.element, self.__createPParam(i, at), U=self.__createUParam(i, U)) for i, at in enumerate(self._crys.atoms)]
    #     else:
    #         atm = copy.deepcopy(self._crys.atoms)
    #     c = CrystalStructure(cell, atm)
    #     if atoms:
    #         c = self.__reduceAtomsBySymmetry(conv, c, U)
    #     default = self.__findDefaults(c, conv)
    #     c.defaultParameters = default
    #     return c

    # def defaultCrystal(self):
    #     return self._crys.subs(self._crys.defaultParameters)

    def __createPParam(self, i, at):
        return sp.symbols("x_" + at.element + str(i + 1) + "," + "y_" + at.element + str(i + 1) + "," + "z_" + at.element + str(i + 1))

    def __createUParam(self, i, enabled):
        if not enabled:
            return 0
        U1, U2, U3, U4, U5, U6 = sp.symbols("U_11, U_22, U_33, U_12, U_23, U_31".replace("_", "_" + str(i + 1) + "_"))
        U = np.array([[U1, U4, U6], [U4, U2, U5], [U6, U5, U3]])
        return U

    def __findDefaults(self, p, orig):
        res = {}
        for c1, c2 in zip(p.cell, orig.cell):
            if type(c1) == sp.Symbol:
                res[c1.name] = c2
        pos_new = np.array([at.Position for at in p.atoms])
        pos_orig = np.array([at.Position for at in orig.atoms])
        for pos1, pos2 in zip(pos_new, pos_orig):
            for p1, p2 in zip(pos1, pos2):
                if type(p1) == sp.Symbol:
                    res[p1.name] = p2
        return res

    def __reduceCellBySymmetry(self, cell, eps=1e-5):
        a = sp.symbols("a")
        if abs(cell[0] - cell[1]) < eps:
            b = a
        else:
            b = sp.symbols("b")
        if abs(cell[0] - cell[2]) < eps:
            c = a
        else:
            c = sp.symbols("c")
        if abs(cell[3] - 120) < eps:
            alpha = 120
        elif abs(cell[3] - 90) < eps:
            alpha = 90
        else:
            alpha = sp.symbols("alpha")
        if abs(cell[4] - 120) < eps:
            beta = 120
        elif abs(cell[4] - 90) < eps:
            beta = 90
        else:
            beta = sp.symbols("beta")
        if abs(cell[5] - 120) < eps:
            gamma = 120
        elif abs(cell[5] - 90) < eps:
            gamma = 90
        else:
            gamma = sp.symbols("alpha")
        return [a, b, c, alpha, beta, gamma]

    def __reduceAtomsBySymmetry(self, orig, new, enableU):
        """
        orig: The original crystal structure that contains actual position as number.
        new: The parametrized crystal structure that contains sympy symbols.
        """
        ops = spglib.get_symmetry(orig._toSpg())
        R, T = ops["rotations"], ops["translations"]

        indice, offset = self.__findRelation(orig, R, T)

        pos_new = np.array([at.Position for at in new.atoms])
        symmetrized = pos_new[indice]
        R = np.vectorize(lambda x: sp.Rational(np.round(x * 12), 12))(R)
        T = np.vectorize(lambda x: sp.Rational(np.round(x * 12), 12))(T)

        pos_r = spf.einsum("Mij,Nj->MNi", R, pos_new)
        pos_rt = np.array([p + t for p, t in zip(pos_r, T)])
        pos_rt = pos_rt + offset

        res = {}
        for pos, sym in zip(pos_rt, symmetrized):
            eqs = np.vectorize(lambda x, y: sp.Eq(x.subs(res), y.subs(res)))(pos, sym)
            res.update(sp.solve(eqs.flatten()))
        res = sp.solve([sp.Eq(key, value) for key, value in res.items()])

        if enableU:
            U_new = np.array([at.Uani for at in new.atoms])
            U_symmetrized = U_new[indice]
            U_r = spf.einsum("Mij,Njk,Mlk->MNil", R, U_new, R)

            res_U = {}
            for u, u2 in zip(U_r, U_symmetrized):
                eqs = np.vectorize(lambda x, y: sp.Eq(x.subs(res), y.subs(res)))(u, u2)
                res_U.update(sp.solve(eqs.flatten()))
            res_U = sp.solve([sp.Eq(key, value) for key, value in res_U.items()])
            res.update(res_U)
        return new.subs(res)

    def __findRelation(self, orig, R, T, eps=1e-5):
        pos_orig = np.array([at.Position for at in orig.atoms])
        # apply symmetry ops.
        pos_orig_r = np.einsum("Mij,Nj->MNi", R, pos_orig)        # M: symmetri operatrion, N: atoms, j: x,y,z
        pos_orig_rt = np.array([p + t for p, t in zip(pos_orig_r, T)])
        offset = np.where(pos_orig_rt < -eps, 1, 0) + np.where(pos_orig_rt > 1 - eps, -1, 0)
        pos_orig_rt = pos_orig_rt + offset
        offset2 = np.where(pos_orig_rt < -eps, 1, 0) + np.where(pos_orig_rt > 1 - eps, -1, 0)
        pos_orig_rt = pos_orig_rt + offset2
        # find indice
        indice = np.array([[np.argmin(np.linalg.norm(pos_orig - pos_orig_rt[m, n], axis=1)) for n in range(len(pos_orig))] for m in range(len(R))])
        return indice, offset + offset2
