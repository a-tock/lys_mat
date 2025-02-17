import numpy as np
import sympy as sp
import spglib
import copy
from . import sympyFuncs as spf
from .Atom import Atom
from .CrystalStructure import CrystalStructure
from sympy.solvers.diophantine.diophantine import diop_linear


class SuperStructure(object):
    def __init__(self, crys):
        self._crys = crys

    def createPrimitiveCell(self):
        cell = self._crys._toSpg()
        lattice, pos, numbers = spglib.find_primitive(cell)
        elems = self._crys.getElements()
        atoms = [Atom(elems[n - 1], p) for n, p in zip(numbers, pos)]
        return CrystalStructure(lattice, atoms)

    def createConventionalCell(self, idealize=False, symprec=1e-5):
        cell = self._crys._toSpg()
        lattice, pos, numbers = spglib.standardize_cell(cell, to_primitive=False, no_idealize=not idealize, symprec=symprec)
        elems = self._crys.getElements()
        atoms = [Atom(elems[n - 1], p) for n, p in zip(numbers, pos)]
        return CrystalStructure(lattice, atoms)

    def refinedCell(self, prec=5):
        cell = (round(self._crys.a, prec), round(self._crys.b, prec), round(self._crys.c, prec), round(self._crys.alpha, prec), round(self._crys.beta, prec), round(self._crys.gamma, prec))
        atoms = copy.deepcopy(self._crys.atoms)
        for at in atoms:
            at.Position = [round(at.Position[0], prec), round(at.Position[1], prec), round(at.Position[2], prec)]
        return CrystalStructure(cell, atoms)

    def createSupercell(self, P):
        from ..Phonopy import Phonopy
        if isinstance(P, CrystalStructure):
            P = self.calculateSupercell(P)
        P = np.array(P)
        if P.shape == (3,):
            P = np.array([[P[0], 0, 0], [0, P[1], 0], [0, 0, P[2]]])
        S = np.linalg.inv(P)
        # return Phonopy.generateSupercell(self._crys, np.array(P))
        shifts = self.__find_shift(P)
        cell = P.dot(self._crys.unit)
        atoms = []
        for at in self._crys.atoms:
            for s in shifts:
                atom = at.duplicate()
                pos = S.dot(at.Position + np.array(s))
                if (pos < 1).all() and (pos >= 0).all():
                    atom.Position = pos
                    atoms.append(atom)
        c = CrystalStructure(cell, atoms)
        return c

    def __find_shift(self, P):
        from collections import deque
        S = np.linalg.inv(P)
        valid, invalid = {(0, 0, 0)}, set()
        queue = deque([(0, 0, 0)])
        while queue:
            n = queue.popleft()
            valid.add(n)
            for neighbor in [(n[0] + 1, n[1], n[2]), (n[0] - 1, n[1], n[2]), (n[0], n[1] + 1, n[2]), (n[0], n[1] - 1, n[2]), (n[0], n[1], n[2] + 1), (n[0], n[1], n[2] - 1)]:
                if neighbor not in valid and neighbor not in invalid:
                    if (0 <= S.dot(neighbor)).all() and (S.dot(neighbor) < 1).all():
                        valid.add(neighbor)
                        queue.append(neighbor)
                    else:
                        invalid.add(neighbor)
        return valid

    def calculateSupercell(self, reference):
        P = self.__LowerSymmetry(reference)
        return P

    def __LowerSymmetry(self, reference):
        R = self.__rotation(reference)
        u1 = R.dot(self._crys.unit.T).T
        S = reference.unit.dot(np.linalg.inv(u1))
        S = np.round(S).astype(int)
        return S

    def __rotation(self, reference):
        def angle(a, b):
            return np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
        rf = self.__modify(reference)

        def rotmat(Ang, n):
            R = np.array([[np.cos(Ang) + n[0] * n[0] * (1 - np.cos(Ang)), n[0] * n[1] * (1 - np.cos(Ang)) - n[2] * np.sin(Ang), n[0] * n[2] * (1 - np.cos(Ang)) + n[1] * np.sin(Ang)],
                          [n[1] * n[0] * (1 - np.cos(Ang)) + n[2] * np.sin(Ang), np.cos(Ang) + n[1] * n[1] * (1 - np.cos(Ang)), n[1] * n[2] * (1 - np.cos(Ang)) - n[0] * np.sin(Ang)],
                          [n[2] * n[0] * (1 - np.cos(Ang)) - n[1] * np.sin(Ang), n[2] * n[1] * (1 - np.cos(Ang)) + n[0] * np.sin(Ang), np.cos(Ang) + n[2] * n[2] * (1 - np.cos(Ang))]])
            return R

        # choose nearest atom and the second nearest atom
        s = self.__modify(self._crys).getAtomicPositions()
        a = s[np.argsort(np.linalg.norm(s, axis=1))[1]]
        b = s[np.argsort(np.linalg.norm(s, axis=1))[2]]

        # Find ux for reference.
        dist1 = np.linalg.norm(rf.getAtomicPositions(), axis=1) - np.linalg.norm(a)
        ux = rf.unit.T.dot(rf.atoms[np.argmin(np.abs(dist1))].Position)

        # Find atoms with length ~b
        dist2 = np.linalg.norm(rf.getAtomicPositions(), axis=1) - np.linalg.norm(b)
        rf.atoms = [at for at, d in zip(rf.atoms, dist2) if np.abs(d) < np.linalg.norm(b) * 0.2]

        dangle1 = np.abs([angle(ux, d) for d in rf.getAtomicPositions()]) - np.abs(angle(a, b))
        rf.atoms = [at for at, angle in zip(rf.atoms, dangle1) if np.abs(angle) < 10 / 180 * np.pi]
        uy = rf.getAtomicPositions()[0]

        theta1 = angle(a, ux)
        R1 = rotmat(theta1, np.cross(a, ux) / np.linalg.norm(np.cross(a, ux)))
        theta2 = angle(b, R1.dot(uy))
        n = R1.dot(np.cross(b, uy) / np.linalg.norm(np.cross(b, uy)))
        R2 = rotmat(theta2, n)
        return R2.dot(R1)

    def __modify(self, reference):
        element0 = [at for at in self._crys.atoms if at.Position == [0, 0, 0]][0].element
        rf = reference.createSupercell([4, 4, 4])
        rf.atoms = [at for at in rf.atoms if at.element == element0]
        dist = np.linalg.norm(rf.getAtomicPositions() - reference.unit.T.dot([2, 2, 2]), axis=1)
        origin = rf.atoms[np.argmin(dist)].Position
        for at in rf.atoms:
            at.Position = (np.array(at.Position) - np.array(origin)).tolist()
        return rf

    def _unimodularBasisTransformation(self, hkl):
        """
        Calculate transformation matrix P that satisfies
        (a',b',c') = (a,b,c)P = unit.dot(P)
        where the direction [hkl] is parallel to a' x b'

        Args:
            hkl(sequence of length 3): The hkl that specifies Miller plane hkl.
            unit(3*3 matrix): The a,b, and c vector of the crystal.
        """
        hkl = np.array(hkl / np.gcd.reduce(hkl), dtype=int)
        zero_number = sum(hkl == 0)

        if zero_number == 2:
            return np.roll(np.eye(3, dtype=int), 2 - np.argmax(hkl), axis=0)

        elif zero_number == 1:
            if hkl[0] == 0:
                y, z, _ = sp.gcdex(hkl[1], hkl[2])
                return np.array([[1, 0, 0], [0, hkl[2], -hkl[1]], [0, int(y), int(z)]], dtype=int)
            elif hkl[1] == 0:
                x, z, _ = sp.gcdex(hkl[0], hkl[2])
                return np.array([[0, 1, 0], [-hkl[2], 0, hkl[0]], [int(x), 0, int(z)]], dtype=int)
            else:
                x, y, _ = sp.gcdex(hkl[0], hkl[1])
                return np.array([[0, 0, 1], [hkl[1], -hkl[0], 0], [int(x), int(y), 0]], dtype=int)

        else:
            xyz = np.array(sp.symbols("x, y, z", integer=True))
            ans_ab = diop_linear(hkl.dot((xyz)))  # find hx + ky + lz = 0 where x,y,z in Z
            free_symbols = np.array(list(ans_ab.free_symbols))
            indice = np.argsort([str(sym) for sym in free_symbols])
            free_symbols = free_symbols[indice]
            a, b = [np.array([a.coeff(sym, 1) for a in ans_ab], dtype=int) for sym in free_symbols]
            a, b = self.__2d_basic_vector_shorten(a, b, self._crys.unit)
            a, b = self.__flip_vectors(a, b)

            P = sp.Matrix([a, b, xyz])
            ans_c = diop_linear(P.det() - 1)
            c = np.array([sp.Poly(a, *ans_c.free_symbols).TC() for a in ans_c], dtype=int)
            c = self.__3d_basic_vector_shorten(c, a, b, self._crys.unit)
            return np.array([a, b, c], dtype=int)

    def __2d_basic_vector_shorten(self, a_vector, b_vector, unit):
        sort_index = np.array([0, 0])
        while (sort_index[:2] == [0, 1]).all():
            vector_list = np.array([a_vector, b_vector, a_vector + b_vector, b_vector - a_vector])
            sort_index = np.argsort(np.linalg.norm(vector_list.dot(unit), axis=1))
            a_vector, b_vector = vector_list[sort_index[:2]]
        return a_vector, b_vector

    def __3d_basic_vector_shorten(self, c_vector, a_vector, b_vector, unit):
        sort_index = -1
        while sort_index != 0:
            vector_list = np.array([c_vector, c_vector - a_vector, c_vector + a_vector, c_vector - b_vector, c_vector + b_vector])
            sort_index = np.argmin(np.linalg.norm(vector_list.dot(unit), axis=1))
            c_vector = vector_list[sort_index]
        return c_vector

    def __flip_vectors(self, a, b):
        if a[0] < 0:
            a = a * -1
        if b[1] < 0:
            b = b * -1
        return a, b

    def transformedCrystal(self, hkl, returnP_T=False):
        """
        calculate new crystal structure whose normal vector is directed to hkl direction of original crystal. 

        Args:
            hkl(length 3 sequence of integer): Miller indice for original crystal

        Returns:
            CrystalStructure: the new crystal structure
        """

        if spf.isSympyObject(self._crys.unit):
            crys_subs = self._crys.defaultCrystal()
            P_T = crys_subs._unimodularBasisTransformation(hkl)
        else:
            P_T = self._crys._unimodularBasisTransformation(hkl)
        new_unit = P_T.dot(self._crys.unit)
        new_atoms = [self._transform_atom(atom, P_T, self._crys.unit, new_unit) for atom in self._crys.atoms]
        if returnP_T:
            return CrystalStructure(new_unit, new_atoms), P_T
        else:
            return CrystalStructure(new_unit, new_atoms)

    def _transform_atom(self, atom, P_T, unit, new_unit):
        new_position = self.__new_Atom_position(atom.Position, P_T)
        new_U = self.__new_U_maker(atom.Uani, P_T, unit, new_unit)
        return Atom(atom.element, new_position, new_U, occupancy=atom.occupancy)

    def __new_Atom_position(self, old_position, P_T):
        new_position = old_position.dot(np.linalg.inv(P_T))
        if np.array([isinstance(p, (sp.Symbol, sp.Mul, sp.Add)) for p in old_position]).any():
            return new_position
        else:
            new_position = np.where(np.modf(new_position)[0] >= 0, np.modf(new_position)[0], np.modf(new_position)[0] + 1) + 0
            return np.where(np.isclose(new_position, 1), 0, new_position)

    def __new_U_maker(self, U, P_T, old_unit, new_unit):
        if spf.isSympyObject(old_unit):
            old_inv = np.array([sp.Matrix(v).norm() for v in self.___make_inverse(old_unit)])
            new_inv = np.array([sp.Matrix(v).norm() for v in self.___make_inverse(new_unit)])
            Q = np.diag(old_inv).dot(np.array(sp.Matrix(P_T).inv())).dot(np.diag(1 / new_inv))
            return (Q.T).dot(U).dot(Q)
        else:
            old_inv = np.linalg.norm(self.___make_inverse(old_unit), axis=1)
            new_inv = np.linalg.norm(self.___make_inverse(new_unit), axis=1)
            Q = np.diag(old_inv).dot(np.linalg.inv(P_T)).dot(np.diag(1 / new_inv))
            return Q.T.dot(U).dot(Q)

    def ___make_inverse(self, unit):
        lib = sp if spf.isSympyObject(unit) else np
        res = []
        a, b, c = unit
        res.append(np.cross(b, c) / np.dot(a, np.cross(b, c)) * 2 * lib.pi)
        res.append(np.cross(c, a) / np.dot(b, np.cross(c, a)) * 2 * lib.pi)
        res.append(np.cross(a, b) / np.dot(c, np.cross(a, b)) * 2 * lib.pi)
        return np.array(res)
