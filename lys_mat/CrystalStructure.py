import spglib  # BSD-3-Clause
import seekpath  # MIT
import numpy as np  # BSD
import copy  # BSD
import scipy  # BSD
import sympy as sp  # BSD
import inspect  # BSD
import random  # BSD
import re  # BSD

from .Atom import Atom
from .ScatteringFactors import ScatteringFactor
from . import sympyFuncs as spf
from .CrystalBase import CrystalBase
from .freeEnergy import FreeEnergy  # MIT
from .Pair_mesh import makePair
from .MD_Analysis import MakePCF  # GNU
from CifFile import CifFile, ReadCif  # free
from sympy.solvers.diophantine.diophantine import diop_linear

NA = 6.0221409e+23


class _CifIO(CrystalBase):
    @staticmethod
    def from_cif(file, index=0):
        cf = ReadCif(file)
        cf = cf[cf.keys()[index]]
        cell = []
        cell.append(float(cf['_cell_length_a']))
        cell.append(float(cf['_cell_length_b']))
        cell.append(float(cf['_cell_length_c']))
        cell.append(float(cf['_cell_angle_alpha']))
        cell.append(float(cf['_cell_angle_beta']))
        cell.append(float(cf['_cell_angle_gamma']))
        list = []
        if '_atom_site_type_symbol' in cf:
            type = '_atom_site_type_symbol'
        else:
            type = '_atom_site_label'
        for i in range(len(cf[type])):
            name = cf[type][i]
            x = float(cf['_atom_site_fract_x'][i])
            y = float(cf['_atom_site_fract_y'][i])
            z = float(cf['_atom_site_fract_z'][i])
            if '_atom_site_occupancy' in cf:
                occu = float(cf['_atom_site_occupancy'][i])
            else:
                occu = 1
            if '_atom_site_U_iso_or_equiv' in cf:
                U = float(cf['_atom_site_U_iso_or_equiv'][i])
            else:
                U = 0
            if '_atom_site_aniso_U_11' in cf:
                U = [0, 0, 0, 0, 0, 0]
                U[0] = float(cf['_atom_site_aniso_U_11'][i])
                U[1] = float(cf['_atom_site_aniso_U_22'][i])
                U[2] = float(cf['_atom_site_aniso_U_33'][i])
                U[3] = float(cf['_atom_site_aniso_U_12'][i])
                U[4] = float(cf['_atom_site_aniso_U_13'][i])
                U[5] = float(cf['_atom_site_aniso_U_23'][i])
            list.append(Atom(name, [x, y, z], U=U, Occupancy=occu))
        if '_symmetry_equiv_pos_as_xyz' in cf:
            sym = [_CifIO.__strToSym(s) for s in cf['_symmetry_equiv_pos_as_xyz']]
        elif '_space_group_symop_operation_xyz' in cf:
            sym = [_CifIO.__strToSym(s) for s in cf['_space_group_symop_operation_xyz']]
        else:
            sym = None
        return CrystalStructure(cell, list, sym)

    def exportAsCif(self, exportAll=True):
        return self._exportAsCif(self.crys, exportAll=exportAll)

    def saveAsCif(self, file):
        with open(file, mode="w") as f:
            f.write(self._exportAsCif(self.crys, exportAll=False))

    def exportTo(self, file, ext=".pcs"):
        if ext == ".cif":
            txt = self.exportAsCif()
        elif ext == ".pcs":
            txt = str(self._exportAsDic())
        if not file.endswith(ext):
            file = file + ext
        with open(file, "w") as f:
            f.write(txt)

    @staticmethod
    def importFrom(file, ext=".pcs"):
        if ext == ".pcs":
            with open(file, "r") as f:
                txt = f.read()
            return _CifIO._importFromDic(eval(txt))
        if ext == ".cif":
            return _CifIO.from_cif(file)

    def _exportAsDic(self):
        d = {}
        cell = []
        for c in self.crys.cell:
            if spf.isSympyObject(c):
                cell.append(str(c))
            else:
                cell.append(c)
        symbols = [str(s) for s in spf.free_symbols(self.crys.cell)]
        atoms = [at.saveAsDictionary() for at in self.crys.atoms]
        d["free_symbols"] = symbols
        d["cell"] = cell
        d["atoms"] = atoms
        return d

    @staticmethod
    def _importFromDic(d):
        if len(d["free_symbols"]) > 0:
            symbols = sp.symbols(",".join(d["free_symbols"]))
        cell = []
        for c in d["cell"]:
            if isinstance(c, str):
                cell.append(sp.sympify(c))
            else:
                cell.append(c)
        atoms = [Atom.loadFromDictionary(at) for at in d["atoms"]]
        return CrystalStructure(cell, atoms)

    def _exportAsCif(self, crys, exportAll=True):
        c = CifFile()
        c.NewBlock("crystal1")
        cf = c[c.keys()[0]]
        cf['_cell_length_a'] = crys.a
        cf['_cell_length_b'] = crys.b
        cf['_cell_length_c'] = crys.c
        cf['_cell_angle_alpha'] = crys.alpha
        cf['_cell_angle_beta'] = crys.beta
        cf['_cell_angle_gamma'] = crys.gamma
        if exportAll:
            atoms = crys.atoms
        else:
            atoms = crys.irreducibleAtoms()
            data = spglib.get_symmetry_dataset(crys._toSpg())
            ops = spglib.get_symmetry(crys._toSpg())
            cf['_symmetry_Int_Tables_number'] = data["number"]
            cf['_symmetry_equiv_pos_as_xyz'] = [_CifIO.__symToStr(r, t) for r, t in zip(ops['rotations'], ops['translations'])]
            cf.CreateLoop(['_symmetry_equiv_pos_as_xyz'])
        newlabel = []
        elem = None
        i = 0
        for at in atoms:
            if elem != at.Element:
                elem = at.Element
                i = 0
            else:
                i += 1
            newlabel.append(elem + str(i))
        cf['_atom_site_label'] = newlabel
        cf['_atom_site_type_symbol'] = [at.Element for at in atoms]
        cf['_atom_site_fract_x'] = [at.Position[0] for at in atoms]
        cf['_atom_site_fract_y'] = [at.Position[1] for at in atoms]
        cf['_atom_site_fract_z'] = [at.Position[2] for at in atoms]
        cf.CreateLoop(['_atom_site_label', '_atom_site_type_symbol', '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z'])

        newlabel = []
        elem = None
        i = 0
        for at in atoms:
            if elem != at.Element:
                elem = at.Element
                i = 0
            else:
                i += 1
            newlabel.append(elem + str(i))
        cf['_atom_site_aniso_label'] = newlabel
        cf['_atom_site_aniso_U_11'] = [at.Uani[0, 0] for at in atoms]
        cf['_atom_site_aniso_U_22'] = [at.Uani[1, 1] for at in atoms]
        cf['_atom_site_aniso_U_33'] = [at.Uani[2, 2] for at in atoms]
        cf['_atom_site_aniso_U_12'] = [at.Uani[0, 1] for at in atoms]
        cf['_atom_site_aniso_U_13'] = [at.Uani[0, 2] for at in atoms]
        cf['_atom_site_aniso_U_23'] = [at.Uani[1, 2] for at in atoms]
        cf.CreateLoop(['_atom_site_aniso_label', '_atom_site_aniso_U_11', '_atom_site_aniso_U_22', '_atom_site_aniso_U_33', '_atom_site_aniso_U_12', '_atom_site_aniso_U_13', '_atom_site_aniso_U_23'])
        return str(c)

    @staticmethod
    def __symToStr(rotation, trans):
        def __xyz(v, axis):
            if v == 1:
                return "+" + axis
            elif v == -1:
                return "-" + axis
            elif v == 0:
                return ""
            else:
                print("[symToStr (CrystalStructure)] error001: ", v, axis)
                if v > 0:
                    return "+" + str(v) + axis
                elif v < 0:
                    return str(v) + axis
                else:
                    return

        def __trans(v):
            if abs(v) < 1e-5:
                return ""
            elif abs(v - 1 / 2) < 1e-5:
                return "1/2"
            elif abs(v - 1 / 3) < 1e-5:
                return "1/3"
            elif abs(v - 2 / 3) < 1e-5:
                return "2/3"
            elif abs(v - 1 / 4) < 1e-5:
                return "1/4"
            elif abs(v - 3 / 4) < 1e-5:
                return "3/4"
            elif abs(v - 1 / 6) < 1e-5:
                return "1/6"
            elif abs(v - 5 / 6) < 1e-5:
                return "5/6"
            else:
                return str(v)
        res = ""
        for r, t in zip(rotation, trans):
            res += __trans(t)
            res += __xyz(r[0], "x")
            res += __xyz(r[1], "y")
            res += __xyz(r[2], "z")
            res += ","
        return res[:-1]

    @staticmethod
    def __strToSym(str):
        def __xyz(s, axis):
            s_axis = re.findall(r"[+-]?" + "\d*" + axis, s)
            if len(s_axis) == 0:
                return 0
            else:
                s = s_axis[0][:-1]
                if s == "-":
                    return -1
                elif s == "+":
                    return 1
                elif s == "":
                    return 1
                else:
                    return int(s_axis[0][:-1])

        def __trans(s):
            if "1/2" in s:
                return 1 / 2
            elif "1/3" in s:
                return 1 / 3
            elif "2/3" in s:
                return 2 / 3
            elif "1/4" in s:
                return 1 / 4
            elif "3/4" in s:
                return 3 / 4
            elif "1/6" in s:
                return 1 / 6
            elif "5/6" in s:
                return 5 / 6
            else:
                return 0
        rotations = [[__xyz(s, axis) for axis in ["x", "y", "z"]] for s in str.split(",")]
        trans = [__trans(s) for s in str.split(",")]
        return rotations, trans


class _Lattice(CrystalBase):
    def __init__(self, crys, cell):
        super().__init__(crys)
        if len(cell) == 6:
            self._cell = np.array(cell)
        else:
            self._cell = self.__unitToCell(cell)

    def __unitToCell(self, unit):
        if spf.isSympyObject(unit):
            unit_pow = unit.dot(unit.T)
            a, b, c = sp.sqrt(unit_pow[0, 0]), sp.sqrt(unit_pow[1, 1]), sp.sqrt(unit_pow[2, 2])
            gamma = sp.re(sp.acos(unit_pow[0, 1] / (a * b))) / sp.pi * 180
            alpha = sp.re(sp.acos(unit_pow[1, 2] / (b * c))) / sp.pi * 180
            beta = sp.re(sp.acos(unit_pow[2, 0] / (c * a))) / sp.pi * 180
            return np.array([a, b, c, alpha, beta, gamma])  # a,b,c type is sympy.Add, alpha,beta,gamma type is sympy.Mul
        else:
            def angle(v1, v2): return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
            a, b, c = np.linalg.norm(unit, axis=1)
            gamma, alpha, beta = angle(unit[0], unit[1]), angle(unit[1], unit[2]), angle(unit[2], unit[0])
            return np.array([a, b, c, alpha, beta, gamma])

    def InverseLatticeVectors(self):
        lib = sp if spf.isSympyObject(self.crys.unit) else np
        res = []
        a, b, c = self.crys.unit
        res.append(np.cross(b, c) / np.dot(a, np.cross(b, c)) * 2 * lib.pi)
        res.append(np.cross(c, a) / np.dot(b, np.cross(c, a)) * 2 * lib.pi)
        res.append(np.cross(a, b) / np.dot(c, np.cross(a, b)) * 2 * lib.pi)
        return np.array(res)

    @property
    def unit(self):
        lib = sp if spf.isSympyObject(self._cell) else np
        unit = []
        a, b, c = self._cell[0:3]
        al, be, ga = self._cell[3:6] / 180 * lib.pi
        unit.append([a, 0, 0])
        unit.append([b * lib.cos(ga), b * lib.sin(ga), 0])
        unit.append([c * lib.cos(be), c * (lib.cos(al) - lib.cos(be) * lib.cos(ga)) / lib.sin(ga), c * lib.sqrt(1 + 2 * lib.cos(al) * lib.cos(be) * lib.cos(ga) - lib.cos(al) * lib.cos(al) - lib.cos(be) * lib.cos(be) - lib.cos(ga) * lib.cos(ga)) / lib.sin(ga)])
        return np.array(unit)

    @property
    def inv(self):
        return self.InverseLatticeVectors()

    @property
    def a(self):
        return self._cell[0]

    @property
    def b(self):
        return self._cell[1]

    @property
    def c(self):
        return self._cell[2]

    @property
    def alpha(self):
        return self._cell[3]

    @property
    def beta(self):
        return self._cell[4]

    @property
    def gamma(self):
        return self._cell[5]

    @property
    def cell(self):
        return self._cell

    def latticeInfo(self):
        if spf.isSympyObject(self._cell):
            res = "a = {:}, b = {:}, c={:}".format(self.a, self.b, self.c)
            res += ", alpha = {:}, beta={:}, gamma = {:}\n".format(self.alpha, self.beta, self.gamma)
        else:
            res = "a = {:.5f}, b = {:.5f}, c={:.5f}".format(self.a, self.b, self.c)
            res += ", alpha = {:.5f}, beta={:.5f}, gamma = {:.5f}\n".format(self.alpha, self.beta, self.gamma)
        return res

    def Volume(self):
        a, b, c = self.unit
        return np.dot(a, np.cross(b, c))


class _Atoms(CrystalBase):
    def __init__(self, crys, atoms, sym):
        super().__init__(crys)
        self.setAtoms(atoms, sym)

    def __extractAtoms(self, atoms, sym):
        def is_same(p1, p2, prec=1e-5):
            if abs(p2[0] - p1[0]) < prec or 1 - abs(p2[0] - p1[0]) < prec:
                if abs(p2[1] - p1[1]) < prec or 1 - abs(p2[1] - p1[1]) < prec:
                    if abs(p2[2] - p1[2]) < prec or 1 - abs(p2[2] - p1[2]) < prec:
                        return True
            return False
        result = []
        for at in atoms:
            plist = []
            for (R, T) in sym:
                pos = np.dot(R, at.Position) + T
                if pos[0] < 0:
                    pos[0] += 1
                if pos[1] < 0:
                    pos[1] += 1
                if pos[2] < 0:
                    pos[2] += 1
                if pos[0] >= 1:
                    pos[0] -= 1
                if pos[1] >= 1:
                    pos[1] -= 1
                if pos[2] >= 1:
                    pos[2] -= 1
                flg = True
                for p in plist:
                    if is_same(p, pos, 1e-3):
                        flg = False
                if flg:
                    plist.append(pos)
            for p in plist:
                flg = True
                for at2 in result:
                    if at.Element == at2.Element:
                        if is_same(p, at2.Position, 1e-3):
                            flg = False
                if flg:
                    result.append(Atom(at.Element, p, U=at.Uani))
        return result

    def setAtoms(self, atoms, sym=None):
        if sym is None:
            self._atoms = copy.deepcopy(atoms)
        else:
            self._atoms = self.__extractAtoms(atoms, sym)
        self.__reorderAtoms()

    def getAtoms(self):
        return self._atoms

    def __reorderAtoms(self):
        result = []
        for e in self.getElements():
            for at in self._atoms:
                if at.Element == e:
                    result.append(at)
        self._atoms = result

    def getElements(self):
        elements = []
        for at in self._atoms:
            if at.Element not in elements:
                elements.append(at.Element)
        return sorted(elements)

    def getAtomicPositions(self, external=True):
        if external:
            u = np.array(self.crys.unit.T)
            return np.array([u.dot(at.Position) for at in self._atoms])
        else:
            np.array([at.Position for at in self._atoms])

    def irreducibleAtoms(self):
        sym = spglib.get_symmetry_dataset(self.crys._toSpg())
        return [self._atoms[i] for i in list(set(sym["equivalent_atoms"]))]

    def atomInfo(self, max_atoms=-1):
        res = "--- atoms (" + str(len(self._atoms)) + ") ---"
        for i, at in enumerate(self._atoms):
            res += "\n" + str(i + 1) + ": " + str(at)
            if i == max_atoms:
                res += "..."
                break
        return res


class _Symmetry(CrystalBase):
    def standardPath(self):
        paths = seekpath.get_path(self._toSpg())["path"]
        res = [paths[0][0]]
        for pp in paths:
            for p in pp:
                if res[-1] != p:
                    res.append(p)
        return res

    def symmetryPoints(self):
        return seekpath.get_path(self._toSpg())["point_coords"]

    def _toSpg(self):
        crys = self.crys
        if crys.isSympyObject():
            crys = crys.subs({s: random.random() for s in crys.free_symbols})
        lattice = crys.unit
        pos = []
        num = []
        for i, e in enumerate(crys.getElements()):
            for at in crys.atoms:
                if at.Element == e:
                    pos.append(at.Position)
                    num.append(i + 1)
        return lattice, pos, num

    def crystalSystem(self):
        n = spglib.get_symmetry_dataset(self._toSpg())["number"]
        if n < 3:
            return "triclinic"
        elif n < 16:
            return "monoclinic"
        elif n < 75:
            return "orthorhombic"
        elif n < 143:
            return "tetragonal"
        elif n < 168:
            return "trigonal"
        elif n < 195:
            return "hexagonal"
        else:
            return "cubic"

    def symmetryInfo(self):
        try:
            data = spglib.get_symmetry_dataset(self._toSpg())
            return "Symmetry: " + self.crystalSystem() + " " + data["international"] + " (No. " + str(data["number"]) + "), Point group: " + data["pointgroup"] + "\n"
        except Exception:
            return "Failed to find symmetry\n"

    def getSymmetryOperations(self, pointGroup=False):
        ops = spglib.get_symmetry(self._toSpg())
        if pointGroup:
            return [r for r, t in zip(ops['rotations'], ops['translations']) if np.allclose(t, [0, 0, 0])]
        else:
            return ops['rotations'], ops['translations']


class _SuperStructure(CrystalBase):
    def createPrimitiveCell(self):
        cell = self.crys._toSpg()
        lattice, pos, numbers = spglib.find_primitive(cell)
        elems = self.crys.getElements()
        atoms = [Atom(elems[n - 1], p) for n, p in zip(numbers, pos)]
        return CrystalStructure(lattice, atoms)

    def createConventionalCell(self, idealize=False, symprec=1e-5):
        cell = self.crys._toSpg()
        lattice, pos, numbers = spglib.standardize_cell(cell, to_primitive=False, no_idealize=not idealize, symprec=symprec)
        elems = self.crys.getElements()
        atoms = [Atom(elems[n - 1], p) for n, p in zip(numbers, pos)]
        return CrystalStructure(lattice, atoms)

    def refinedCell(self, prec=5):
        cell = (round(self.crys.a, prec), round(self.crys.b, prec), round(self.crys.c, prec), round(self.crys.alpha, prec), round(self.crys.beta, prec), round(self.crys.gamma, prec))
        atoms = copy.deepcopy(self.crys.atoms)
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
        # return Phonopy.generateSupercell(self.crys, np.array(P))
        shifts = self.__find_shift(P)
        cell = P.dot(self.crys.unit)
        atoms = []
        for at in self.crys.atoms:
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
        u1 = R.dot(self.crys.unit.T).T
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
        s = self.__modify(self.crys).getAtomicPositions()
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
        element0 = [at for at in self.crys.atoms if at.Position == [0, 0, 0]][0].Element
        rf = reference.createSupercell([4, 4, 4])
        rf.atoms = [at for at in rf.atoms if at.Element == element0]
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
            a, b = self.__2d_basic_vector_shorten(a, b, self.crys.unit)
            a, b = self.__flip_vectors(a, b)

            P = sp.Matrix([a, b, xyz])
            ans_c = diop_linear(P.det() - 1)
            c = np.array([sp.Poly(a, *ans_c.free_symbols).TC() for a in ans_c], dtype=int)
            c = self.__3d_basic_vector_shorten(c, a, b, self.crys.unit)
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

        if spf.isSympyObject(self.crys.unit):
            crys_subs = self.crys.defaultCrystal()
            P_T = crys_subs._unimodularBasisTransformation(hkl)
        else:
            P_T = self.crys._unimodularBasisTransformation(hkl)
        new_unit = P_T.dot(self.crys.unit)
        new_atoms = [self._transform_atom(atom, P_T, self.crys.unit, new_unit) for atom in self.crys.atoms]
        if returnP_T:
            return CrystalStructure(new_unit, new_atoms), P_T
        else:
            return CrystalStructure(new_unit, new_atoms)

    def _transform_atom(self, atom, P_T, unit, new_unit):
        new_position = self.__new_Atom_position(atom.Position, P_T)
        new_U = self.__new_U_maker(atom.Uani, P_T, unit, new_unit)
        return Atom(atom.Element, new_position, new_U, Occupancy=atom.Occupancy)

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


class _Strain(CrystalBase):
    def createStrainedCrystal(self, eps):  # 1,2,3,4,5,6 = xx,yy,zz,xy,yz,zx
        R = np.array([[1 + eps[0], eps[3], eps[5]], [eps[3], 1 + eps[1], eps[4]], [eps[5], eps[4], 1 + eps[2]]])
        return CrystalStructure(R.dot(self.crys.unit.T).T, self.crys.atoms)

    def calculateStrain(self, ref):
        R = self.crys.unit.T.dot(np.linalg.inv(ref.unit.T))
        U, P = scipy.linalg.polar(R)
        return (P[0, 0] - 1, P[1, 1] - 1, P[2, 2] - 1, P[0, 1], P[1, 2], P[0, 2])


class _Diffraction(CrystalBase):
    def StructureFactor(self, indices, Trat=1):
        res = 0
        G = np.dot(indices, self.crys.inv)
        G4p = np.linalg.norm(G) / (4 * np.pi)
        for at in self.crys.atoms:
            if hasattr(at, "Occupancy"):
                oc = at.Occupancy
            else:
                oc = 1
            res += np.exp(-2j * np.pi * np.dot(indices, at.Position)) * self.DebyeWallerFactor(G, at.Uani * Trat) * ScatteringFactor(at.Z, G4p, occupancy=oc)
        return res

    def DebyeWallerFactor(self, G, U, type='Uani'):
        if type == 'Uani':
            return np.exp(-G.dot(U.dot(G)) / 2)
        else:
            aG = np.linalg.norm(G) / (4 * np.pi)
            return np.exp(-8 * np.pi**2 * U * aG**2)

    def ExtinctionDistance(self, indices, beam):
        xi_inv = (beam.getWavelength() * self.StructureFactor(indices)) / (np.pi * self.crys.Volume()) * 1j
        return 1 / np.imag(xi_inv)

    def crystalFittingFunc(self, func, kspace="k"):
        function = self.__loadFunction(func)
        crys = self.crys

        def cfunc(k, *x):
            dic = {n: val for n, val in zip(crys.symbolNames(), x[1:])}
            c = crys.subs(dic)
            if kspace == "indice":
                k = np.tensordot(c.inv, k, [0, -1])
            return x[0] * function(c, k)
        parameters = [inspect.Parameter("k", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        parameters += [inspect.Parameter("scale", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        for name in crys.symbolNames():
            if crys.defaultParameters is not None:
                if name in crys.defaultParameters:
                    parameters += [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=self.crys.defaultParameters[name])]
                    continue
            parameters += [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        cfunc.__signature__ = inspect.Signature(parameters)
        return cfunc

    def __loadFunction(self, func):
        from ..DiffSim.Diffraction import structureFactors
        if func == "F2":
            def function(c, k):
                return abs(structureFactors(c, k))**2
        else:
            function = func
        return function


class _Sympy(CrystalBase):
    def subs(self, *args, **kwargs):
        return CrystalStructure(spf.subs(self.crys.cell, *args, **kwargs), spf.subs(self.crys.atoms, *args, **kwargs))

    def isSympyObject(self):
        return spf.isSympyObject(self.crys.atoms) or spf.isSympyObject(self.crys.cell)

    @property
    def free_symbols(self):
        res = [spf.free_symbols(self.crys.atoms), spf.free_symbols(self.crys.cell)]
        return set().union(*res)

    def symbolNames(self):
        cellName = ["a", "b", "c", "alpha", "beta", "gamma"]
        res = sorted([s.name for s in self.free_symbols if s.name not in cellName])
        cell = [c for c in cellName if c in [s.name for s in self.free_symbols]]
        return cell + res

    def createParametrizedCrystal(self, cell=True, atoms=True, U=False):
        conv = self.crys.createConventionalCell()
        if cell:
            cell = self.__reduceCellBySymmetry(self.crys.cell)
        else:
            cell = self.crys.cell
        if atoms:
            atm = [Atom(at.Element, self.__createPParam(i, at), U=self.__createUParam(i, U)) for i, at in enumerate(self.crys.atoms)]
        else:
            atm = copy.deepcopy(self.crys.atoms)
        c = CrystalStructure(cell, atm)
        if atoms:
            c = self.__reduceAtomsBySymmetry(conv, c, U)
        default = self.__findDefaults(c, conv)
        c.defaultParameters = default
        return c

    def defaultCrystal(self):
        return self.crys.subs(self.crys.defaultParameters)

    def __createPParam(self, i, at):
        return sp.symbols("x_" + at.Element + str(i + 1) + "," + "y_" + at.Element + str(i + 1) + "," + "z_" + at.Element + str(i + 1))

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


class CrystalStructure(object):
    def __init__(self, cell, atoms, sym=None, stress=(0, 0, 0, 0, 0, 0), energy=0):
        self._list = []
        self._register(_Lattice(self, cell))
        self._atoms = _Atoms(self, atoms, sym)
        self._register(self._atoms)
        self._register(_Symmetry(self))
        self._register(_SuperStructure(self))
        self._register(_Diffraction(self))
        self._register(_Strain(self))
        self._register(_Sympy(self))
        self._register(_CifIO(self))
        self._register(FreeEnergy(self))
        self.stress = stress
        self.energy = energy

    def _register(self, item):
        self._list.append(item)

    def __getattr__(self, key):
        for item in self._list:
            if hasattr(item, key):
                return getattr(item, key)

    @staticmethod
    def from_cif(*args, **kwargs):
        return _CifIO.from_cif(*args, **kwargs)

    @staticmethod
    def importFrom(*args, **kwargs):
        return _CifIO.importFrom(*args, **kwargs)

    @staticmethod
    def _importFromDic(*args, **kwargs):
        return _CifIO._importFromDic(*args, **kwargs)

    def density(self):
        V = self.Volume()
        mass = 0
        for at in self.atoms:
            mass += Atom.getAtomicMass(at.Element)
        return mass / V / NA * 1e27

    def __str__(self):
        return self.symmetryInfo() + self.latticeInfo() + self.atomInfo()

    def setPair(self, *args, **kwArgs):
        makePair(self, *args, **kwArgs)

    def createPCF(self, *args, **kwArgs):
        return MakePCF(self, *args, **kwArgs)

    def getFreeLatticeParameters(self):
        sym = self.crystalSystem()
        c = self.createConventionalCell()
        if "cubic" in sym:
            return [c.a]
        if "tetragonal" in sym:
            return [c.a, c.c]
        if "ortho" in sym:
            return [c.a, c.b, c.c]
        if "monoclinic" in sym:
            return [c.a, c.b, c.c, c.beta]
        if "triclinic" in sym:
            return [c.a, c.b, c.c, c.alpha, c.beta, c.gamma]
        if "hexagonal" in sym:
            return [self.a, self.c]

    def setFreeLatticeParameters(self, value):
        from DiffSim.Phonopy import Phonopy
        sym = self.crystalSystem()
        c = self.createConventionalCell()
        if "cubic" in sym:
            c.__setUnitVectors([value[0], value[0], value[0], 90, 90, 90])
        if "tetragonal" in sym:
            c.__setUnitVectors([value[0], value[0], value[1], 90, 90, 90])
        if "ortho" in sym:
            c.__setUnitVectors([value[0], value[1], value[2], 90, 90, 90])
        if "monoclinic" in sym:
            c.__setUnitVectors([value[0], value[1], value[2], 90, value[3], 90])
        if "triclinic" in sym:
            c.__setUnitVectors(*value)
        if "hexagonal" in sym:
            c.__setUnitVectors([value[0], value[0], value[1], 120, 90, 90])
        mat = Phonopy.generateTransformationMatrix(self)
        res = Phonopy.generateSupercell(c.createPrimitiveCell(), mat)
        self.__setUnitVectors(res.unit)

    @property
    def atoms(self):
        return self._atoms.getAtoms()

    @atoms.setter
    def atoms(self, value):
        self.setAtoms(value)

    def __reduce_ex__(self, proto):
        return _produceCrystal, (self._exportAsDic(), )


def _produceCrystal(d):
    return _CifIO._importFromDic(d)
