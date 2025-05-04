from lys import *

import numpy as np
import sympy as sp

from . import sympyFuncs as spf
from .smith import smithNormalTransform
from . import CrystalStructure, Atom


def createSupercell(crys,P):
    """
    create Supercell CrystalStructure as deformation of original CrystalStructure by deformation matrix P.

    {a',b',c'} = {a,b,c}P

    Args:
        crys(CrystalStructure):Original CrystalStructure.
        P(array of shape (3,) or (3,3)):Deformation matrix.
        
    Returns:
        CrystalStructure:Supercell CrystalStructure that is deformed by P.

    """
    P = np.array(P)
    if P.shape == (3,):
        p = np.zeros((3,3))
        for i in range(3):
            p[i,i] = P[i]
        P = p  
    new_unit = P.T.dot(crys.unit)
    new_atoms = _makeNewAtoms(crys,P)
    return CrystalStructure(new_unit, new_atoms)


def _makeNewAtoms(crys,P):
    smith,L,R = smithNormalTransform(P) # P = L.dot(smith).dot(R)
    primitive_atoms = _transform_atom(crys.atoms, L.T, crys.unit, L.T.dot(crys.unit))
    smith_atoms = _expandAtoms(primitive_atoms, smith)
    return _transform_atom(smith_atoms, R.T, smith.dot(L.T).dot(crys.unit), P.T.dot(crys.unit))


def _expandAtoms(primitive_atoms,smith):
    a,b,c=smith[0,0], smith[1,1],smith[2,2]
    translations = _makeTranslations(a,b,c)
    expand_atoms = []
    for atom in primitive_atoms:
        new_positions = (atom.Position + translations)/np.array([a,b,c])
        for new_position in new_positions: 
            expand_atoms.append(Atom(atom.Element, new_position, atom.Uani, Occupancy=atom.Occupancy))
    return expand_atoms


def _makeTranslations(a,b,c):
    a_arange = np.arange(a)
    b_arange = np.arange(b)
    c_arange = np.arange(c)
    mesh = np.array(np.meshgrid(a_arange,b_arange,c_arange,indexing="ij"))
    return np.concatenate(np.concatenate(np.transpose(mesh,(1,2,3,0))))


def _transform_atom(atoms, P_T, unit, new_unit):
    res = []
    Q = _calc_Q(P_T, unit, new_unit)
    P_Ti = np.linalg.inv(P_T)

    for at in atoms:
        new_position = _limit_range(at.Position.dot(P_Ti))
        new_U = (Q.T).dot(at.Uani).dot(Q)
        new_at = at.duplicate()
        new_at.Position = new_position
        new_at.Uani = new_U
        res.append(new_at)
    return res


def _limit_range(new_position):
    if spf.isSympyObject(new_position):
        return new_position
    else:
        mod = np.modf(new_position)[0]
        new_position = np.where(mod >= 0, mod, mod + 1) + 0
        new_position[abs(new_position-1)<1e-8]=0
        return new_position


def _calc_Q(P_T, old_unit, new_unit):
    if spf.isSympyObject(old_unit):
        old_inv = np.array([sp.Matrix(v).norm() for v in _make_inverse(old_unit)])
        new_inv = np.array([sp.Matrix(v).norm() for v in _make_inverse(new_unit)])   
        return np.diag(old_inv).dot(np.array(sp.Matrix(P_T).inv())).dot(np.diag(1/new_inv))
    else:
        old_inv = np.linalg.norm(_make_inverse(old_unit),axis=1)
        new_inv = np.linalg.norm(_make_inverse(new_unit),axis=1)
        return np.diag(old_inv).dot(np.linalg.inv(P_T)).dot(np.diag(1/new_inv))


def _make_inverse(unit):
    lib = sp if spf.isSympyObject(unit) else np
    return 2 * lib.pi * np.linalg.inv(unit) 