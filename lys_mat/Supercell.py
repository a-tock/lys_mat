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
    primitive_atoms = [_transform_atom(atom, L.T, crys.unit, L.T.dot(crys.unit)) for atom in crys.atoms]
    smith_atoms = _expandAtoms(primitive_atoms, smith)
    return [_transform_atom(atom, R.T, smith.dot(L.T).dot(crys.unit), P.T.dot(crys.unit)) for atom in smith_atoms]


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


def _transform_atom(atom, P_T, unit, new_unit):
        new_position = _new_Atom_position(atom.Position, P_T)
        new_U = _new_U_maker(atom.Uani, P_T, unit, new_unit)
        return Atom(atom.Element, new_position, new_U, Occupancy=atom.Occupancy)


def _new_Atom_position(old_position, P_T):
    new_position = old_position.dot(np.linalg.inv(P_T))
    if np.array([isinstance(p, (sp.Symbol, sp.Mul, sp.Add)) for p in old_position]).any():
        return new_position
    else:
        new_position =np.where(np.modf(new_position)[0] >= 0, np.modf(new_position)[0], np.modf(new_position)[0] + 1) + 0
        return np.where(np.isclose(new_position,1), 0, new_position)


def _new_U_maker(U, P_T, old_unit,new_unit):
    if spf.isSympyObject(old_unit):
        old_inv = np.array([sp.Matrix(v).norm() for v in _make_inverse(old_unit)])
        new_inv = np.array([sp.Matrix(v).norm() for v in _make_inverse(new_unit)])   
        Q = np.diag(old_inv).dot(np.array(sp.Matrix(P_T).inv())).dot(np.diag(1/new_inv))
        return (Q.T).dot(U).dot(Q)
    else:
        old_inv = np.linalg.norm(_make_inverse(old_unit),axis=1)
        new_inv = np.linalg.norm(_make_inverse(new_unit),axis=1)
        Q = np.diag(old_inv).dot(np.linalg.inv(P_T)).dot(np.diag(1/new_inv))
        return Q.T.dot(U).dot(Q)


def _make_inverse(unit):
    lib = sp if spf.isSympyObject(unit) else np
    res = []
    a, b, c = unit
    res.append(np.cross(b, c) / np.dot(a, np.cross(b, c)) * 2 * lib.pi)
    res.append(np.cross(c, a) / np.dot(b, np.cross(c, a)) * 2 * lib.pi)
    res.append(np.cross(a, b) / np.dot(c, np.cross(a, b)) * 2 * lib.pi)
    return np.array(res)
