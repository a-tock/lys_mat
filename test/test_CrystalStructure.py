import unittest
import numpy as np
import sympy as sp
import pickle
from lys_mat import Atom, Atoms, CartesianLattice, Symmetry, CrystalStructure
from lys_mat import sympyFuncs as spf


class TestAtoms(unittest.TestCase):
    def test_cubic(self):
        at1 = Atom("Au", (0, 0, 0))
        at2 = Atom("Au", (0.5, 0.5, 0))
        at3 = Atom("Au", (0, 0.5, 0.5))
        at4 = Atom("Au", (0.5, 0, 0.5))
        cell = [4.07825, 4.07825, 4.07825, 90, 90, 90]
        crys = CrystalStructure(cell, [at1, at2, at3, at4])
        pick = pickle.dumps(crys)
        new_crys = pickle.loads(pick)
        print(new_crys.crystalSystem())
        print([[a.Element, a.Position] for a in new_crys.atoms])
