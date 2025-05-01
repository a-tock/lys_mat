import unittest
import numpy as np
import sympy as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal
from sympy.matrices.normalforms import smith_normal_form 

from lys_mat import CrystalStructure
from lys_mat.Supercell import createSupercell, _makeTranslations, _expandAtoms
from lys_mat.smith import _switchingMatrix, _multiplyingMatrix, _addMatrix, _searchMinElement, _minElementMove, _replaceByRemainder, _processForUndivisible, _smithProcess, smithNormalTransform

class TestSupercell(unittest.TestCase):
    def test_createSuperCell(self):
        crys = CrystalStructure.loadFrom("test/DataFiles/VO2_monoclinic.cif")

        P1 =  np.array([[2,0,0],[0,3,0],[0,0,4]])
        self.__compare(crys, createSupercell(crys,P1))

        P2 = np.array([[1,2,3],[1,1,3],[2,4,2]])
        self.__compare(crys, createSupercell(crys,P2))
        
        P3 = np.array([[6,3,0],[0,1,0],[-2,-1,1]]).T
        self.__compare(crys, createSupercell(crys,P3))
        
        P4 =  [2,1,1]
        self.__compare(crys, createSupercell(crys,P4))
        
        P5 =  [[2,0,0],[0,3,0],[0,0,4]]
        self.__compare(crys, createSupercell(crys,P5))

    def __compare(self, c1, c2):
        c1 = c1.createConventionalCell()
        c2 = c2.createConventionalCell()
        assert_array_almost_equal(c1.cell, c2.cell)
        self.assertEqual(len(c1.atoms), len(c2.atoms))

    def test_expandAtoms(self):
        crys = CrystalStructure.loadFrom("test/DataFiles/VO2_monoclinic.cif")
        atoms = crys.atoms

        test_array1 = np.eye(3,dtype=int)
        expand_atoms1 = _expandAtoms(atoms,test_array1)
        self.assertTrue(np.array([np.array_equal(original_atom.Position,expand_atom.Position) for original_atom, expand_atom in zip(atoms,expand_atoms1)]).all())

        test_array2 = np.array([[2,0,0],[0,3,0],[0,0,4]])
        expand_atoms2 = _expandAtoms(atoms,test_array2)
        expand_positions = np.array([atom.Position for atom in expand_atoms2])
        test_positions = []
        for atom in atoms:
            test_positions.append(atom.Position/np.array([2,3,4]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([0,0,1/4]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([0,0,1/2]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([0,0,3/4]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([0,1/3,0]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([0,1/3,1/4]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([0,1/3,1/2]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([0,1/3,3/4]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([0,2/3,0]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([0,2/3,1/4]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([0,2/3,1/2]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([0,2/3,3/4]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([1/2,0,0]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([1/2,0,1/4]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([1/2,0,1/2]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([1/2,0,3/4]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([1/2,1/3,0]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([1/2,1/3,1/4]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([1/2,1/3,1/2]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([1/2,1/3,3/4]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([1/2,2/3,0]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([1/2,2/3,1/4]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([1/2,2/3,1/2]))
            test_positions.append(atom.Position/np.array([2,3,4]) + np.array([1/2,2/3,3/4]))
        self.assertTrue(np.allclose(np.array(test_positions),expand_positions))

    def test_translation(self):
        test_array = np.array([[0,0,0],[0,0,1],[0,0,2],[0,0,3],[0,0,4],
                            [0,1,0],[0,1,1],[0,1,2],[0,1,3],[0,1,4],
                            [0,2,0],[0,2,1],[0,2,2],[0,2,3],[0,2,4],
                            [0,3,0],[0,3,1],[0,3,2],[0,3,3],[0,3,4],

                            [1,0,0],[1,0,1],[1,0,2],[1,0,3],[1,0,4],
                            [1,1,0],[1,1,1],[1,1,2],[1,1,3],[1,1,4],
                            [1,2,0],[1,2,1],[1,2,2],[1,2,3],[1,2,4],
                            [1,3,0],[1,3,1],[1,3,2],[1,3,3],[1,3,4],

                            [2,0,0],[2,0,1],[2,0,2],[2,0,3],[2,0,4],
                            [2,1,0],[2,1,1],[2,1,2],[2,1,3],[2,1,4],
                            [2,2,0],[2,2,1],[2,2,2],[2,2,3],[2,2,4],
                            [2,3,0],[2,3,1],[2,3,2],[2,3,3],[2,3,4]])
        assert_array_equal(_makeTranslations(3,4,5),test_array)


class TestSmith(unittest.TestCase):
    def test_switchingMatrix(self):
        assert_array_equal(_switchingMatrix(1,2,3),np.array([[1,0,0],[0,0,1],[0,1,0]]))
        assert_array_equal(_switchingMatrix(2,0,4),np.array([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]]))
        assert_array_equal(_switchingMatrix(1,2,3).dot(np.arange(12).reshape(3,4)),np.arange(12).reshape(3,4)[[0,2,1],:])
        assert_array_equal(np.arange(12).reshape(3,4).dot(_switchingMatrix(2,0,4)),np.arange(12).reshape(3,4)[:,[2,1,0,3]])

    def test_multiplyingMatrix(self):
        assert_array_equal(_multiplyingMatrix(1,2,3),np.array([[1,0,0],[0,2,0],[0,0,1]]))
        assert_array_equal(_multiplyingMatrix(2,-3,4),np.array([[1,0,0,0],[0,1,0,0],[0,0,-3,0],[0,0,0,1]]))
        assert_array_equal(_multiplyingMatrix(1,2,3).dot(np.arange(12).reshape(3,4)),np.array([[0,1,2,3],[8,10,12,14],[8,9,10,11]]))
        assert_array_equal(np.arange(12).reshape(3,4).dot(_multiplyingMatrix(2,-3,4)),np.array([[0,1,-6,3],[4,5,-18,7],[8,9,-30,11]]))

    def test_addMatrix(self):
        assert_array_equal(_addMatrix(1,2,3,3),np.array([[1,0,0],[0,1,3],[0,0,1]]))
        assert_array_equal(_addMatrix(2,3,-4,4),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-4],[0,0,0,1]]))
        assert_array_equal(_addMatrix(1,2,3,3).dot(np.arange(12).reshape(3,4)),np.array([[0,1,2,3],[28,32,36,40],[8,9,10,11]]))
        assert_array_equal(np.arange(12).reshape(3,4).dot(_addMatrix(2,3,-4,4)),np.array([[0,1,2,-5],[4,5,6,-17],[8,9,10,-29]]))

    def test_serchMin(self):
        test_array1 = np.array([[5,-2,3,-4],[-2,0,-3,4],[11,-2,1,0]])
        test_array2 = np.array([[0,0,0],[-1,0,4],[-3,-2,0]])
        assert_array_equal(np.array(_searchMinElement(test_array1)),np.array([2,2]))
        assert_array_equal(np.array(_searchMinElement(test_array2)),np.array([1,0]))

    def test_minMove(self):
        test_array1 = np.array([[3,4,5,8],[0,3,-4,5],[-1,3,-4,2]])
        P1,L1,R1 = _minElementMove(test_array1,0)
        assert_array_equal(P1,[[1,3,-4,2],[0,3,-4,5],[-3,4,5,8]])
        assert_array_equal(test_array1,L1.dot(P1).dot(R1))

        P2,L2,R2 = _minElementMove(test_array1,1)
        assert_array_equal(P2,[[3,8,5,4],[-1,2,-4,3],[0,5,-4,3]])
        assert_array_equal(test_array1,L2.dot(P2).dot(R2))

        P3,L3,R3 = _minElementMove(test_array1,2)
        assert_array_equal(P3,[[3,4,8,5],[0,3,5,-4],[-1,3,2,-4]])
        assert_array_equal(test_array1,L3.dot(P3).dot(R3))
 
        for i in range(10):
            test_array4 = np.random.randint(-10,10,(5,5))
            P4,L4,R4 = _minElementMove(test_array4,0)
            assert_array_equal(test_array4,L4.dot(P4).dot(R4))

    def test_replaceRemainder(self):
        test_array = np.array([[1,2,-4,5],[0,2,-4,5],[-3,4,4,8]])
        P1,L1,R1 = _replaceByRemainder(test_array,0)
        assert_array_equal(P1, [[1,0,0,0],[0,2,-4,5],[0,10,-8,23]])
        assert_array_equal(test_array, L1.dot(P1).dot(R1))
     
        P2,L2,R2 = _replaceByRemainder(test_array,1)
        assert_array_equal(P2, [[1,2,0,1],[0,2,0,1],[-3,0,12,-2]])
        assert_array_equal(test_array, L2.dot(P2).dot(R2))
        
        P3,L3,R3 = _replaceByRemainder(test_array,2)
        assert_array_equal(P3, [[1,2,-4,13],[0,2,-4,13],[-3,4,4,0]])
        assert_array_equal(test_array, L3.dot(P3).dot(R3))

    def test_processForUndivisible(self):
        test_array1 = np.array([[2,0,0,0],[0,2,-4,5],[0,4,4,8]])
        P1,L1,R1 = _processForUndivisible(test_array1,0)
        assert_array_equal(P1, [[1,2,-4,2],[-4,0,0,2],[8,4,4,0]])
        assert_array_equal(test_array1,L1.dot(P1).dot(R1))

        test_array2 = np.array([[1,0,0,0],[0,3,0,0],[0,0,4,9]])
        P2,L2,R2 = _processForUndivisible(test_array2,1)
        assert_array_equal(P2, [[1,0,0,0],[0,1,3,9],[0,-3,3,0]])
        assert_array_equal(test_array2,L2.dot(P2).dot(R2))

    def test_smithProcess(self):
        test_array = np.array([[4,-2,-3,5],[-6,6,7,-4],[3,-9,11,-1]])

        P1,L1,R1 = _smithProcess(test_array,0)
        assert_array_equal(P1, [[1,0,0,0],[0,42,-37,-18],[0,-47,52,19]])
        assert_array_equal(test_array,L1.dot(P1).dot(R1))
 
        P2,L2,R2 = _smithProcess(P1,1)
        assert_array_equal(P2, [[1,0,0,0],[0,1,0,0],[0,0,-123,25]])
        assert_array_equal(P1,L2.dot(P2).dot(R2))
        
        P3,L3,R3 = _smithProcess(P2,2)
        assert_array_equal(P3, [[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        assert_array_equal(P2,L3.dot(P3).dot(R3))

    def test_smithNormalTransform(self):
        for i in range(200):
            test_array = np.random.randint(-30,30,np.random.randint(1,10,2))
            P,L,R = smithNormalTransform(test_array)
            test_smith = np.array(smith_normal_form(sp.Matrix(test_array)))
            assert_array_equal(P,np.abs(test_smith))
            assert_array_equal(test_array,L.dot(P).dot(R))
