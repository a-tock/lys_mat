import spglib
import numpy as np
import copy

from .Atom import Atom


class Atoms(object):
    """
    A class representing a list of atoms in a crystal structure.

    Args:
        atoms (list of Atom): The list of Atom objects to initialize the Atoms object with.
        sym (list of (3x3 rotation matrix, 3-length translation vector)): The symmetry operations to apply.
    """

    def __init__(self, atoms, sym=None):
        super().__init__()
        self.setAtoms(atoms, sym)

    def __extractAtoms(self, atoms, sym):
        """
        Extract atoms by symmetry operations.

        Args:
            atoms (list): A list of Atom objects.
            sym (list): A list of symmetry operations. Each operation is given as a tuple of a rotation matrix and a translation vector.

        Returns:
            list: A list of Atom objects, which are extracted by symmetry operations.
        """
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
        """
        Set the list of atoms in the CrystalStructure.

        Args:
            atoms (list of Atom): The list of Atom objects to set.
            sym (list of (3x3 rotation matrix, 3-length translation vector), optional):
            The symmetry operations to apply. If None, the list of atoms is set as is. Otherwise, the list of atoms is extracted by the symmetry operations.
        """
        if sym is None:
            self._atoms = copy.deepcopy(atoms)
        else:
            self._atoms = self.__extractAtoms(atoms, sym)
        self.__reorderAtoms()

    def getAtoms(self):
        """
        Get the list of atoms in the CrystalStructure.

        Return:
            list of Atom: list of atoms in the CrystalStructure.
        """
        return self._atoms

    def __reorderAtoms(self):
        result = []
        for e in self.getElements():
            for at in self._atoms:
                if at.Element == e:
                    result.append(at)
        self._atoms = result

    def getElements(self):
        """
        Get the list of elements in the CrystalStructure.

        Return:
            list of str: list of elements in the CrystalStructure.
        """
        elements = []
        for at in self._atoms:
            if at.Element not in elements:
                elements.append(at.Element)
        return sorted(elements)

    def atomInfo(self, max_atoms=-1):
        """
        Get a string representation of the atoms in the crystal structure.

        Args:
            max_atoms (int): The maximum number of atoms to include in the string representation.
                             If -1, all atoms are included.

        Returns:
            str: A string representation of the atoms in the crystal structure.
        """
        res = "--- atoms (" + str(len(self._atoms)) + ") ---"
        for i, at in enumerate(self._atoms):
            res += "\n" + str(i + 1) + ": " + str(at)
            if i == max_atoms:
                res += "..."
                break
        return res

    def getAtomicPositions(self):
        """
        Get the atomic positions in the crystal structure.

        Returns:
            numpy.ndarray: Array of atomic positions in the crystal structure.
        """
        return np.array([at.Position for at in self._atoms])
