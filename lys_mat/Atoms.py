import numpy as np
import copy

from .Atom import Atom


class Atoms(object):
    """
    Atoms class represents a list of atoms in a crystal structure.

    Args:
        atoms (list of Atom): The list of Atom objects to initialize the Atoms object with.
        sym (list of (3x3 rotation matrix, 3-length translation vector)): The symmetry operations to apply.

    Example::

        from lys_mat import Atom, Atoms
        at1 = Atom("Na", (0, 0, 0))
        at2 = Atom("Na", (0.5, 0.5, 0.5))
        atoms = Atoms([at1, at2])
        print(atoms.atomInfo())
        #--- atoms (2) ---
        #1: Na (Z = 11, Occupancy = 1) Pos = (0.00000, 0.00000, 0.00000)
        #2: Na (Z = 11, Occupancy = 1) Pos = (0.50000, 0.00000, 0.50000)


        at1 = Atom("Na", (0, 0, 0))
        at2 = Atom("Cl", (0.5, 0, 0))
        sym = []
        sym.append(([[1, 0, 0],[0, 1, 0], [0, 0, 1]], [0, 0, 0]))
        sym.append(([[1, 0, 0],[0, 1, 0], [0, 0, 1]], [0.5, 0.5, 0]))
        sym.append(([[-1, 0, 0],[0, 1, 0], [0, 0, 1]], [0, 0.5, 0.5]))
        sym.append(([[-1, 0, 0],[0, 1, 0], [0, 0, 1]], [0.5, 0, 0.5]))
        atoms = Atoms([at1, at2], sym)
        print(atoms.atomInfo())
        #--- atoms (8) ---
        #1: Cl (Z = 17, Occupancy = 1) Pos = (0.50000, 0.00000, 0.00000)
        #2: Cl (Z = 17, Occupancy = 1) Pos = (0.00000, 0.50000, 0.00000)
        #3: Cl (Z = 17, Occupancy = 1) Pos = (0.50000, 0.50000, 0.50000)
        #4: Cl (Z = 17, Occupancy = 1) Pos = (0.00000, 0.00000, 0.50000)
        #5: Na (Z = 11, Occupancy = 1) Pos = (0.00000, 0.00000, 0.00000)
        #6: Na (Z = 11, Occupancy = 1) Pos = (0.50000, 0.50000, 0.00000)
        #7: Na (Z = 11, Occupancy = 1) Pos = (0.00000, 0.50000, 0.50000)
        #8: Na (Z = 11, Occupancy = 1) Pos = (0.50000, 0.00000, 0.50000)

    """

    def __init__(self, atoms, sym=None):
        super().__init__()
        self.setAtoms(atoms, sym)

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

    def getElements(self):
        """
        Get the list of elements in the CrystalStructure.

        Return:
            list of str: list of elements in the CrystalStructure.
        """
        elements = []
        for at in self._atoms:
            if at.element not in elements:
                elements.append(at.element)
        return sorted(elements)

    def getAtomicPositions(self):
        """
        Get the atomic positions in the crystal structure.

        Returns:
            numpy.ndarray: Array of atomic positions in the crystal structure.
        """
        return np.array([at.Position for at in self._atoms])

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

    def __reorderAtoms(self):
        result = []
        for e in self.getElements():
            for at in self._atoms:
                if at.element == e:
                    result.append(at)
        self._atoms = result

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
                    if at.element == at2.element:
                        if is_same(p, at2.Position, 1e-3):
                            flg = False
                if flg:
                    result.append(Atom(at.element, p, U=at.Uani))
        return result
