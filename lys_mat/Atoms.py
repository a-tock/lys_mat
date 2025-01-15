import spglib
import numpy as np
import copy

from .Atom import Atom


class Atoms(object):
    def __init__(self, atoms, sym):
        """
        Initialize an Atoms object with a list of atoms and symmetry operations.

        Args:
            atoms (list of Atom): The list of Atom objects to initialize the Atoms object with.
            sym (list of (3x3 rotation matrix, 3-length translation vector)): The symmetry operations to apply.

        This method calls the setAtoms method to set the atoms and apply the symmetry operations.
        """
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
        Set the atoms of the CrystalStructure, with optional symmetry reduction

        Args:
            atoms (list of Atom): The list of atoms to set
            sym (list of (3x3 rotation matrix, 3-length translation vector) or None):
                The list of symmetry operations to use to reduce the atoms. If None, no reduction is performed.

        Notes:
            The atoms are reordered by element after setting or reduction.
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

    def getAtomicPositions(self, external=True):
        """
        Retrieve the atomic positions in the crystal structure.

        Args:
            external (bool): If True, returns the atomic positions in the external coordinate system
                             by applying the unit transformation. If False, returns the atomic positions
                             in the internal coordinate system.

        Returns:
            numpy.ndarray: An array of atomic positions. The positions are transformed if external is True,
            otherwise they are returned as stored.
        """
        if external:
            u = np.array(self.crys.unit.T)
            return np.array([u.dot(at.Position) for at in self._atoms])
        else:
            np.array([at.Position for at in self._atoms])

    def irreducibleAtoms(self):
        """
        Get the irreducible atoms in the crystal structure.

        This method uses symmetry operations to identify and return the unique (irreducible) atoms
        in the crystal structure. It leverages the symmetry dataset obtained from spglib to find
        equivalent atoms and filters out the redundant ones.

        Returns:
            list: A list of Atom objects representing the irreducible atoms.
        """
        sym = spglib.get_symmetry_dataset(self.crys._toSpg())
        return [self._atoms[i] for i in list(set(sym["equivalent_atoms"]))]

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
