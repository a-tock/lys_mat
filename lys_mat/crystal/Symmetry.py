import spglib
import seekpath
import numpy as np
import random
from . import sympyFuncs as spf


class Symmetry(object):
    """
    Symmetry class is used to calculate symmetry information of a crystal structure.
    All methods in this class can be accessed from :class:`CrystalStructure <lys_mat.crystal.Crystal.CrystalStructure>`, and therefore users do not need to instantiate this class.

    Args:
        atoms (Atoms): The Atoms object to get the symmetry information from.
        lattice (CartesianLattice) : The CartesianLattice object to get the symmetry information from.

    Example::

        from lys_mat import Atom, CrystalStructure

        # Create crystal
        at1 = Atom("Na", (0, 0, 0))
        at2 = Atom("Na", (0.5, 0.5, 0.5))
        c = CrystalStructure([4.2214, 4.2214, 4.2214, 90, 90, 90], [at1, at2])

        # You can access symmetryInfo method in this class from CrystalStructure object.
        print(c.symmetryInfo())  
        #Symmetry: cubic Im-3m (No. 229), Point group: m-3m
    """

    def __init__(self, atoms, lattice):
        super().__init__()
        self._atoms = atoms
        self._lattice = lattice

    def crystalSystem(self):
        """
        Return the crystal system of the crystal structure.

        This method uses the symmetry information computed by spglib to determine
        the crystal system of the crystal structure.

        Returns:
            str: The crystal system of the crystal structure.
        """
        n = spglib.get_symmetry_dataset(self._toSpg()).number
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

    def standardPath(self):
        """
        Get the standard path of Brillouin zone.

        Returns:
            list of str: The standard path of Brillouin zone.
        """
        paths = seekpath.get_path(self._toSpg())["path"]
        res = [paths[0][0]]
        for pp in paths:
            for p in pp:
                if res[-1] != p:
                    res.append(p)
        return res

    def symmetryPoints(self):
        """
        Retrieve the symmetry points of the Brillouin zone.

        This method computes and returns the coordinates of the symmetry points
        in the Brillouin zone for the crystal structure using the seekpath library.

        Returns:
            dict: A dictionary containing the coordinates of the symmetry points.
        """
        return seekpath.get_path(self._toSpg())["point_coords"]

    def getSymmetryOperations(self, pointGroup=False):
        """
        Retrieve symmetry operations of the crystal structure.

        This method uses spglib to obtain the symmetry operations for the
        crystal structure. If the `pointGroup` argument is set to True, it
        returns only the rotations that correspond to the point group by
        filtering out any translations. Otherwise, it returns both the
        rotations and translations.

        Args:
            pointGroup (bool): If True, returns only symmetry operations
                            corresponding to the point group.

        Returns:
            list: A list of rotation matrices if `pointGroup` is True.
            tuple: A tuple containing lists of rotation matrices and translation vectors if `pointGroup` is False.
        """
        ops = spglib.get_symmetry(self._toSpg())
        if pointGroup:
            return [r for r, t in zip(ops['rotations'], ops['translations']) if np.allclose(t, [0, 0, 0])]
        else:
            return ops['rotations'], ops['translations']

    def irreducibleAtoms(self):
        """
        Retrieve the irreducible atoms of the crystal structure.

        The list contains the atoms in the crystal structure whose positions are not
        equivalent to any other atoms.

        Returns:
            list: A list of the irreducible atoms in the crystal structure.
        """
        sym = spglib.get_symmetry_dataset(self._toSpg())
        return [self._atoms.getAtoms()[i] for i in list(set(sym.equivalent_atoms))]

    def symmetryInfo(self):
        """
        Retrieve symmetry information of the crystal structure.

        This method uses spglib to obtain the symmetry dataset of the crystal
        structure and constructs a string containing the symmetry information.
        The string includes the crystal system, international symbol, space
        group number, and point group.

        Returns:
            str: A string representation of the symmetry information.

        Raises:
            Exception: If the symmetry information cannot be determined.
        """
        try:
            data = spglib.get_symmetry_dataset(self._toSpg())
            return "Symmetry: " + self.crystalSystem() + " " + data.international + " (No. " + str(data.number) + "), Point group: " + data.pointgroup + "\n"
        except Exception:
            return "Failed to find symmetry\n"

    def _toSpg(self):
        """
        Return the input for spglib.get_symmetry_dataset.

        This method calculates and returns the lattice and atomic positions
        in the format required by spglib.get_symmetry_dataset. If the crystal
        structure is represented by sympy expressions, the method substitutes
        the free symbols with random values and then computes the positions.

        Returns:
            tuple: A tuple containing the lattice and atomic positions.
        """

        lattice = self._lattice.unit
        atoms = self._atoms
        if spf.isSympyObject(lattice):
            lattice = spf.subs(lattice, {s: random.random() for s in spf.free_symbols(lattice)})
        if spf.isSympyObject(atoms):
            atoms = spf.subs(atoms, {s: random.random() for s in spf.free_symbols(atoms)})

        pos = []
        num = []
        for i, e in enumerate(atoms.getElements()):
            for at in atoms.getAtoms():
                if at.element == e:
                    pos.append(at.Position)
                    num.append(i + 1)
        return lattice, pos, num
