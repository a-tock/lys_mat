import numpy as np
import spglib

from .Atom import Atom
from .Atoms import Atoms
from .Lattice import CartesianLattice
from .Symmetry import Symmetry
from .Pair_mesh import makePair
from .MD_Analysis import MakePCF

NA = 6.0221409e+23


def _importStrain(crys):
    from .Strain import Strain
    return Strain(crys)


def _importSympy(crys):
    from .SympyCrystalStructure import SympyCS
    return SympyCS(crys)


def _importSupercell():
    from .Supercell import createSupercell
    return createSupercell


def _importIO():
    from .CrystalStructureIO import CrystalStructureIO
    return CrystalStructureIO


class CrystalStructure(object):
    """
    A class representing a crystal structure. You can access all methods implemented in :class:`CartesianLattice <lys_mat.crystal.Lattice.CartesianLattice>`, :class:`Atoms <lys_mat.crystal.Atoms.Atoms>`, :class:`Symmetry <lys_mat.crystal.Symmetry.Symmetry>`

    Args:
        cell (list): The cell parameters of the crystal structure in the form [a, b, c, alpha, beta, gamma].
        atoms (list): A list of Atom objects representing the atoms in the crystal structure.
        sym (list, optional): The symmetry operations of the crystal structure.
        basis (list, optional): The basis vectors of the crystal structure.

    Note:
        The methods and properties of the Atoms, CartesianLattice, and Symmetry classes
        are also available as methods and properties of this class.

    Example::

        from lys_mat import Atom, CrystalStructure

        # Create gold crystal
        at1 = Atom("Au", (0, 0, 0))
        at2 = Atom("Au", (0.5, 0.5, 0))
        at3 = Atom("Au", (0, 0.5, 0.5))
        at4 = Atom("Au", (0.5, 0, 0.5))
        cell = [4.0773, 4.0773, 4.0773, 90, 90, 90]
        crys = CrystalStructure(cell, [at1, at2, at3, at4])

        print(crys)
        # Symmetry: cubic Fm-3m (No. 225), Point group: m-3m
        # a = 4.07730, b = 4.07730, c = 4.07730, alpha = 90.00000, beta = 90.00000, gamma = 90.00000
        # --- atoms (4) ---
        # 1: Au (Z = 79, Occupancy = 1) Pos = (0.00000, 0.00000, 0.00000)
        # 2: Au (Z = 79, Occupancy = 1) Pos = (0.50000, 0.50000, 0.00000)
        # 3: Au (Z = 79, Occupancy = 1) Pos = (0.00000, 0.50000, 0.50000)
        # 4: Au (Z = 79, Occupancy = 1) Pos = (0.50000, 0.00000, 0.50000)

        # Basic informations
        print(crys.volume))    # 67.782565369917
        print(crys.density())    # 19.301168279078535

        # You can access methods of CartesianLattice, Atoms, Symmetry as if they are the methods in CrystalStructure.
        print(crys.getElements())   # ['Au']: Functionarity in Atoms
        print(crys.cell)    # [4.0773, 4.0773, 4.0773, 90, 90, 90]: CatesianLattice
        print(crys.crystalSystem())   # cubic: Symmetry

    """

    def __init__(self, cell, atoms, sym=None, basis=None):
        atoms = Atoms(atoms, sym)
        lattice = CartesianLattice(cell, basis=basis)
        self._list = [atoms, lattice, Symmetry(atoms, lattice)]

    def density(self):
        """
        Calculate the density of the crystal structure in g/cm^3.

        Returns:
            float: The density of the crystal structure in g/cm^3.
        """
        mass = 0
        for at in self.atoms:
            mass += Atom.getAtomicMass(at.element)
        return mass / self.volume / NA * 1e24

    def createSupercell(self, P):
        """
        create Superstructure of original CrystalStructure determined by matrix P.
        {a',b',c'} = {a,b,c}P

        If P is array of length 3, it is used as a diagonal matrix.

        Args:
            P(3*3 array):Deformation matrix.

        Returns:
            CrystalStructure:Supercell CrystalStructure that is determined by P.
        """
        createSupercell = _importSupercell()
        return createSupercell(self, P)

    def createPrimitiveCell(self):
        """
        Calculate primitive unitcell.

        Returns:
            CrystalStructure: The primitive unitcell.
        """
        cell = self._toSpg()
        lattice, pos, numbers = spglib.find_primitive(cell)
        elems = self.getElements()
        atoms = [Atom(elems[n - 1], p) for n, p in zip(numbers, pos)]
        return CrystalStructure(lattice, atoms)

    def createConventionalCell(self, idealize=False, symprec=1e-5):
        """
        Calculate conventional unitcell.

        Returns:
            CrystalStructure: The conventional unitcell.
        """
        cell = self._toSpg()
        lattice, pos, numbers = spglib.standardize_cell(cell, to_primitive=False, no_idealize=not idealize, symprec=symprec)
        elems = self.getElements()
        atoms = [Atom(elems[n - 1], p) for n, p in zip(numbers, pos)]
        return CrystalStructure(lattice, atoms)

    def setPair(self, Elem1, Elem2, max_dist, allow_same=False):
        """
        Create pairs of atoms between two elements within a specified distance for a given crystal structure.

        Args:
            Elem1 (list of str): List of element names.
            Elem2 (list of str): List of element names.
            max_dist (float): Maximum distance between two atoms.
            allow_same (bool, optional): If True, allow same element to be paired. Default to False.
        """
        makePair(self, Elem1, Elem2, max_dist, allow_same=allow_same)

    def createPCF(self, element1, element2, sig=0.1, dim=3):
        """
        Calculate the pair correlation function (PCF) for the crystal structure.

        Args:
            element1 (list of str): The first element type for calculating the PCF. Each element should be a string of the atomic symbol.
            element2 (list of str): The second element type for calculating the PCF. Each element should be a string of the atomic symbol.
            sig (float, optional): The standard deviation for Gaussian smoothing. Default is 0.1.
            dim (int, optional): The dimensionality of the calculation (2 or 3). Default is 3.

        Returns:
            A dictionary with the calculated PCF data and corresponding axes.
        """
        return MakePCF(self, element1, element2, sig=sig, dim=dim)

    @property
    def atoms(self):
        """
        Get the list of atoms in the crystal structure.

        Returns:
            list of Atom: List of Atom objects in the crystal structure.
        """
        return self.getAtoms()

    @atoms.setter
    def atoms(self, value):
        self.setAtoms(value)

    def getAtomicPositions(self, external=True):
        """
        Get the atomic positions in the crystal structure.

        Args:
            external (bool, optional): Whether to return the positions in the
                external coordinate system. Defaults to True.

        Returns:
            numpy.ndarray: Array of atomic positions in the crystal structure.
        """

        pos = self.__getattr__("getAtomicPositions")()
        if external:
            u = np.array(self.unit.T)
            return np.array([u.dot(p) for p in pos])
        else:
            return pos

    # Strain
    def createStrainedCrystal(self, eps):
        """
        Create a strained crystal structure.

        Args:
            eps (array_like): The strain components in Voigt notation (xx,yy,zz,xy,yz,zx).

        Returns:
            CrystalStructure: The strained crystal structure.
        """
        return _importStrain(self).createStrainedCrystal(eps)

    def calculateStrain(self, ref):
        """
        Calculate the strain components in Voigt notation (xx,yy,zz,xy,yz,zx) of a crystal structure relative to a reference structure.

        Args:
            ref (CrystalStructure): The reference crystal structure.

        Returns:
            tuple: The strain components in Voigt notation (xx,yy,zz,xy,yz,zx).
        """
        return _importStrain(self).calculateStrain(ref)

    # SympyCrystalStructure
    def isSympyObject(self):
        """
        Check if the crystal structure contains any sympy objects.

        Returns:
            bool: True if either the atoms or cell of the crystal structure are sympy objects, else False.
        """
        return _importSympy(self).isSympyObject()

    @property
    def free_symbols(self):
        """
        Returns the set of free symbols in the crystal structure.

        Returns:
            set: A set containing all free symbols in the crystal structure.
        """
        return _importSympy(self).free_symbols

    def symbolNames(self):
        """
        Returns a list of symbol names used in the crystal structure, prioritizing cell parameter names.

        Returns:
            list: A list of symbol names, with cell parameter names appearing first if present.
        """
        return _importSympy(self).symbolNames()

    def subs(self, *args, **kwargs):
        """
        Substitute the given arguments and keyword arguments in the sympy objects of the current object.

        Args:
            args: see example.
            kwargs: see example.

        Returns:
            CrystalStructure: A new CrystalStructure object with the substituted sympy objects.

        Examples::

            import sympy as sp
            from lys_mat import SympyCrystalStructure

            x,y,z = sp.symbols("x,y,z")
            at = Atom("H", (x, y, z))
        """
        return _importSympy(self).subs(*args, **kwargs)

    def createParametrizedCrystal(self, cell=True, atoms=True, U=False):
        """
        Create a parametrized crystal structure with lattice parameters, atomic positions and atomic displacement parameters (if enabled) replaced by sympy symbols.

        Args:
            cell (bool): If True, lattice parameters are replaced by sympy symbols. Default to True.
            atoms (bool): If True, atomic positions are replaced by sympy symbols. Default to True.
            U (bool): If True, atomic displacement parameters are replaced by sympy symbols. Default to False.

        Returns:
            CrystalStructure: A new CrystalStructure with lattice parameters, atomic positions and atomic displacement parameters replaced by sympy symbols.

        Note:
            The resulting crystal structure can be used with the defaultCrystal method to restore the original crystal structure.
        """
        return _importSympy(self).createParametrizedCrystal(cell=cell, atoms=atoms, U=U)

    def defaultCrystal(self):
        """
        Returns a CrystalStructure with all free symbols replaced by their default values.
        This method can be executed on a CrystalStructure created by createParametrizedCrystal.

        Returns:
            CrystalStructure: A new CrystalStructure with all free symbols replaced by their default values.
        """
#        return _importSympy(self).defaultCrystal()

    # CrystalStructureIO
    def saveAs(self, file, ext=".cif"):
        """
        Save the current CrystalStructure to a file.

        Args:
            file (str): The file path where the crystal structure will be saved.
            ext (str, optional): The file extension indicating the format to save the structure in.
                                Supported extensions are ".cif" and ".pcs". Defaults to ".cif".

        """

        return _importIO().saveAs(self, file, ext=ext)

    @staticmethod
    def loadFrom(file, ext=".cif"):
        """
        Load a CrystalStructure from a file.

        Args:
            file (str): The file path to load the structure from.
            ext (str, optional): The file extension. Supported extensions are ".cif" and ".pcs". Defaults to ".cif".

        Returns:
            CrystalStructure: The structure loaded from the file.
        """
        return _importIO().loadFrom(file, ext=ext)

    def __str__(self):
        return self.symmetryInfo() + self.latticeInfo() + self.atomInfo()

    def __getattr__(self, key):
        for item in self._list:
            if hasattr(item, key):
                return getattr(item, key)

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__
