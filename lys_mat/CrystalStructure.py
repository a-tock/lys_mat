import inspect
import os
from importlib import import_module

from .Atom import Atom
from .Atoms import Atoms
from .Lattice import CartesianLattice
from .Symmetry import Symmetry
from .Pair_mesh import makePair
from .MD_Analysis import MakePCF

NA = 6.0221409e+23


def __importStrain(crys):
    from .Strain import Strain
    return Strain(crys)


def __importSympy(crys):
    from .SympyCrystalStructure import SympyCS
    return SympyCS(crys)


def __importSuperStructure(crys):
    from .SuperStructure import SuperStructure
    return SuperStructure(crys)


def __importIO():
    from .CrystalStrucutureIO import CrystalStructureIO
    return CrystalStructureIO()


class CrystalStructure(object):
    def __init__(self, cell, atoms, basis=None, sym=None, stress=(0, 0, 0, 0, 0, 0), energy=0):
        self._list = []
        atoms = Atoms(atoms, sym)
        lattice = CartesianLattice(cell, basis=basis)
        self._register(atoms)
        self._register(lattice)
        self._register(Symmetry(atoms, lattice))
        self.stress = stress
        self.energy = energy

    def density(self):
        """
        Calculate the density of the crystal structure in g/cm^3.

        Returns:
            float: The density of the crystal structure in g/cm^3.
        """
        V = self.Volume()
        mass = 0
        for at in self.atoms:
            mass += Atom.getAtomicMass(at.Element)
        return mass / V / NA * 1e27

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
        """
        Get the list of atoms in the crystal structure.

        Returns:
            list of Atom: List of Atom objects in the crystal structure.
        """
        return self.getAtoms()

    @atoms.setter
    def atoms(self, value):
        self.setAtoms(value)

    # Strain
    def createStrainedCrystal(self, eps):
        """
        Create a strained crystal structure.

        Args:
            eps (array_like): The strain components in Voigt notation (xx,yy,zz,xy,yz,zx).

        Returns:
            CrystalStructure: The strained crystal structure.
        """
        return __importStrain(self).createStrainedCrystal(eps)

    def calculateStrain(self, ref):
        """
        Calculate the strain components in Voigt notation (xx,yy,zz,xy,yz,zx) of a crystal structure relative to a reference structure.

        Args:
            ref (CrystalStructure): The reference crystal structure.

        Returns:
            tuple: The strain components in Voigt notation (xx,yy,zz,xy,yz,zx).
        """
        return __importStrain(self).calculateStrain(ref)

    # SympyCrystalStructure
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
        return __importSympy(self).subs(*args, **kwargs)

    def isSympyObject(self):
        """
        Check if the crystal structure contains any sympy objects.

        Returns:
            bool: True if either the atoms or cell of the crystal structure are sympy objects, else False.
        """
        return __importSympy(self).isSympyObject()

    @property
    def free_symbols(self):
        """
        Returns the set of free symbols in the crystal structure.

        Returns:
            set: A set containing all free symbols in the crystal structure.
        """
        return __importSympy(self).free_symbols

    def symbolNames(self):
        """
        Returns a list of symbol names used in the crystal structure, prioritizing cell parameter names.

        Returns:
            list: A list of symbol names, with cell parameter names appearing first if present.
        """
        return __importSympy(self).symbolNames()

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
        return __importSympy(self).createParametrizedCrystal(cell=cell, atoms=atoms, U=U)

    def defaultCrystal(self):
        """
        Returns a CrystalStructure with all free symbols replaced by their default values.
        This method can be executed on a CrystalStructure created by createParametrizedCrystal.

        Returns:
            CrystalStructure: A new CrystalStructure with all free symbols replaced by their default values.
        """
        return __importSympy(self).defaultCrystal()

    # CrystalStructureIO
    def saveAs(self, file, ext=".cif"):
        return __importIO().saveAs(self, file, ext=ext)

    def __str__(self):
        return self.symmetryInfo() + self.latticeInfo() + self.atomInfo()

    def _register(self, item):
        self._list.append(item)

    def __getattr__(self, key):
        for item in self._list:
            if hasattr(item, key):
                return getattr(item, key)

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__


def _produceCrystal(d):
    return __importIO()._importFromDic(d)
