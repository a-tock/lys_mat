import warnings
import copy
import numpy as np
import sympy as sp

from . import sympyFuncs as spf


class Atom(object):
    """
    Atom class is a basic element class used for crystal structures.

    :attr:`name` is a string representing the element symbol. :attr:`position` is a one-dimensional tuple with three elements.

    :attr:`U` is the value of U. :attr:`Occupancy` is the float value.  # ? Uは何？

    Args:
        name (str): The atomic symbol of the atom.
        position (tuple, optional): The position of the atom in 3D space. Defaults to (0, 0, 0).
        U (int, optional): The value of U. Defaults to 0.   # ? Uは何？
        Occupancy (float, optional): The occupancy of the atom.
        **kwargs: Additional keyword arguments to set attributes of the atom.

    Example::

        >>> from lys_mat import Atom
        >>> at = Atom("H", (0,0.5,0.5))
        >>> print(at)
        H (Z = 1, Occupancy = 1) Pos = (0.00000, 0.50000, 0.50000)

        >>> at = Atom("H", (0,0.5,0.5),U=1,Occupancy=0.5)
        >>> print(at)
        H (Z = 1, Occupancy = 0.5) Pos = (0.00000, 0.50000, 0.50000)

    """

    __numbers = eval('{\'Rn\': 86, \'Zr\': 40, \'W\': 74, \'I\': 53, \'At\': 85, \'B\': 5, \'U\': 92, \'In\': 49, \'N\': 7, \'Pu\': 94, \'Si\': 14, \'Am\': 95, \'Sn\': 50, \'Cf\': 98, \'Ta\': 73, \'P\': 15, \'Kr\': 36, \'C\': 6, \'Co\': 27, \'Nb\': 41, \'V\': 23, \'Sr\': 38, \'Pt\': 78, \'F\': 9, \'Gd\': 64, \'Ti\': 22, \'K\': 19, \'Eu\': 63, \'Tc\': 43, \'Rh\': 45, \'S\': 16, \'Ho\': 67, \'Th\': 90, \'Hf\': 72, \'Ir\': 77, \'Fe\': 26, \'Au\': 79, \'Tm\': 69, \'Er\': 68, \'Cd\': 48, \'Cm\': 96, \'Ru\': 44, \'Yb\': 70, \'Nd\': 60, \'Pa\': 91, \'Ga\': 31, \'Np\': 93, \'Al\': 13, \'Re\': 75, \'Pr\': 59, \'Ra\': 88, \'Lu\': 71, \'Y\': 39, \'Pm\': 61, \'Mn\': 25, \'Li\': 3, \'Se\': 34, \'Sm\': 62, \'Ce\': 58, \'Zn\': 30, \'Bi\': 83, \'Pd\': 46, \'Ni\': 28, \'Cs\': 55, \'Mg\': 12, \'Cr\': 24, \'Dy\': 66, \'Rb\': 37, \'Ca\': 20, \'Te\': 52, \'Ar\': 18, \'As\': 33, \'Cu\': 29, \'Ac\': 89, \'Cl\': 17, \'Ne\': 10, \'Mo\': 42, \'Be\': 4, \'Sc\': 21, \'Na\': 11, \'Ag\': 47, \'Xe\': 54, \'Tb\': 65, \'H\': 1, \'Ge\': 32, \'Po\': 84, \'Br\': 35, \'He\': 2, \'O\': 8, \'Tl\': 81, \'Bk\': 97, \'Fr\': 87, \'Sb\': 51, \'Pb\': 82, \'Ba\': 56, \'Hg\': 80, \'La\': 57, \'Os\': 76, \'Void\': 0}')
    __masses = {'Ac': 227.0, 'Ar': 39.948, 'Lr': 262.0, 'P': 30.973762, 'B': 10.811, 'Fm': 257.0, 'Tb': 158.92535, 'Ir': 192.217, 'Ho': 164.93032, 'Tm': 168.93421, 'Sn': 118.71, 'Pr': 140.90765, 'Xe': 131.293, 'Zr': 91.224, 'Pu': 244.0, 'He': 4.002602, 'La': 138.90547, 'Md': 258.0, 'Ru': 101.07, 'Ce': 140.116, 'Rh': 102.9055, 'Rn': 220.0, 'Fe': 55.845, 'In': 114.818, 'Ca': 40.078, 'Cd': 112.411, 'Mn': 54.938045, 'Hf': 178.49, 'K': 39.0983, 'Br': 79.904, 'Fr': 223.0, 'Bk': 247.0, 'Cm': 247.0, 'V': 50.9415, 'Ra': 226.0, 'Cu': 63.546, 'Nd': 144.242, 'Yb': 173.04, 'O': 15.9994, 'Sm': 150.36, 'Tc': 98.0, 'H': 1.00794, 'Dy': 162.5, 'Am': 243.0, 'Y': 88.90585, 'Co': 58.933195, 'Ne': 20.1797, 'Ni': 58.6934, 'Kr': 83.798, 'Th': 232.03806, 'Er': 167.259, 'C': 12.0107,
                'Np': 237.0, 'Re': 186.207, 'As': 74.9216, 'Nb': 92.90638, 'Ga': 69.723, 'Po': 210.0, 'Cs': 132.9054519, 'Gd': 157.25, 'N': 14.0067, 'Sb': 121.76, 'Se': 78.96, 'Lu': 174.967, 'Ag': 107.8682, 'At': 210.0, 'Zn': 65.409, 'Es': 252.0, 'S': 32.065, 'Li': 6.941, 'Be': 9.012182, 'Pa': 231.03588, 'Rb': 85.4678, 'W': 183.84, 'Pt': 195.084, 'Hg': 200.59, 'Ti': 47.867, 'Eu': 151.964, 'Si': 28.0855, 'Bi': 208.9804, 'Al': 26.9815386, 'Sc': 44.955912, 'Cf': 251.0, 'F': 18.9984032, 'Ge': 72.64, 'Au': 196.966569, 'Ta': 180.94788, 'U': 238.02891, 'Te': 127.6, 'Tl': 204.3833, 'Cl': 35.453, 'Mg': 24.305, 'No': 259.0, 'Sr': 87.62, 'Cr': 51.9961, 'Pb': 207.2, 'Pm': 145.0, 'Mo': 95.94, 'Os': 190.23, 'I': 126.90447, 'Pd': 106.42, 'Na': 22.98976928, 'Ba': 137.327}

    def __init__(self, name, position=(0, 0, 0), U=0, occu=None, **kwargs):

        self.Z = Atom.__numbers[name]
        self.Element = name
        self.Position = np.array(position)
        if occu is not None:
            warnings.warn("[Atom] Initializing Ocuupancy by 'occu' is deprecated. Use 'Occupancy' instead.")
            self.Occupancy = occu
        self.__loadU(U)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattribute__(self, key):
        if key == "Occupancy":
            if "Occupancy" not in self.__dict__:
                return 1
        return super().__getattribute__(key)

    def __loadU(self, U):
        if not hasattr(U, "__iter__"):
            self.Uani = U * np.eye(3)
        elif len(U) == 6:  # Uaniso->11,22,33,12,13,23
            self.Uani = np.array([[U[0], U[3], U[4]], [U[3], U[1], U[5]], [U[4], U[5], U[2]]])
        else:
            self.Uani = np.array(U)

    def __str__(self):
        res = self.Element + " (Z = " + str(self.Z)
        if hasattr(self, "Occupancy"):
            res += ", Occupancy = " + str(self.Occupancy)
        res += ")"
        if spf.isSympyObject(self.Position):
            res += " Pos = ({:}, {:}, {:})".format(*self.Position)
        else:
            res += " Pos = ({:.5f}, {:.5f}, {:.5f})".format(*self.Position)
        if hasattr(self, "spin"):
            res += " Spin = " + str(self.spin) + "mu_B"
        return res

    def duplicate(self):
        """
        Returns a deep copy of the current object.

        Returns:
            Atom: A deep copy of the current object.
        """
        return copy.deepcopy(self)

    # Sympy related methods
    def subs(self, *args, **kwargs):
        """
        Substitute the given arguments and keyword arguments in the Sympy objects of the current object.

        Parameters:
            *args: The positional arguments to be substituted.
            **kwargs: The keyword arguments to be substituted.

        Returns:
            Atom: A new Atom object with the substituted Sympy objects.
        """

        res = self.duplicate()
        for key, val in res.__dict__.items():
            if spf.isSympyObject(val):
                setattr(res, key, spf.subs(val, *args, **kwargs))
        return res

    def isSympyObject(self):
        """
        Check if any value in the object's dictionary is a Sympy object.

        Returns:
            bool: True if any value is a Sympy object, False otherwise.
        """

        return np.array([spf.isSympyObject(val) for val in self.__dict__.values()]).any()




    @property
    def free_symbols(self):
        """
        Returns the set of free symbols in the Sympy objects of the current object.

        Returns:
            set: The set of free symbols in the Sympy objects of the current object.
        """
        res = [spf.free_symbols(val) for val in self.__dict__.values() if spf.isSympyObject(val)]
        return set().union(*res)

    # static methods
    @ classmethod
    def getAtomicNumber(cls, name):
        """
        Get the atomic number of an element given its name.

        Args:
            cls (class): The class object.
            name (str): The name of the element.

        Returns:
            int: The atomic number of the element.
        """

        return cls.__numbers[name]

    @ classmethod
    def getAtomicMass(cls, name):
        """
        Get the atomic mass of an element given its name.

        Args:
            cls (class): The class object.
            name (str): The name of the element.

        Returns:
            float: The atomic mass of the element.
        """
        return cls.__masses[name]

    def saveAsDictionary(self):
        """
        Generates a dictionary representation of the `Atom` object.

        Returns:
            dict: A dictionary containing the element name, position coordinates, and free symbols.
        """
        d = {"Element": self.Element}
        for i, p in enumerate(self.Position):
            if spf.isSympyObject(p):
                d["Position_" + str(i)] = str(p)
            else:
                d["Position_" + str(i)] = p
        d["free_symbols"] = [str(s) for s in self.free_symbols]
        return d

    @classmethod
    def loadFromDictionary(cls, d):
        """
        Load an `Atom` object from a dictionary representation.

        Args:
            cls (class): The class object.
            d (dict): A dictionary containing the element name, position coordinates, and free symbols.

        Returns:
            Atom: An `Atom` object with the loaded data.

        Raises:
            KeyError: If the dictionary is missing the "Element" key or any of the "Position_" keys.

        This class method loads an `Atom` object from a dictionary representation. It takes a dictionary `d` as input, where the "Element" key represents the name of the element, and the "Position_" keys represent the position coordinates. The method also checks if the dictionary contains any free symbols and creates a list of `sympy` symbols accordingly. It then iterates over the "Position_" keys and appends the corresponding position coordinate to the `pos` list. Finally, it returns an `Atom` object with the loaded data.

        Example:
            >>> d = {"Element": "H", "Position_0": 0, "Position_1": 0.5, "Position_2": 0.5, "free_symbols": ["x", "y"]}
            >>> Atom.loadFromDictionary(d)
            Atom(H, (0, 0.5, 0.5))
        """
        e = d["Element"]
        if len(d["free_symbols"]) > 0:
            symbols = sp.symbols(",".join(d["free_symbols"]))
        i = 0
        pos = []
        while "Position_" + str(i) in d:
            item = d["Position_" + str(i)]
            if isinstance(item, str):
                pos.append(sp.sympify(item))
            else:
                pos.append(item)
            i += 1
        return Atom(e, pos)
