import warnings
import copy
import numpy as np
import sympy as sp

from . import sympyFuncs as spf


class Atom(object):
    """
    Atom class is a basic element class used for crystal structures.

    Args:
        name (str or int): The atomic symbol or atomic number of the atom.
        position (length 3 sequence of float or sympy expression, optional): The fractional coordinates of the atom in 3D space. Defaults to (0, 0, 0).
        U (float or 3 x 3 array like or length 6 sequence of float, optional): Atomic displacement parameter. If U is scalar, it assume that U is isotropic. If U is length 6 sequence, it gives U11, 22, 33, 12, 23, 31.
        occupancy (float between 0 and 1, optional): The occupancy of the atom.
        **kwargs: Additional keyword arguments to set attributes of the atom.

    Example::

        from lys_mat import Atom
        at = Atom("H", (0,0.5,0.5))
        print(at)     # H (Z = 1, Occupancy = 1) Pos = (0.00000, 0.50000, 0.50000)

        at = Atom("H", (0,0.5,0.5), U = 1, occupancy = 0.5)
        print(at)     # H (Z = 1, Occupancy = 0.5) Pos = (0.00000, 0.50000, 0.50000)

    """

    __numbers = eval('{\'Rn\': 86, \'Zr\': 40, \'W\': 74, \'I\': 53, \'At\': 85, \'B\': 5, \'U\': 92, \'In\': 49, \'N\': 7, \'Pu\': 94, \'Si\': 14, \'Am\': 95, \'Sn\': 50, \'Cf\': 98, \'Ta\': 73, \'P\': 15, \'Kr\': 36, \'C\': 6, \'Co\': 27, \'Nb\': 41, \'V\': 23, \'Sr\': 38, \'Pt\': 78, \'F\': 9, \'Gd\': 64, \'Ti\': 22, \'K\': 19, \'Eu\': 63, \'Tc\': 43, \'Rh\': 45, \'S\': 16, \'Ho\': 67, \'Th\': 90, \'Hf\': 72, \'Ir\': 77, \'Fe\': 26, \'Au\': 79, \'Tm\': 69, \'Er\': 68, \'Cd\': 48, \'Cm\': 96, \'Ru\': 44, \'Yb\': 70, \'Nd\': 60, \'Pa\': 91, \'Ga\': 31, \'Np\': 93, \'Al\': 13, \'Re\': 75, \'Pr\': 59, \'Ra\': 88, \'Lu\': 71, \'Y\': 39, \'Pm\': 61, \'Mn\': 25, \'Li\': 3, \'Se\': 34, \'Sm\': 62, \'Ce\': 58, \'Zn\': 30, \'Bi\': 83, \'Pd\': 46, \'Ni\': 28, \'Cs\': 55, \'Mg\': 12, \'Cr\': 24, \'Dy\': 66, \'Rb\': 37, \'Ca\': 20, \'Te\': 52, \'Ar\': 18, \'As\': 33, \'Cu\': 29, \'Ac\': 89, \'Cl\': 17, \'Ne\': 10, \'Mo\': 42, \'Be\': 4, \'Sc\': 21, \'Na\': 11, \'Ag\': 47, \'Xe\': 54, \'Tb\': 65, \'H\': 1, \'Ge\': 32, \'Po\': 84, \'Br\': 35, \'He\': 2, \'O\': 8, \'Tl\': 81, \'Bk\': 97, \'Fr\': 87, \'Sb\': 51, \'Pb\': 82, \'Ba\': 56, \'Hg\': 80, \'La\': 57, \'Os\': 76, \'Void\': 0}')
    __masses = {'Ac': 227.0, 'Ar': 39.948, 'Lr': 262.0, 'P': 30.973762, 'B': 10.811, 'Fm': 257.0, 'Tb': 158.92535, 'Ir': 192.217, 'Ho': 164.93032, 'Tm': 168.93421, 'Sn': 118.71, 'Pr': 140.90765, 'Xe': 131.293, 'Zr': 91.224, 'Pu': 244.0, 'He': 4.002602, 'La': 138.90547, 'Md': 258.0, 'Ru': 101.07, 'Ce': 140.116, 'Rh': 102.9055, 'Rn': 220.0, 'Fe': 55.845, 'In': 114.818, 'Ca': 40.078, 'Cd': 112.411, 'Mn': 54.938045, 'Hf': 178.49, 'K': 39.0983, 'Br': 79.904, 'Fr': 223.0, 'Bk': 247.0, 'Cm': 247.0, 'V': 50.9415, 'Ra': 226.0, 'Cu': 63.546, 'Nd': 144.242, 'Yb': 173.04, 'O': 15.9994, 'Sm': 150.36, 'Tc': 98.0, 'H': 1.00794, 'Dy': 162.5, 'Am': 243.0, 'Y': 88.90585, 'Co': 58.933195, 'Ne': 20.1797, 'Ni': 58.6934, 'Kr': 83.798, 'Th': 232.03806, 'Er': 167.259, 'C': 12.0107,
                'Np': 237.0, 'Re': 186.207, 'As': 74.9216, 'Nb': 92.90638, 'Ga': 69.723, 'Po': 210.0, 'Cs': 132.9054519, 'Gd': 157.25, 'N': 14.0067, 'Sb': 121.76, 'Se': 78.96, 'Lu': 174.967, 'Ag': 107.8682, 'At': 210.0, 'Zn': 65.409, 'Es': 252.0, 'S': 32.065, 'Li': 6.941, 'Be': 9.012182, 'Pa': 231.03588, 'Rb': 85.4678, 'W': 183.84, 'Pt': 195.084, 'Hg': 200.59, 'Ti': 47.867, 'Eu': 151.964, 'Si': 28.0855, 'Bi': 208.9804, 'Al': 26.9815386, 'Sc': 44.955912, 'Cf': 251.0, 'F': 18.9984032, 'Ge': 72.64, 'Au': 196.966569, 'Ta': 180.94788, 'U': 238.02891, 'Te': 127.6, 'Tl': 204.3833, 'Cl': 35.453, 'Mg': 24.305, 'No': 259.0, 'Sr': 87.62, 'Cr': 51.9961, 'Pb': 207.2, 'Pm': 145.0, 'Mo': 95.94, 'Os': 190.23, 'I': 126.90447, 'Pd': 106.42, 'Na': 22.98976928, 'Ba': 137.327}

    def __init__(self, name, position=(0, 0, 0), U=0, occu=None, **kwargs):
        if isinstance(name, int):
            self.element = self.getAtomicName(name)
        else:
            self.element = name
        self.position = np.array(position)
        if occu is not None:
            warnings.warn("[Atom] Initializing Ocuupancy by 'occu' is deprecated. Use 'occupancy' instead.")
            self.occupancy = occu
        self.Uani = self.__loadU(U)

        # Set all keyward arguments as attributes.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __loadU(self, U):
        if not hasattr(U, "__iter__"):
            Uani = U * np.eye(3)
        elif len(U) == 6:  # Uaniso->11,22,33,12,23,13
            Uani = np.array([[U[0], U[3], U[5]], [U[3], U[1], U[4]], [U[5], U[4], U[2]]])
        else:
            Uani = np.array(U)
        return Uani

    def __getattribute__(self, key):
        if key == "occupancy":
            if "occupancy" not in self.__dict__:
                return 1
        if key == "Occupancy":
            return self.occupancy
        if key == "Position":
            return self.position
        if key == "Element":
            return self.element
        return super().__getattribute__(key)

    def __str__(self):
        res = self.element + " (Z = " + str(self.Z)
        if hasattr(self, "occupancy"):
            res += ", Occupancy = " + str(self.Occupancy)
        res += ")"
        if spf.isSympyObject(self.position):
            res += " Pos = ({:}, {:}, {:})".format(*self.position)
        else:
            res += " Pos = ({:.5f}, {:.5f}, {:.5f})".format(*self.position)
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

    @property
    def Z(self):
        """
        Returns the atomic number of the current object.

        Returns:
            int: The atomic number of the current object.
        """
        return self.getAtomicNumber(self.element)

    # Sympy related methods
    def subs(self, *args, **kwargs):
        """
        Substitute the given arguments and keyword arguments in the sympy objects of the current object.

        Parameters:
            args: see example.
            kwargs: see example.

        Returns:
            Atom: A new Atom object with the substituted sympy objects.

        Examples::

            import sympy as sp
            from lys_mat import Atom

            x,y,z = sp.symbols("x,y,z")
            at = Atom("H", (x, y, z))

            # There are many ways to substitute.
            at.subs(x=0.2, y=0.3)           # H (Z = 1) Pos = (0.2, 0.3, z)
            at.subs(x, 0.3)                 # H (Z = 1) Pos = (0.3, y, z)
            at.subs({x:0.2, y:0.3, z:0.4})  # H (Z = 1) Pos = (0.2, 0.3, 0.4)
        """

        res = self.duplicate()
        for key, val in res.__dict__.items():
            if spf.isSympyObject(val):
                setattr(res, key, spf.subs(val, *args, **kwargs))
        return res

    def isSympyObject(self):
        """
        Check if the atom is a sympy object.

        Returns:
            bool: True if the atom is a sympy object, False otherwise.
        """

        return bool(np.array([spf.isSympyObject(val) for val in self.__dict__.values()]).any())

    @property
    def free_symbols(self):
        """
        Returns the set of free symbols in the current object.

        Returns:
            set: The set of free symbols in the current object.
        """
        res = [spf.free_symbols(val) for val in self.__dict__.values() if spf.isSympyObject(val)]
        return set().union(*res)

    # static methods
    @classmethod
    def getAtomicNumber(cls, name):
        """
        Get the atomic number of an element given its name.

        Args:
            name (str): The name of the element.

        Returns:
            int: The atomic number of the element.
        """

        return cls.__numbers[name]

    @classmethod
    def getAtomicName(cls, number):
        """
        Get the name of an element given its atomic number.

        Args:
            number (int): The atomic number of the element.

        Returns:
            str: The name of the element.
        """
        for key, val in cls.__numbers.items():
            if val == number:
                return key

    @classmethod
    def getAtomicMass(cls, name):
        """
        Get the atomic mass of an element given its name.

        Args:
            name (str): The name of the element.

        Returns:
            float: The atomic mass of the element.
        """
        return cls.__masses[name]

    def saveAsDictionary(self):  # save as string like
        """
        Generates a dictionary representation of the `Atom` object.

        Returns:
            dict: A dictionary containing the all information of the atom.
        """
        d = {"Element": self.element}
        for i, p in enumerate(self.position):
            d["Position_" + str(i)] = self._parse(p)

        for k, v in self.__dict__.items():
            if k in ["position", "element"]:
                continue
            d[k] = self._parse(v)
        return d

    @classmethod
    def loadFromDictionary(cls, d):
        """
        Load an `Atom` object from a dictionary representation, which is generated by `saveAsDictionary`.

        Args:
            d (dict): A dictionary generated by `saveAsDictionary`.

        Returns:
            Atom: An `Atom` object with the loaded data.

        Example::

            at = Atom("H", (0, 0.5, 0.5))
            d = at.saveAsDictionary()

            at_load = Atom.loadFromDictionary(d)
            type(at_load)  # Atom
        """
        e = d["Element"]
        i = 0
        pos = []
        while "Position_" + str(i) in d:
            item = cls._deparse(d["Position_" + str(i)])
            pos.append(item)
            i += 1

        kwargs = {}
        for k, v in d.items():
            if k in ["Element", "Position_0", "Position_1", "Position_2"]:
                continue
            kwargs[k] = cls._deparse(v)
        return Atom(e, pos, **kwargs)

    @classmethod
    def _parse(cls, x):
        if isinstance(x, str):
            return "[String]" + x
        elif hasattr(x, "__iter__"):
            return [cls._parse(y) for y in x]
        elif spf.isSympyObject(x):
            return str(x)
        else:
            return x

    @classmethod
    def _deparse(cls, x):
        if isinstance(x, str):
            if x.startswith("[String]"):
                return x[8:]
            else:
                return sp.simplify(x)
        elif hasattr(x, "__iter__"):
            return [cls._deparse(y) for y in x]
        else:
            return x
