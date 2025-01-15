import spglib
import seekpath
import numpy as np
import random


class Symmetry(CrystalBase):
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
        crys = self.crys
        if crys.isSympyObject():
            crys = crys.subs({s: random.random() for s in crys.free_symbols})
        lattice = crys.unit
        pos = []
        num = []
        for i, e in enumerate(crys.getElements()):
            for at in crys.atoms:
                if at.Element == e:
                    pos.append(at.Position)
                    num.append(i + 1)
        return lattice, pos, num

    def crystalSystem(self):
        """
        Return the crystal system of the crystal structure.

        This method uses the symmetry information computed by spglib to determine
        the crystal system of the crystal structure.

        Returns:
            str: The crystal system of the crystal structure.
        """
        n = spglib.get_symmetry_dataset(self._toSpg())["number"]
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
            return "Symmetry: " + self.crystalSystem() + " " + data["international"] + " (No. " + str(data["number"]) + "), Point group: " + data["pointgroup"] + "\n"
        except Exception:
            return "Failed to find symmetry\n"

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
            tuple: A tuple containing lists of rotation matrices and translation
                   vectors if `pointGroup` is False.
        """
        ops = spglib.get_symmetry(self._toSpg())
        if pointGroup:
            return [r for r, t in zip(ops['rotations'], ops['translations']) if np.allclose(t, [0, 0, 0])]
        else:
            return ops['rotations'], ops['translations']
