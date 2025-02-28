import sympy as sp
import spglib
import re
from .Atom import Atom
from . import sympyFuncs as spf
from .CrystalStructure import CrystalStructure
from CifFile import CifFile, ReadCif


class CrystalStructureIO(object):
    @staticmethod
    def saveAs(crys, file, ext=".cif"):
        """
        Save a CrystalStructure to a file.

        Args:
            crys (CrystalStructure): The structure to save.
            file (str): The file to save the structure to.
            ext (str, optional): The file extension. ".cif" and ".pcs" are supported. Defaults to ".cif".
        """
        if ext == ".cif":
            txt = _exportAsCif(crys, exportAll=False)
        elif ext == ".pcs":
            txt = str(_exportAsDic(crys))
        if not file.endswith(ext):
            file = file + ext
        with open(file, mode="w") as f:
            f.write(txt)

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
        if ext == ".cif":
            return _from_cif(file)
        if ext == ".pcs":
            with open(file, "r") as f:
                txt = f.read()
            return _importFromDic(eval(txt))


def _from_cif(file, index=0):
    """
    Create a CrystalStructure from a cif file.

    Args:
        file (str): path to the cif file
        index (int, optional): Which structure to read from the cif file. Defaults to 0.

    Returns:
        CrystalStructure: The structure read from the cif file
    """
    cf = ReadCif(file)
    cf = cf[cf.keys()[index]]
    cell = [
        float(cf['_cell_length_a']),
        float(cf['_cell_length_b']),
        float(cf['_cell_length_c']),
        float(cf['_cell_angle_alpha']),
        float(cf['_cell_angle_beta']),
        float(cf['_cell_angle_gamma'])
    ]
    atom_list = _parse_atoms_from_cif(cf)
    sym = _parse_symmetry_from_cif(cf)
    return CrystalStructure(cell, atom_list, sym)


def _parse_atoms_from_cif(cf):
    """
    Parse atoms from cif file content.

    Args:
        cf (CifFile): The cif file content.

    Returns:
        list: List of Atom objects.
    """
    atom_list = []
    type = '_atom_site_type_symbol' if '_atom_site_type_symbol' in cf else '_atom_site_label'
    for i in range(len(cf[type])):
        name = cf[type][i]
        x = float(cf['_atom_site_fract_x'][i])
        y = float(cf['_atom_site_fract_y'][i])
        z = float(cf['_atom_site_fract_z'][i])
        occu = float(cf['_atom_site_occupancy'][i]) if '_atom_site_occupancy' in cf else 1
        U = float(cf['_atom_site_U_iso_or_equiv'][i]) if '_atom_site_U_iso_or_equiv' in cf else 0
        if '_atom_site_aniso_U_11' in cf:
            U = [
                float(cf['_atom_site_aniso_U_11'][i]),
                float(cf['_atom_site_aniso_U_22'][i]),
                float(cf['_atom_site_aniso_U_33'][i]),
                float(cf['_atom_site_aniso_U_12'][i]),
                float(cf['_atom_site_aniso_U_13'][i]),
                float(cf['_atom_site_aniso_U_23'][i])
            ]
        atom_list.append(Atom(name, [x, y, z], U=U, occupancy=occu))
    return atom_list


def _parse_symmetry_from_cif(cf):
    """
    Parse symmetry operations from cif file content.

    Args:
        cf (CifFile): The cif file content.

    Returns:
        list: List of symmetry operations.
    """
    if '_symmetry_equiv_pos_as_xyz' in cf:
        return [__strToSym(s) for s in cf['_symmetry_equiv_pos_as_xyz']]
    elif '_space_group_symop_operation_xyz' in cf:
        return [__strToSym(s) for s in cf['_space_group_symop_operation_xyz']]
    else:
        return None


def _exportAsDic(crys):
    """
    Export a CrystalStructure as a dictionary.

    Args:
        crys (CrystalStructure): The structure to export

    Returns:
        dict: The exported structure as a dictionary
    """
    d = {}
    cell = []
    for c in crys.cell:
        if spf.isSympyObject(c):
            cell.append(str(c))
        else:
            cell.append(c)
    symbols = [str(s) for s in spf.free_symbols(crys.cell)]
    atoms = [at.saveAsDictionary() for at in crys.atoms]
    d["free_symbols"] = symbols
    d["cell"] = cell
    d["atoms"] = atoms
    return d


def _importFromDic(d):
    """
    Import a CrystalStructure from a dictionary.

    Args:
        d (dict): The dictionary containing the structure information

    Returns:
        CrystalStructure: The structure imported from the dictionary
    """
    if len(d["free_symbols"]) > 0:
        symbols = sp.symbols(",".join(d["free_symbols"]))
    cell = []
    for c in d["cell"]:
        if isinstance(c, str):
            cell.append(sp.sympify(c))
        else:
            cell.append(c)
    atoms = [Atom.loadFromDictionary(at) for at in d["atoms"]]
    return CrystalStructure(cell, atoms)


def _exportAsCif(crys, exportAll=True):
    """
    Export a CrystalStructure as a CIF file format string.

    Args:
        crys (CrystalStructure): The structure to export.
        exportAll (bool, optional): Flag indicating whether to export all atoms or only the irreducible set. Defaults to True.

    Returns:
        str: A string representation of the crystal structure in CIF format.
    """
    c = CifFile()
    c.NewBlock("crystal1")
    cf = c[c.keys()[0]]
    cf['_cell_length_a'] = crys.a
    cf['_cell_length_b'] = crys.b
    cf['_cell_length_c'] = crys.c
    cf['_cell_angle_alpha'] = crys.alpha
    cf['_cell_angle_beta'] = crys.beta
    cf['_cell_angle_gamma'] = crys.gamma
    if not exportAll:
        atoms = crys.irreducibleAtoms()
        data = spglib.get_symmetry_dataset(crys._toSpg())
        ops = spglib.get_symmetry(crys._toSpg())
        cf['_symmetry_Int_Tables_number'] = data["number"]
        cf['_symmetry_equiv_pos_as_xyz'] = [__symToStr(r, t) for r, t in zip(ops['rotations'], ops['translations'])]
        cf.CreateLoop(['_symmetry_equiv_pos_as_xyz'])
    else:
        atoms = crys.atoms
    _add_atoms_to_cif(cf, atoms)
    return str(c)


def _add_atoms_to_cif(cf, atoms):
    """
    Add atoms to CIF file content.

    Args:
        cf (CifFile): The cif file content.
        atoms (list): List of Atom objects.
    """
    newlabel = []
    elem = None
    i = 0
    for at in atoms:
        if elem != at.Element:
            elem = at.Element
            i = 0
        else:
            i += 1
        newlabel.append(elem + str(i))
    cf['_atom_site_label'] = newlabel
    cf['_atom_site_type_symbol'] = [at.Element for at in atoms]
    cf['_atom_site_fract_x'] = [at.Position[0] for at in atoms]
    cf['_atom_site_fract_y'] = [at.Position[1] for at in atoms]
    cf['_atom_site_fract_z'] = [at.Position[2] for at in atoms]
    cf.CreateLoop(['_atom_site_label', '_atom_site_type_symbol', '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z'])

    newlabel = []
    elem = None
    i = 0
    for at in atoms:
        if elem != at.Element:
            elem = at.Element
            i = 0
        else:
            i += 1
        newlabel.append(elem + str(i))
    cf['_atom_site_aniso_label'] = newlabel
    cf['_atom_site_aniso_U_11'] = [at.Uani[0, 0] for at in atoms]
    cf['_atom_site_aniso_U_22'] = [at.Uani[1, 1] for at in atoms]
    cf['_atom_site_aniso_U_33'] = [at.Uani[2, 2] for at in atoms]
    cf['_atom_site_aniso_U_12'] = [at.Uani[0, 1] for at in atoms]
    cf['_atom_site_aniso_U_13'] = [at.Uani[0, 2] for at in atoms]
    cf['_atom_site_aniso_U_23'] = [at.Uani[1, 2] for at in atoms]
    cf.CreateLoop(['_atom_site_aniso_label', '_atom_site_aniso_U_11', '_atom_site_aniso_U_22', '_atom_site_aniso_U_33', '_atom_site_aniso_U_12', '_atom_site_aniso_U_13', '_atom_site_aniso_U_23'])


def __symToStr(rotation, trans):
    """
    Convert a symmetry operation to a CIF string format.

    Takes in a rotation matrix and a translation vector and returns a string
    in the format "x,y,z" where x, y, and z are the coefficients of the symmetry operation.

    Args:
        rotation (numpy.array): 3x3 rotation matrix
        trans (numpy.array): 3 element translation vector

    Returns:
        str: The symmetry operation as a CIF string
    """
    def __xyz(v, axis):
        if v == 1:
            return "+" + axis
        elif v == -1:
            return "-" + axis
        elif v == 0:
            return ""
        else:
            print("[symToStr (CrystalStructure)] error001: ", v, axis)
            if v > 0:
                return "+" + str(v) + axis
            elif v < 0:
                return str(v) + axis
            else:
                return

    def __trans(v):
        if abs(v) < 1e-5:
            return ""
        elif abs(v - 1 / 2) < 1e-5:
            return "1/2"
        elif abs(v - 1 / 3) < 1e-5:
            return "1/3"
        elif abs(v - 2 / 3) < 1e-5:
            return "2/3"
        elif abs(v - 1 / 4) < 1e-5:
            return "1/4"
        elif abs(v - 3 / 4) < 1e-5:
            return "3/4"
        elif abs(v - 1 / 6) < 1e-5:
            return "1/6"
        elif abs(v - 5 / 6) < 1e-5:
            return "5/6"
        else:
            return str(v)
    res = ""
    for r, t in zip(rotation, trans):
        res += __trans(t)
        res += __xyz(r[0], "x")
        res += __xyz(r[1], "y")
        res += __xyz(r[2], "z")
        res += ","
    return res[:-1]


def __strToSym(str):
    """
    Convert a symmetry operation string from a CIF file to a rotation matrix and translation vector.

    Takes in a string in the format "x,y,z" and returns a tuple of two elements.
    The first element is a list of 3x3 rotation matrices, and the second element
    is a list of 3 element translation vectors.

    Args:
        str (str): The symmetry operation as a CIF string

    Returns:
        tuple: A tuple containing a list of rotation matrices and a list of translation vectors
    """
    def __xyz(s, axis):
        s_axis = re.findall(r"[+-]?" + "\d*" + axis, s)
        if len(s_axis) == 0:
            return 0
        else:
            s = s_axis[0][:-1]
            if s == "-":
                return -1
            elif s == "+":
                return 1
            elif s == "":
                return 1
            else:
                return int(s_axis[0][:-1])

    def __trans(s):
        if "1/2" in s:
            return 1 / 2
        elif "1/3" in s:
            return 1 / 3
        elif "2/3" in s:
            return 2 / 3
        elif "1/4" in s:
            return 1 / 4
        elif "3/4" in s:
            return 3 / 4
        elif "1/6" in s:
            return 1 / 6
        elif "5/6" in s:
            return 5 / 6
        else:
            return 0
    rotations = [[__xyz(s, axis) for axis in ["x", "y", "z"]] for s in str.split(",")]
    trans = [__trans(s) for s in str.split(",")]
    return rotations, trans
