"""
*sympyFuncs* module gives functions for working with sympy objects and iterables of sympy objects.

Sympy objects must have "free_symbols" and "subs" attributes.
"""


import numpy as np
import sympy as sp


def subs(obj, *args, **kwargs):
    """
    Substitute the given arguments and keyword arguments in the sympy objects of the given object.

    If the given object is not an sympy object, it will be returned as is.

    Args:
        obj (object): The expression or array of expressions to substitute symbols in.
        args: see example.
        kwargs: see example.

    Returns:
        same type as the given object: The expression(s) with symbols substituted. If the given object is not an sympy object, it will be returned as is.

    Examples::

        import sympy as sp
        from lys_mat import sympyFuncs as spf

        x,y,z = sp.symbols("x,y,z")
        expr = 2*x + y
        print(spf.subs(expr, x, 0.3))           # y + 0.6
        print(spf.subs(expr, {y: 0.5})          # 2*x + 0.5
        print(spf.subs(expr, [(x, z), (y, 0.8)]))         # 2*z + 0.8

        arr = [x, y, z]
        print(spf.subs(arr, {x: 0.2, y: 0.3, z: 0.4}))       # [0.2 0.3 0.4]
        print(spf.subs(arr, [(x, 1), (y, 2)]))               # [1.0 2.0 z]

    """

    if not isSympyObject(obj):
        return obj

    if hasattr(obj, "__iter__"):
        if isinstance(obj, dict):
            res = {key: subs(value, *args, **kwargs) for key, value in obj.items()}
        else:
            res = [subs(value, *args, **kwargs) for value in obj]
            if type(obj) is np.ndarray:
                res = np.array(res)
            else:
                res = type(obj)(res)
    else:
        res = obj.subs(*args, **kwargs)
        if hasattr(res, "is_number"):
            if res.is_number:
                return float(res)

    return res


def isSympyObject(obj):
    """
    Check if the input object is a sympy object.

    Args:
        obj (object): The input object to check.

    Returns:
        bool: True if the input object is a sympy object, False otherwise.
    """
    if hasattr(obj, "__iter__"):
        if type(obj) is str or len(obj) == 0:
            return False
        if isinstance(obj, dict):
            return any([isSympyObject(value) for value in obj.values()])
        else:
            return any([isSympyObject(y) for y in obj])

    if hasattr(obj, "free_symbols"):
        return len(obj.free_symbols) > 0

    return False


def free_symbols(obj):
    """
    Get the free symbols in the given object.

    Args:
        obj (object): The expression or array of expressions to get free symbols from.

    Returns:
        set: The set of free symbols in the object. An empty set will be returned if `obj` is not a sympy object.
    """

    if not isSympyObject(obj):
        return set()

    if hasattr(obj, "__iter__"):
        symbols = set()
        if isinstance(obj, dict):
            for value in obj.values():
                symbols |= free_symbols(value)
        else:
            for value in obj:
                symbols |= free_symbols(value)
        return symbols
    else:
        return obj.free_symbols


def einsum(string, *arrays):
    """
    Calculates the Einstein summation convention on the given arrays.

    Args:
        string (str): The string specifying the subscripts of the desired summation.
        *arrays (numpy.ndarray): The arrays to perform the summation on. Elements can include sympy objects.

    Returns:
        numpy.ndarray: The result of the Einstein summation.

    Notes:
        This function tries numpy.einsum first. If it fails, it tries its own version of einsum.
        This does not support "...", list input or repeating the same axes identifier like 'ii'.
    """
    try:
        return np.einsum(string, *arrays)
    except TypeError:
        pass

    s = string.split('->')
    in_op = [axes.replace(' ', '') for axes in s[0].split(',')]
    for axes in in_op:
        if len(set(axes)) != len(axes):
            raise RuntimeError("spf.einsum does not support repeating the same axes identifier like 'ii' for general object")
    out_op = None if len(s) == 1 else s[1].replace(' ', '')

    all_axes = set("".join(in_op))
    if out_op is None:
        out_op = [axes for axes in sorted(all_axes) if s[0].count(axes) == 1]
    else:
        all_axes.update(out_op)
    all_axes = list(all_axes)

    for array in arrays:
        if type(array) in (list, tuple, np.ndarray):
            if len(array) == 0:
                return 0 if len(out_op) == 0 else []

    op_axes = []
    for axes in in_op + [out_op]:
        op = [-1] * len(all_axes)
        for i, ax in enumerate(axes):
            op[all_axes.index(ax)] = i
        op_axes.append(op)

    op_flags = [('readonly',)] * len(in_op) + [('readwrite', 'allocate')]
    dtypes = [np.object_] * (len(in_op) + 1)  # cast all to object

    nditer = np.nditer(arrays + (None,), op_axes=op_axes, flags=['buffered', 'delay_bufalloc', 'reduce_ok', 'grow_inner', 'refs_ok'], op_dtypes=dtypes, op_flags=op_flags)

    nditer.operands[-1][...] = 0
    nditer.reset()

    for vals in nditer:
        out, prod = vals[-1], vals[0]
        for value in vals[1:-1]:
            prod = prod * value
        out += prod

    return nditer.operands[-1]
