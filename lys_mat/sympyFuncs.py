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
        object: The expression(s) with symbols substituted. If the given object is not an sympy object, it will be returned as is.

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
            res = {}
            for key, value in obj.items():
                res[key] = subs(value, *args, **kwargs)
        else:
            res = []
            for value in obj:
                res.append(subs(value, *args, **kwargs))
    else:
        if hasattr(obj, "subs"):
            res = obj.subs(*args, **kwargs)
            if hasattr(res, "is_number"):
                if res.is_number:
                    return float(res)
        else:
            return obj

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
        if len(obj) == 0:
            return False
        if isinstance(obj, dict):
            return any([isSympyObject(value) for value in obj.values()])
        elif type(obj) in (list, tuple, np.ndarray):
            return any([isSympyObject(y) for y in obj])

    if hasattr(obj, "free_symbols"):
        return len(obj.free_symbols) > 0
    else:
        return isinstance(obj, sp.Basic)


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

    symbols = set()
    if hasattr(obj, "__iter__"):
        if isinstance(obj, dict):
            for value in obj.values():
                symbols.update(free_symbols(value))
        else:
            for value in obj:
                symbols.update(free_symbols(value))
    else:
        if hasattr(obj, "free_symbols"):
            return obj.free_symbols
        else:
            return set()

    return symbols


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
    in_op = s[0].split(',')
    out_op = None if len(s) == 1 else s[1].replace(' ', '')

    in_op = [axes.replace(' ', '') for axes in in_op]
    all_axes = set()

    for axes in in_op:
        all_axes.update(axes)

    if out_op is None:
        out_op = []
        for axes in sorted(all_axes):
            if s[0].count(axes) == 1:
                out_op.append(axes)
    else:
        all_axes.update(out_op)

    for array in arrays:
        if type(array) in (list, tuple, np.ndarray):
            if len(array) == 0:
                if len(out_op) == 0:
                    return 0
                else:
                    return []

    perm_dict = {_[1]: _[0] for _ in enumerate(all_axes)}

    dims = len(perm_dict)
    op_axes = []
    for axes in (in_op + list((out_op,))):
        op = [-1] * dims
        for i, ax in enumerate(axes):
            op[perm_dict[ax]] = i
        op_axes.append(op)

    op_flags = [('readonly',)] * len(in_op) + [('readwrite', 'allocate')]
    dtypes = [np.object_] * (len(in_op) + 1)  # cast all to object

    nditer = np.nditer(arrays + (None,), op_axes=op_axes, flags=[
        'buffered', 'delay_bufalloc', 'reduce_ok', 'grow_inner', 'refs_ok'], op_dtypes=dtypes, op_flags=op_flags)

    nditer.operands[-1][...] = 0
    nditer.reset()

    for vals in nditer:
        out = vals[-1]
        prod = vals[0]
        for value in vals[1:-1]:
            prod = prod * value
        out += prod

    return nditer.operands[-1]
