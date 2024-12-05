import numpy as np
import sympy as sp

# must have free_symbols and subs


def subs(_x, *args, **kwargs):
    """
    Substitute the given arguments and keyword arguments in the sympy objects of the given object.

    If the given object is not an sympy object, it will be returned as is.

    Args:
        _x (object): The expression or array of expressions to substitute symbols in.
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

    def _subs(_y):
        if hasattr(_y, "subs"):
            res = _y.subs(*args, **kwargs)
            if hasattr(res, "is_number"):
                if res.is_number:
                    return float(res)
            return res
        else:
            return _y
    if not isSympyObject(_x):
        return _x
    elif not hasattr(_x, "__iter__"):
        return _subs(_x)
    try:
        res = np.vectorize(_subs)(_x)
    except TypeError:
        res = np.vectorize(_subs, otypes=[object])(_x)
    if isinstance(res, np.ndarray):
        if len(res.shape) == 0:
            return res.item()
    return res


def isSympyObject(_x):
    """
    Check if the input object is a sympy object.

    Args:
        x (object): The input object to check.

    Returns:
        bool: True if the input object is a sympy object, False otherwise.
    """
    if hasattr(_x, "__iter__"):
        if len(_x) == 0:
            return False
        if isinstance(_x, dict):
            return any([isSympyObject(value) for value in _x.values()])
        elif type(_x) in (list, tuple, np.ndarray):
            return any([isSympyObject(y) for y in _x])

    if hasattr(_x, "free_symbols"):
        return len(_x.free_symbols) > 0
    else:
        return isinstance(_x, sp.Basic)


def free_symbols(_x):
    """
    Get the free symbols in the given object.

    Args:
        x (object): The expression or array of expressions to get free symbols from.

    Returns:
        set or float: The set of free symbols in the object if `x` is an array like, or a float if `x` is a scalar. An empty set will be returned if `x` is not a sympy object.
    """

    def _get(_y):
        if hasattr(_y, "free_symbols"):
            return _y.free_symbols
        else:
            return set()
    if not isSympyObject(_x):
        return set()
    symbols = np.vectorize(_get)(_x)
    if len(symbols.shape) == 0:
        return symbols.item()
    res = set().union(*np.ravel(symbols))
    return res


def einsum(string, *arrays):
    """
    Calculates the Einstein summation convention on the given arrays.

    Args:
        string (str): The string specifying the subscripts of the desired summation.
        *arrays (numpy.ndarray): The arrays to perform the summation on. Elements can include sympy objects.

    Returns:
        numpy.ndarray: The result of the Einstein summation.

    Notes:
        does not support "..." or list input and will see "...", etc. as three times
        an axes identifier, tries normal einsum first!
    """

    try:
        pass
#        return np.einsum(string, *arrays)
    except TypeError:
        pass

    if string == "jii -> ji":
        #    print("use original einsum")
        print(arrays)

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

    if string == "jii -> ji":
        print("in_op:", in_op)
        print("out_op:", out_op)
        print("all_axes:", all_axes)

    perm_dict = {_[1]: _[0] for _ in enumerate(all_axes)}

    if string == "jii -> ji":
        print("perm_dict:", perm_dict)

    dims = len(perm_dict)
    op_axes = []
    for axes in (in_op + list((out_op,))):
        #        print("axes:", axes)
        op = [-1] * dims  # dims -> len(axes)
        for i, ax in enumerate(axes):
            #            print(i, ax)
            op[perm_dict[ax]] = i
        op_axes.append(op)

    if string == "jii -> ji":
        op_axes = [[1, 2, 0], [1, 0, -1]]
        print("op_axes:", op_axes)

    op_flags = [('readonly',)] * len(in_op) + [('readwrite', 'allocate')]
    dtypes = [np.object_] * (len(in_op) + 1)  # cast all to object

    nditer = np.nditer(arrays + (None,), op_axes=op_axes, flags=[
        'buffered', 'delay_bufalloc', 'reduce_ok', 'grow_inner', 'refs_ok'], op_dtypes=dtypes, op_flags=op_flags)
#    print("nditer.operands:", nditer.operands)

    nditer.operands[-1][...] = 0
    nditer.reset()

    for vals in nditer:
        #        print("vals:", vals)
        out = vals[-1]
        prod = vals[0]
#        print("out:", out)
#        print("prod:", prod)
        for value in vals[1:-1]:
            prod = prod * value
        out += prod
#        print(nditer.operands)
#        print("out:", out)

    if string == "jii -> ji":
        print(nditer.operands[-1])
    return nditer.operands[-1]
