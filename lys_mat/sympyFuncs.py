import numpy as np
import sympy as sp

# TODO: define sympy objects
# ? An nested array but not a tensor is not a sympy object?
# ? A dictionary is not a sympy object?

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

    def _subs(x):
        if hasattr(x, "subs"):
            res = x.subs(*args, **kwargs)
            if hasattr(res, "is_number"):
                if res.is_number:
                    return float(res)
            return res
        else:
            return x
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


def isSympyObject(x):
    """
    Check if the input object is a sympy object.

    Args:
        x (object): The input object to check.

    Returns:
        bool: True if the input object is a sympy object, False otherwise.
    """
    def _isSympy(y):
        if hasattr(y, "isSympyObject"):
            return y.isSympyObject()
        else:
            return isinstance(y, sp.Basic)
#            return hasattr(y, "subs")
    if hasattr(x, "__iter__"):
        if len(x) == 0:
            return False
    return bool(np.vectorize(_isSympy)(x).any())


def free_symbols(x):
    """
    Get the free symbols in the given object.

    Args:
        x (object): The expression or array of expressions to get free symbols from.

    Returns:
        set or float: The set of free symbols in the object if `x` is an array like, or a float if `x` is a scalar. An empty set will be returned if `x` is not a sympy object.
    """

    def _get(y):
        if hasattr(y, "free_symbols"):
            return y.free_symbols
        else:
            return set()
    if not isSympyObject(x):
        return set()
    symbols = np.vectorize(_get)(x)
    if len(symbols.shape) == 0:
        return symbols.item()
    res = set().union(*np.ravel(symbols))
    return res


def einsum(string, *arrays):
    """
    Calculates the Einstein summation convention on the given arrays.

    Args:
        string (str): The string specifying the subscripts of the desired summation.
        *arrays (numpy.ndarray): The arrays to perform the summation on.

    Returns:
        numpy.ndarray: The result of the Einstein summation.

    Raises:
        TypeError: If the input arrays are not of type numpy.ndarray.

    Notes:
        - The function first tries to use the numpy.einsum function to perform the summation.
        - If numpy.einsum raises a TypeError, the function falls back to a custom implementation.
        - The custom implementation splits the input string into the input and output subscripts.
        - It then updates the input and output subscripts to remove any spaces.
        - It calculates the permutation dictionary and the number of dimensions.
        - It creates the operand axes and flags for the numpy.nditer.
        - It initializes the output operand to zero.
        - It iterates over the numpy.nditer and performs the summation.
        - Finally, it returns the result of the summation.
    """
#    """Simplified object einsum

#    does not support "..." or list input and will see "...", etc. as three times
#    an axes identifier, tries normal einsum first!
#    """

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
        out_op = sorted(all_axes)
    else:
        all_axes.update(out_op)

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
