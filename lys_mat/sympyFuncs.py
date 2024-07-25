import numpy as np


def subs(_x, *args, **kwargs):
    """
    Substitute symbols in an expression with given values.

    Args:
        x (Union[sympy.Expr, numpy.ndarray]): The expression or array of expressions to substitute symbols in.
        *args: Positional arguments to pass to the `subs` method of each expression in `x`.
        **kwargs: Keyword arguments to pass to the `subs` method of each expression in `x`.

    Returns:
        Union[sympy.Expr, numpy.ndarray]: The expression(s) with symbols substituted. If `x` is an array, the output will also be an array.

    Raises:
        TypeError: If the `subs` method of an expression in `x` is not callable.

    Note:
        This function uses the `np.vectorize` function to apply the `subs` method to each element of `x`.
        If `x` is an array, the output will also be an array.
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
    Check if any element in the input array is a Sympy object.
    
    Args:
        x (array): The array of elements to check.
    
    Returns:
        bool: True if any element in the array is a Sympy object, False otherwise.
    """
    def _isSympy(y):
        if hasattr(y, "isSympyObject"):
            return y.isSympyObject()
        else:
            return hasattr(y, "subs")
    if hasattr(x, "__iter__"):
        if len(x)==0:
            return False
    return np.vectorize(_isSympy)(x).any()


def free_symbols(x):
    """
    Calculate the free symbols in the input array x.

    Args:
        x (array): The array to extract free symbols from.

    Returns:
        set: The set of free symbols extracted from the input array x.
    """
    def _get(y):
        if hasattr(y, "free_symbols"):
            return y.free_symbols
        else:
            return {}
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