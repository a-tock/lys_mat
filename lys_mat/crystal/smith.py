import numpy as np


def smithNormalTransform(P):
    """
    
    calcurate Smith normal form and transform matrix.

    Args:
        P(numpy.ndarray): (m,n) array that you want to normalize
    
    Return:
        S,L,R(tuple of numpy.ndarray): tuple of result. S is Smith normal form of P. L and R satisfy P = LSR.
    """
    S = P
    L,R = np.eye(S.shape[0], dtype=int),np.eye(S.shape[1],dtype=int)
    for pivot in range(np.min(np.shape(S))):
        S,L_mul,R_mul = _smithProcess(S, pivot)
        L = L.dot(L_mul)
        R = R_mul.dot(R)
    return S,L,R


def _smithProcess(P,pivot):
    L,R = np.eye(np.shape(P)[0], dtype=int),np.eye(np.shape(P)[1],dtype=int)
    while P[pivot,pivot+1:].any() or P[pivot+1:,pivot].any():
        P, L_mul, R_mul = _minElementMove(P,pivot)
        L, R = L.dot(L_mul), R_mul.dot(R)

        P, L_mul, R_mul = _replaceByRemainder(P,pivot)
        L, R = L.dot(L_mul), R_mul.dot(R)

    if (P[pivot+1:,pivot+1:]%P[pivot,pivot]).any():
        P, L_mul, R_mul = _processForUndivisible(P,pivot)
        L, R = L.dot(L_mul), R_mul.dot(R)
        P, L_mul, R_mul = _smithProcess(P,pivot)
        L, R = L.dot(L_mul), R_mul.dot(R)
        return P,L,R
    else:
        if P[pivot,pivot]<0:
            flipMatrix = _multiplyingMatrix(pivot,-1,np.shape(P)[1])
            R = flipMatrix.dot(R)
            P = P.dot(flipMatrix)
        return P,L,R


def _minElementMove(P,pivot):
    """
    this function moves the element of P[pivot:,pivot:] who is non zero and whose absolute value is minimum to [pivot,pivot] position.
    this moving can be expressed by basis deformation matrix.
    where, P = L_mul.dot(P_new).dot(R_mul)

    Args:
        P(numpy.ndarray): Input array whose shape is (M,N).
        pivot: pivot of the process.

    Returns:
        P_new(np.array): array whose shape is (M,N). P[pivot,pivot] element is minimum in P[pivot:,pivot:] 
        L_mul(np.array): Product of row basis deformation matrix. Matrix shape is (M,M). 
        R_mul(np.array): Product of colomn basis deformation matrix. Matrix shape is (N,N).
    """
    i,j = _searchMinElement(P[pivot:,pivot:])
    i,j = i+pivot,j+pivot
    L_mul = _switchingMatrix(pivot,i,np.shape(P)[0])
    R_mul = _switchingMatrix(pivot,j,np.shape(P)[1])
    P_new = L_mul.dot(P).dot(R_mul)
    if P_new[pivot,pivot]<0:
        flipMatrix = _multiplyingMatrix(pivot,-1,np.shape(P)[1])
        R_mul = flipMatrix.dot(R_mul)
        P_new = P_new.dot(flipMatrix)
    return P_new,L_mul,R_mul


def _replaceByRemainder(P,pivot):
    """
    this function replaces elements of P[pivot, pivot+1:] and P[pivot+1:, pivot] with Remainder divided by P[pivot,pivot].
    this replacement can be expressed by product of basis deformation matrix L_mul and R_mul. 
    Where, P = L.dot(P_new).dot(R)

    Args:
        P(np.array): Input array whose shape is (M,N).
        pivot: pivot of the process.

    Returns:
        P_new(np.array): array whose shape is (M,N).
        L(np.array): Product of row basis deformation matrix. Matrix shape is (M,M). 
        R(np.array): Product of colomn basis deformation matrix. Matrix shape is (N,N).
    """
    dim = np.shape(P)
    L,L_inv = np.eye(dim[0],dtype=int), np.eye(dim[0],dtype=int)
    R,R_inv = np.eye(dim[1],dtype=int), np.eye(dim[1],dtype=int)

    quot_row = P[pivot,pivot+1:]//P[pivot,pivot]
    quot_column = P[pivot+1:,pivot]//P[pivot,pivot]
    for i, quot in enumerate(quot_column):
        L_mul, L_inv_mul = _addMatrix(pivot+1+i,pivot,quot,dim[0],inv=True)
        L, L_inv = L.dot(L_mul),L_inv_mul.dot(L_inv)
    for j, quot in enumerate(quot_row):
        R_mul, R_inv_mul = _addMatrix(pivot,pivot+1+j,quot,dim[1],inv=True)
        R, R_inv = R_mul.dot(R),R_inv.dot(R_inv_mul)
    P_new = L_inv.dot(P).dot(R_inv)
    return P_new,L,R


def _processForUndivisible(P,pivot):
    """
    this function replaces elements of P[pivot+1:, pivot+1:] with Remainder divided by P[pivot,pivot].
    this replacement can be expressed by product of basis deformation matrix L_mul and R_mul. 
    Where, P = L_mul.dot(P_new).dot(R_mul)

    Args:
        P(np.array): Input array whose shape is (M,N).
        pivot: pivot of the process.

    Returns:
        P_new(np.array): array whose shape is (M,N).
        L_mul(np.array): Product of row basis deformation matrix. Matrix shape is (M,M). 
        R_mul(np.array): Product of colomn basis deformation matrix. Matrix shape is (N,N).
    """
    dim = np.shape(P)
    i,j = _searchMinElement(P[pivot+1:,pivot+1:]%P[pivot,pivot])
    i,j = i+pivot+1, j+pivot+1
    L_add,L_inv_add = _addMatrix(i,pivot,-1,dim[0],inv=True)
    R_add, R_inv_add = _addMatrix(pivot,j, P[i,j]//P[pivot,pivot],dim[1],inv=True)# -(P[i,j]//P[pivot,pivot]) is different from -P[i,j]//P[pivot,pivot]
    P_add = L_inv_add.dot(P).dot(R_inv_add)
    P_new, L_move, R_move = _minElementMove(P_add,pivot) 
    L_mul = L_add.dot(L_move)
    R_mul = R_move.dot(R_add)
    return P_new, L_mul, R_mul


def _searchMinElement(array):
    array_flatten = array.flatten()
    argmin_flatten = np.argmin(np.where(array_flatten, np.abs(array_flatten),1e10))
    return argmin_flatten//np.shape(array)[1], argmin_flatten%np.shape(array)[1]


def _switchingMatrix(i,j,dim):
    S = np.eye(dim,dtype=int)
    indice=np.arange(dim)
    indice[i],indice[j] = indice[j],indice[i]
    return S[:,indice]


def _multiplyingMatrix(i,m,dim):
    M = np.eye(dim,dtype=int)
    M[i,i] = m
    return M


def _addMatrix(i,j,m,dim,inv=False):
    """
    A(ijm)P -> P_new[i,:] = P[i,:] + m*P[j,:]
    PA(ijm) -> P_new[:,j] = P[:,j] + m*P[:,i]
    """
    A = np.eye(dim,dtype=int)
    A[i,j] = m
    if inv:
        A_inv = np.eye(dim,dtype=int)
        A_inv[i,j] = -m
        
        return A, A_inv
    else:
        
        return A
