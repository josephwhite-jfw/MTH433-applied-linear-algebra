import numpy as np

def row_add(A,row1,row2,scalar):
    """Adds a scalar multiple of row1 of A to row2
       
       Parameters
       ----------
       A : np array
       row1, row2 : int
       scalar : float

       Returns
       -------
       np array after row operation
    """
    n = A.shape[0] 
    E = np.eye(n)
    E[row2,row1] = scalar
    return E @ A

def row_swap(A,row1,row2):
    """Swaps two rows of a matrix
    
       Parameters
       ---------
       A : np array
       row1,row1 : int

       Returns
       -------  
       np array after row swap 
    """
    C = A[row1,:].copy() #copy or it will change A
    A[row1,:] = A[row2,:]
    A[row2,:] = C
    return A

def row_reduce(A):
    """Puts a matrix in row echelon form
       
       Parameters
       ----------
       A : np array

       Returns
       -------
       np array that is row equivalent to A and in row echelon form
    """
    row = 0 #current row
    column = 0 #current column
    n,m = A.shape 
    while row < n and column < m:
        #if the (row,column) entry is not zero then clear everything 
        #out below it and move to the right one column and down one row
        if not np.allclose(A[row,column],0):  
            for j in range(row + 1,n):
                A = row_add(A,row,j,-A[j,column]/A[row,column])
            row = row + 1
            column = column + 1
        #if all zeros are below the current position
        #keep the same row and move over one column
        elif np.allclose(A[row + 1:,column],np.zeros((n-row-1,1))):
            column = column + 1
        #If the pivot position is non-zero but there is 
        #non-zero entry below it then perform a row swap 
        #and go back to the top of the while loop
        else:
            for jj in range(row + 1,n):
                if not np.allclose(A[jj,column],0):
                    A = row_swap(A,row,jj)
                    break
    return A