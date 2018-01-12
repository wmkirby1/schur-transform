"""
William M. Kirby, wmkirby1@gmail.com, 2018
Computes an efficient quantum algorithm for the Schur transform on n qubits.

Main procedures:

schuralg(n) returns a list of operators whose product is the Schur transform on n qubits.
The list has length O(n^3).
List elements have the form [t,op,b], where op is a one- or two-level rotation operator, t is a number of qubits to be tensored in above the operator, and b is a number of qubits to be tensored in below the operator.
That is, the Schur transform is given by the product of the operations identity(2**t) tensor op tensor identity(2**b) for each element [t,op,b] in schuralg(n).

schurmat(n) returns the Schur transform on n qubits in the matrix form as it will be implemented by the sequence of operations schuralg(n).

schurmatrestrict(n) returns the Schur transform on n qubits in matrix form, showing only the computational rows and columns: this will be a 2**n by 2**n matrix.

schurindices(n) returns a dict s mapping keys of the form 'j,l,m' to indices corresponding to output entries of the quantum operation implemented by schuralg(n).
Here j refers to the total spin of some spin-subspace, l is a multiplicity identifier for that spin-subspace, and m is a particular spin-projection.
Thus 'j,l,m' refers to a specific basis state in the output of the Schur transform as implemented by schuralg(n), and s['j,l,m'] returns the row encoding that basis state.

"""

import math
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Returns the lowering operator on a composite system composed of two subsystems: one with dimension d, and the other with dimension 2.
def lop(d):
    j=(d-1)/2 # Get effective total spin for d-dimensional subsystem.
    lop1=[] # Lowering operator for d-dimensional subsystem.
    lop1.append([0 for i in range(d)]) # 0th row of lowering operator is zero.
    for i in range(d-1):
        e=[0 for i in range(d)]
        e[i]=np.sqrt(j*(j+1)-(j-i)*(j-i-1)) # Build (i+1)th row in lowering operator.
        lop1.append(e)
    return np.kron(lop1,np.identity(2))+np.kron(np.identity(d),[[0,0],[1,0]]) # Add lowering operator on first subsystem to lowering operator on second subsystem.

# Returns the Clebsch-Gordan transform that combines a d-dimensional subsystem with a qubit.
# Input entries (columns) correspond to |m_1,m_2>, where m_1 and m_2 are the subsystem spin-projections.
# Output entries (rows) correspond to |j_t,m_t>, where j_t is the composite total spin and m_t is the composite spin-projection.
def cg(d):
    if d==1:
        return np.identity(2) # If the first subsystem has dimension 1, the CG transform is the identity of dimension 1*2==2.
    else:
        j=(d-1)/2 # Get effective total spin for d-dimensional subsystem.
        out=[] # Will contain output matrix.
        vec=[0 for i in range(2*d)] # Initialize current row vector for first state in spin-(d+1/2) output subsystem.
        vec[0]=1
        out.append(vec)
        lopd=lop(d) # Lowering operator for composite of d-dimensional subsystem and qubit.
        for i in range(d): # Compute the remainder of the rows corresponding to composite total spin d+1/2.
            vec=np.dot(lopd,vec)/np.linalg.norm(np.dot(lopd,vec)) # Compute new row (apply lowering operator to previous row and normalize).
            out.append(vec)
        vec=[0 for i in range(2*d)] # Initialize current row vector for first state in spin-(d-1/2) output subsystem.
        # Must be a linear combination of input states |d,-1/2> and |d-1,1/2> that is orthogonal to the linear combination of these states in the spin-(d+1/2) subspace.
        vec[1]=out[1][2]
        vec[2]=-out[1][1]
        out.append(vec)
        for i in range(d-2): # Compute the remainder of the rows corresponding to composite total spin d-1/2.
            vec=np.dot(lopd,vec)/np.linalg.norm(np.dot(lopd,vec)) # Compute new row (apply lowering operator to previous row and normalize).
            out.append(vec)
        return out

# The Schur transform is obtained recursively.
# Assume that we have obtained the Schur transform on n qubits, that is, we have a unitary map from the individual spin eigenstates |m_1,...,m_n> to the composite spin eigenstates |j,m,l>, where l is an index for the multiplicity of the spin-j subspace.

# Let D=n+1, the highest dimension of any spin-subspace.
# Then assume that in the output of the existing map, each spin-subspace (particular value of j and l) occupies the first 2j+1 states in some block of 2^(ceil(log_2(D))) entries:
# that is, assume that our output vector is divided into blocks, called J-blocks, whose size is the least power of two such that any spin-subspace can be contained in one J-block.

# Let K=ceil(D/2): this is the number of distinct values of j in the output of the existing map.
# Assume the J-blocks are organized into sets of 2^(ceil(log_2(K))), called L-blocks:
# the first K J-blocks in each L-block correspond to the K distinct spin-j subspaces, organized by decreasing j.

# Lastly, let L be the largest value of l for any j: assume that there are 2^(ceil(log_2(L))) L-blocks.

# The iteration performs the same operation on each L-block.

# Our iteration must add a qubit, map to the new spin-subspaces, and reorganize so that the output matches the description above for the new values of D, K, and L.

# We break up the iteration step into two piece: cgblocks, which maps to the new spin-subspaces, and cgrearrange, which reorganizes the output as described above.
# cgblocks and cgrearrange will describe only the operation on a single L-block: this operation will be copied over the L-blocks later.

# Adds a new qubit to n qubits that have already been mapped to composite spin eigenvectors.
def cgblocks(n):
    # Set parameters:
    d_in=[n+1-2*i for i in range(math.ceil((n+1)/2))] # Dimensions of spin-subspaces for n qubits, in decreasing order.
    d_out=[n+2-2*i for i in range(math.ceil((n+2)/2))] # Dimensions of spin-subspaces for n+1 qubits, in decreasing order.
    jblock_in=2**(math.ceil(np.log2(d_in[0]))) # Number of entries in input J-block.
    jblock_out=2**(math.ceil(np.log2(d_out[0]))) # Number of entries in output J-block.
    k_in=len(d_in) # Number of input spin-subspaces.
    k_out=len(d_out) # Number of output spin-subspaces.
    lblock_in=jblock_in*(2**(math.ceil(np.log2(k_in)))) # Number of entries in input L-block.
    lblock_out=jblock_out*(2**(math.ceil(np.log2(k_out)))) # Number of entries in output L-block.
    if not n==1:
        dim=2*lblock_out # Typically, at least one output spin-subspace will have two copies, so need two output L-blocks...
    else:
        dim=lblock_out # Except in the case n==1.
    # Build operation:
    mat=np.identity(dim) # Initialize output matrix.
    for i in range(k_in): # Iterate over input spin-subspaces.
        current_cg=cg(d_in[i]) # Get CG transform for current spin-subspace.
        for j in range(2*d_in[i]): # Iterate over entries in current CG transform.
            for k in range(2*d_in[i]):
                mat[2*i*jblock_in+j][2*i*jblock_in+k]=current_cg[j][k] # Insert current CG transform.
    return mat

# Permutes the rows so that the output matches the assumptions described above.
def cgrearrange(n):
    # Set parameters:
    d_in=[n+1-2*i for i in range(math.ceil((n+1)/2))] # Dimensions of spin-subspaces for n qubits, in decreasing order.
    d_out=[n+2-2*i for i in range(math.ceil((n+2)/2))] # Dimensions of spin-subspaces for n+1 qubits, in decreasing order.
    jblock_in=2**(math.ceil(np.log2(d_in[0]))) # Number of entries in input J-block.
    jblock_out=2**(math.ceil(np.log2(d_out[0]))) # Number of entries in output J-block.
    k_in=len(d_in) # Number of input spin-subspaces.
    k_out=len(d_out) # Number of output spin-subspaces.
    lblock_in=jblock_in*(2**(math.ceil(np.log2(k_in)))) # Number of entries in input L-block.
    lblock_out=jblock_out*(2**(math.ceil(np.log2(k_out)))) # Number of entries in output L-block.
    if not n==1:
        dim=2*lblock_out # Typically, at least one output spin-subspace will have two copies, so need two output L-blocks...
    else:
        dim=lblock_out # In the case n==1.
    # Build operation:
    pert=[] # Will contain the permutations to be implemented by the operation.
    for i in range(k_in): # Iterate over input spin-subspaces.
        for j in range(d_in[i]+1): # Iterate over rows corresponding to larger output spin-subspace.
            pert.append([2*i*jblock_in+j,i*jblock_out+j])# Put larger output spin-subspace in first output L-block.
        if d_in[i]>1: # If there is a second, smaller output spin-subspace...
            for j in range(d_in[i]-1): # Iterate over rows corresponding to smaller output spin-subspace.
                if not n==1:
                    pert.append([2*i*jblock_in+d_in[i]+1+j,lblock_out+(i+1)*jblock_out+j]) # Put smaller output spin-subspace in second output L-block, unless n==1.
                else:
                    pert.append([2*i*jblock_in+d_in[i]+1+j,(i+1)*jblock_out+j]) # If n==1, put smaller output spin-subspace also in first output L-block.
    # Generate active parts of the permutation matrix that implements the two-cycles represented by pert.
    mat=[[0 for i in range(dim)] for i in range(dim)] # Initialize output matrix.
    for i in pert: # Insert permutations into mat.
        mat[i[1]][i[0]]=1
    # Fill in the remainder of the permutation matrix.
    starts=[i[0] for i in pert]
    ends=[i[1] for i in pert]
    for i in range(dim): # Insert main diagonal 1s where possible.
        if (not i in starts) and (not i in ends):
            mat[i][i]=1
            ends.append(i)
    unused=[i for i in range(dim) if i not in ends]
    for i in range(dim): # Map remaining input entries to free output entries.
        if (not 1 in np.array(mat)[:,i]):
            mat[unused.pop()][i]=1
    return mat

# Special row swap operator: swaps rows 1 and 4 in an 8x8 matrix.
# Will be used to reconcile output of one iteration with input of the next when the L-block and J-block size changes.
spec_swap=np.array([[1,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])

# Returns a list of quantum operations to performed in order to implement the quantum Schur transform on n qubits.
# Output is a list whose elements have the form [t,op,b]:
# op is a generalized CG transform,
# t is a number of qubits to be tensored in above op,
# b is a number of qubits to be tensored in below op.
# That is, the element [t,op,b] corresponds to the quantum operation
# identity(2**t) tensor op tensor identity(2**b).
# Assumes n is an integer greater than 1, since the Schur transform on 1 qubit is the identity matrix.
def schurops(n):
    if n==2:
        return [[0,np.dot(cgrearrange(1),cgblocks(1)),0]] # The Schur transform on 2 qubits is the CG transform.
    else:
        prev=schurops(n-1) # Get Schur transform on n-1 qubits.
        for elem in prev:
            elem[2]=elem[2]+1 # Add one new qubit below each operation in previous Schur transform.
        if math.ceil(np.log2(n))==np.log2(n): # If n is a power of 2...
            for elem in prev:
                elem[0]=elem[0]+2 # Add two new qubits above each operation in previous Schur transform.
            prev.insert(0,[0,spec_swap,n-6+np.log2(len(cgrearrange(n-1)))])#spec_swap
            prev.insert(0,[n-3,np.dot(cgrearrange(n-1),cgblocks(n-1)),0])
        else:
            prev.insert(0,[n-3,np.dot(cgrearrange(n-1),cgblocks(n-1)),0])
        return prev

# Returns the decomposition of real unitary matrix u into a product of two-level "Givens" rotations and one-level phase operations.
# Output is a list of matrices, all one- or two-level rotations, whose product is u.
def givens(u):
    d=len(u) # Dimension of target matrix.
    v=u # Working matrix.
    out=[] # Will contain output list.
    # Get two-level rotations to reduce v to diagonal:
    for i in range(1,d): # Iterate over lower triangle in v.
        for j in range(i):
            if not v[i,j]==0: # If current entry in v is not already zero...
                # Build two-level rotation [[b,-c],[c,b]] to zero current entry in v:
                nm=np.sqrt(abs(v[i,j])**2+abs(v[j,j])**2)
                b=v[j,j]/nm
                c=-v[i,j]/nm
                g=np.identity(d)
                g[j,j]=b
                g[i,j]=c
                g[j,i]=-c
                g[i,i]=b
                # Add conjugate transpose of g to output:
                out.append(np.matrix(g).getH())
                # Update v:
                v=np.dot(g,v)
    # Get one-level phase rotations to reduce v to identity:
    for i in range(d):
        if not v[i][i]==1:
            g=np.identity(d)
            g[i][i]=v[i][i]
            out.append(np.matrix(g).getH())
            v=np.dot(g,v)
    return out

# MAIN PROCEDURE
# Returns a list of quantum operations to performed in order to implement the quantum Schur transform on n qubits.
# Output is a list whose elements have the form [t,op,b]:
# op is a one- or two-level rotation,
# t is a number of qubits to be tensored in above op,
# b is a number of qubits to be tensored in below op.
# That is, the element [t,op,b] corresponds to the quantum operation
# identity(2**t) tensor op tensor identity(2**b).
# Assumes n is an integer greater than 1, since the Schur transform on 1 qubit is the identity matrix.
def schuralg(n):
    inlist=schurops(n)
    outlist=[]
    while inlist:
        next=inlist.pop()
        ngivens=givens(next[1])
        while ngivens:
            outlist.insert(0,[next[0],ngivens.pop(),next[2]])
    return outlist

# Returns the Schur transform matrix in the form that will be implemented by the quantum algorithm returned by schuralg(n).
def schurmat(n):
    ops=schurops(n)
    mat=ops.pop()
    mat=np.kron(np.kron(np.identity(2**mat[0]),mat[1]),np.identity(2**mat[2]))
    while ops:
        next=ops.pop()
        next=np.kron(np.kron(np.identity(2**next[0]),next[1]),np.identity(2**next[2]))
        mat=np.dot(next,mat)
    return mat

# Returns the computational part of the Schur transform matrix: i.e. the first 2^n columns and the rows they map to.
def schurmatrestrict(n):
    mat=schurmat(n)
    mat=[mat[i][0:2**n] for i in range(len(mat))]
    i=0
    while i<len(mat):
        if not np.any(mat[i]):
            del mat[i]
        else:
            i=i+1
    return mat

# Returns a dict s mapping keys of the form 'j,l,m' to indices corresponding to output entries in the quantum operation implemented by schuralg(n).
# Here j refers to the total spin of some spin-subspace, l is a multiplicity identifier for that spin-subspace, and m is a particular spin-projection.
# Thus 'j,l,m' refers to a specific basis state in the output of the Schur transform as implemented by schuralg(n), and s['j,l,m'] returns the row encoding that basis state.
# Assumes l is an integer and j, m are half-integers to one decimal place precision.
def schurindices(n):
    out={}
    mat=schurmat(n)
    mat=[mat[i][0:2**n] for i in range(len(mat))]
    activerows=[] # Stores indices of active rows.
    mults=[0 for i in range(math.floor(n/2)+1)] # Stores current values of multiplicity identifiers.
    for i in range(len(mat)):
        if np.any(mat[i]):
            activerows.append(i)
    i=0
    d=1
    while i<len(activerows):
       while (i+d<len(activerows)) and (d<=n) and (activerows[i+d]-1==activerows[i+d-1]):
           d=d+1
       j=(d-1)/2 # Total spin of current spin-subspace.
       multindex=int((n-d+1)/2)
       for k in range(d):
           out['{:.1f},{:d},{:.1f}'.format(j,mults[multindex],j-k)]=activerows[i+k]
       i=i+d
       mults[multindex]=mults[multindex]+1
       d=1
    return out

# Return some Schur transform matrices, restricted to active rows:
"""
for n in range(2,7):
    plt.matshow(schurmat(n))
    plt.savefig("output{:d}.pdf".format(i))
"""

# Analysis check:
"""
xdata=[]
ydata=[]
max=15
for n in range(2,max+1):
    xdata.append(np.log(n))
    ydata.append(np.log(len(schuralg(n))))
xdata=np.array(xdata)
ydata=np.array(ydata)
A=np.vstack([xdata,np.ones(len(xdata))]).T
m,c=np.linalg.lstsq(A,ydata)[0]
plt.plot(xdata,ydata,'o',label='log-log sequence lengths',markersize=10)
plt.plot(xdata,m*xdata+c,'r',label='fitted line: {:0.5f}x+b'.format(m,c))
plt.axis([1,np.log(max+1),0,11])
plt.legend()
plt.savefig('analysis.png')
"""
