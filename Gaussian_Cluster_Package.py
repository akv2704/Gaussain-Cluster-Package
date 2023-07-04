# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:51:28 2022

@author: Abhinav Verma
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 14:42:31 2022
@author: abhve
"""
import sys
import scipy as sp
import csv
import pandas as pd
import numpy as np
import strawberryfields as sf
import matplotlib.pyplot as plt
from numpy.linalg import qr
from numpy import linalg as la
####################################################Functions######################################################

def chop(expr, delta=10**-10):
    return np.ma.masked_inside(expr, -delta, delta).filled(0)


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

#X is a symplectic square matrix
def SymToUni(X):
    '''Turn a Symplectic matrix X in the xxpp format to a Unitary matrix using the rectangular decomposition.
    '''
    n=len(X[0]);
    
    A=X[0:int(0.5*n),0:int(0.5*n)];
    B=X[int(0.5*n):n,0:int(0.5*n)];
    
    U=A+1j*B;
    
    return U;

def PatternedHomodyne(X,Y):
    '''Homodyne the B spatial modes out from a covaiance matrix X, using thepattern Y.
    Note that the inputmatrix X must be in the xpxpformat and NOT in the xxpp format.'''
    p=len(X[0]); #Find the size of the matrix
    n=int(p/4); #Number of EPR pairs
    
    ################################FIXED BASIS SETTINGS############################################################
    
    PMM=[[1, 0],[0, 0]]; #projector matrix
    Target=X;
    
    #The modes to be homodyned.
    Bq_index=[];
    for j in range(2,4*n,4):
        Bq_index.append(j);
    
    j=0;
    while j<n:
        #select and construct a rotator in the SO(2) group using the pattern.
        R=[[np.cos(Y[j]), -np.sin(Y[j])],[np.sin(Y[j]), np.cos(Y[j])]]; 

        #Where are the q elements of the B spatial mode
        k=Bq_index[n-j-1];   
        #Extract the matrix (2 x 2) for homodyned mode with rotation
        #implemented.

        if (k==4*n-2*j-2):
            B=np.matmul(np.matmul(R,Target[(4*n-1)-2*j-1:(4*n)-2*j,(4*n-1)-2*j-1:(4*n)-2*j]),np.transpose(R));
            #Extract the matrix (2n-2 x 2n-2) for the rest of the modes
            A=Target[0:(4*n)-2*j-2,0:(4*n)-2*j-2];
            #extract the off diagonal block matrix and its transpose with rotation
            #implemented.
            C=np.matmul(Target[0:(4*n)-2*j-2,(4*n-1)-2*j-1:(4*n)-2*j],np.transpose(R));
            CT=np.transpose(C);
        else:
            B=np.matmul(np.matmul(R,Target[k:k+2,k:k+2]),np.transpose(R));
            #Extract the matrix (2n-2 x 2n-2) for the rest of the modes
            A=np.block([[Target[0:k,0:k],Target[0:k, k+2:(4*n)-2*j]],[Target[k+2:(4*n)-2*j,0:k], Target[k+2:(4*n)-2*j,k+2:(4*n)-2*j]]]);
            #extract the off diagonal block matrix and its transpose with rotation
            #implemented.
            C=np.matmul(np.block([[Target[0:k,k:k+2]],[Target[k+2:(4*n)-2*j,k:k+2]]]),np.transpose(R));
            CT=np.transpose(C);
               
        #subtrahend for the homodyne result
        G=np.matmul(np.matmul(C,PMM),CT);
        #The matrix for the Moore-Penrose psuedoinverse
        v=np.matmul(np.matmul(PMM,B),PMM);
        scalar=v[0,0];

        if scalar==0:
            Target=A; #Moore penrose inverse of a null matrix is null.
        else:
            Target=A-G*(1/scalar); #Use the (1,1) element of the MP pseudoinverse as in Brask, pg. 5.
        

        
        j=j+1;
        
        
    return Target
        
def BeamSplitter(M,N,d,eta):

    '''Beam Splitter is the function to get the beamsplitter transform for d modes to mix the jth mode with the
    kth mode and so on with efficiency eta. It is a 2d x 2d symplectic transform.
    
    The function returns the beam splitter transform in the xpxp format.
    Make sure M is a sensible number and is less than the total number ofmodes.
    
    %%%%%%%%%--M--%%%%%%-M+1-%%%%%%%%%%%%-N-%%%%%%%%%%-N+1-%%%%%%%%%
    %% || sqrt(eta)       0         sqrt(1-eta)        0       || %%-M-
    %% || 0           sqrt(eta)          0        sqrt(1-eta)  || %%-M+1-
    %% ||-sqrt(1-eta)     0           sqrt(eta)        0       || %%-N-
    %% || 0         -sqrt(1-eta)         0          sqrt(eta)  || %%-N+1-
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    '''
    Z=np.zeros((2*d,2*d), dtype=float);
      
    q_index=[];
    p_index=[];
    M=M-1;
    N=N-1;

    for j in range(0,2*d,2):
            q_index.append(j);
            p_index.append(j+1);
    
   
        
    for j in range(2*d):
               
                        
                 
            if (j==q_index[M]):
                
                        
                        
                        Z[j][j]=np.sqrt(eta);
                        Z[j][j+1]=0;
                        Z[j][q_index[N]]=np.sqrt(1-eta);
                        Z[j][q_index[N]+1]=0;
            
                        Z[j+1][j]=0;
                        Z[j+1][j+1]=np.sqrt(eta);
                        Z[j+1][q_index[N]]=0;
                        Z[j+1][q_index[N]+1]=np.sqrt(1-eta);
            
                        Z[q_index[N]][j]=np.sqrt(1-eta);
                        Z[q_index[N]][j+1]=0;
                        Z[q_index[N]][q_index[N]]=-np.sqrt(eta);
                        Z[q_index[N]][q_index[N]+1]=0;
            
                        Z[q_index[N]+1][j]=0;
                        Z[q_index[N]+1][j+1]=np.sqrt(1-eta);
                        Z[q_index[N]+1][q_index[N]]=0;
                        Z[q_index[N]+1][q_index[N]+1]=-np.sqrt(eta);
            elif ((j in [(q_index[M]),(p_index[M]),(q_index[N]),(p_index[N])])==False):
                        #Here, we set the idle B modes to 1 so the quadratures
                        #don't disappear.
                        
                        Z[j][j]=1;
                        
                
               
        
        
    BSm=Z;                   
    return BSm;    
        
def BS(M,d,eta):

    '''BS is the function to get the beamsplitter transform for d modes to mix the now A-1th mode with the
    B-Mth mode and so on with efficiency eta. it too is a 2d x 2d symplectic transform. The output of the
    function is BS. The number M corresponds to a delay of Tau=(M-1) modes. 
    
    The function returns the beam splitter transform in the xpxp format.
    Make sure M is a sensible number and is less than the total number ofmodes.
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% || sqrt(eta)       0         sqrt(1-eta)        0       || %%
    %% || 0           sqrt(eta)          0        sqrt(1-eta)  || %%
    %% ||-sqrt(1-eta)     0           sqrt(eta)        0       || %%
    %% || 0         -sqrt(1-eta)         0          sqrt(eta)  || %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    '''
    Z=np.zeros((2*d,2*d), dtype=float);
      
    Aq_index=[];
    Ap_index=[];
    Bq_index=[];
    Bp_index=[];
    Aq_index_extended=[];
    for j in range(0,2*d,4):
            Aq_index.append(j);
            Ap_index.append(j+1);
        
        
    for j in range(2,2*d,4):
            Bq_index.append(j);
            Bp_index.append(j+1);
        
    for j in range (0,2*d+8*(M-1),4):
            Aq_index_extended.append(j);
        
    for j in range(len(Bq_index)):
            
            k=Bq_index[j];
            N=Aq_index_extended[j+M-1];
                 
            if (N in Aq_index)*(N+1 in Ap_index)==1:
                        Z[k][k]=np.sqrt(eta);
                        Z[k][k+1]=0;
                        Z[k][N]=np.sqrt(1-eta);
                        Z[k][N+1]=0;
            
                        Z[k+1][k]=0;
                        Z[k+1][k+1]=np.sqrt(eta);
                        Z[k+1][N]=0;
                        Z[k+1][N+1]=np.sqrt(1-eta);
            
                        Z[N][k]=np.sqrt(1-eta);
                        Z[N][k+1]=0;
                        Z[N][N]=-np.sqrt(eta);
                        Z[N][N+1]=0;
            
                        Z[N+1][k]=0;
                        Z[N+1][k+1]=np.sqrt(1-eta);
                        Z[N+1][N]=0;
                        Z[N+1][N+1]=-np.sqrt(eta);
            else:
                        #Here, we set the idle B modes to 1 so the quadratures
                        #don't disappear.
                        
                        Z[k][k]=1;
                        Z[k+1][k+1]=1;
                
               
        
        
    BSm=Z;
        
    for j in range(M-1):
            index=Aq_index[j];
            BSm[index,index]=1;
            BSm[index+1,index+1]=1;
        
    return BSm;

def SingleModeSqueezing(r,n,angle):
    '''
    
    Parameters
    ----------
    r : Array
        Array of mode wise squeezing parameters.
    n : integer number
        number of modes.
    angle : Array
        Array of angles of squeezing. Defaults to 0.
    Returns
    -------
    Symplectic corresponding to single mode squeezing on n modes.
    '''
    
    if len(sys.argv)==3:
        angle=np.zeros(n)
    
    
    
    SMSV=[]
    for i in range(n):
        SMSV.append((np.cosh(r[i])*np.eye(2))-np.sinh(r[i])*np.array([[np.cos(angle[i]), np.sin(angle[i])],[np.sin(angle[i]), -np.cos(angle[i])]]))
    
    SingleModeSym=sp.linalg.block_diag(*tuple(SMSV))
    
    return np.array(SingleModeSym)

def TwoModeSqueezing(r,n,angle):
    '''
    
    Parameters
    ----------
    r : Array
        Array of mode wise squeezing parameters.
    n : integer number
        number of modes.
    angle : Array
        Array of angles of squeezing. Defaults to 0.
    Returns
    -------
    Symplectic corresponding to two mode squeezing on n modes.
    '''
    
    if len(sys.argv)==3:
        angle=np.zeros(n)
    
    
    
    TMSV=[]
    for i in range(n):
        rot=np.array([[np.cos(angle[i]), np.sin(angle[i])],[np.sin(angle[i]),-np.cos(angle[i])]])
        TMSV.append(
            
            np.block([
                
                [np.cosh(r[i])*np.eye(2), -np.sinh(r[i])*rot],
                [-np.sinh(r[i])*rot, np.cosh(r[i])*np.eye(2)]
                
                
                ])
            
            )
    
    TwoModeSym=sp.linalg.block_diag(*tuple(TMSV))
    
    return np.array(TwoModeSym)

def Rotation(n, phi):
    '''
    n=number of modes one wants to rotate
    phi=array of angles by which each modes is to be rotated
    
    Returns: a symplectic of rotation in all the modes.
    '''
    #rotation matrix of a single mode
    rot=[];
    
    for i in range(n):
            rot.append([[np.cos(phi[i]), np.sin(phi[i])],[-np.sin(phi[i]), np.cos(phi[i])]]);
    
    #build rotation symplectic for all n modes
    rot_sym=sp.linalg.block_diag(*tuple(rot))
    
    return np.array(rot_sym)

def MakeCluster(r,n,delay1,delay2,cltype):
        '''Function returns the ''4n X 4n'' covariance matrix of a 2D cluster state in the AxApBxBp (xpxp) format.
        r sets the initial squeezing level n is the number of temporal modes in each spatial mode, delay 1 sets the first delay and delay 2 sets the second delay. 
        cltype sets the cluster type to 1D (1) or 2D(2)(default).
        '''    
        TMSV=np.multiply([[np.cosh(2*r), 0, np.sinh(2*r), 0],[0, np.cosh(2*r), 0, -np.sinh(2*r)], [np.sinh(2*r), 0, np.cosh(2*r), 0],[0, -np.sinh(2*r), 0, np.cosh(2*r)]],0.5);
        b=[TMSV]*(n)
        #The matrix of all EPR pairs uncorrelated
        EPR=sp.linalg.block_diag(*b)
        #1D cluster
        M1D= np.matmul(np.matmul(BS(delay1+1,2*n,0.5),EPR),np.transpose(BS(delay1+1,2*n,0.5)));
        if cltype==1:
            return M1D;            
            
        elif(len(sys.argv)<3):
            #2D cluster
            M2D= np.matmul(np.matmul(BS(delay2+1,2*n,0.5),M1D),np.transpose(BS(delay2+1,2*n,0.5)));
            return M2D;   
        
        elif(cltype==2):
            M2D= np.matmul(np.matmul(BS(delay2+1,2*n,0.5),M1D),np.transpose(BS(delay2+1,2*n,0.5)));
            return M2D;
        
def MakeDynamicCluster(r,n,angle,delay1,delay2,cltype):
        '''Function returns the ''4n X 4n'' covariance matrix of a 2D cluster state in the AxApBxBp (xpxp) format.
        r sets the initial squeezing levels, n is the number of temporal modes in each spatial mode, delay 1 sets the first delay and delay 2 sets the second delay. 
        cltype sets the cluster type to 1D (1) or 2D(2)(default).
        '''    
        TMSV=TwoModeSqueezing(r,n,angle)
        #The matrix of all EPR pairs uncorrelated
        EPR=0.5*TMSV.dot(np.transpose(TMSV))
        #1D cluster
        M1D= np.matmul(np.matmul(BS(delay1+1,2*n,0.5),EPR),np.transpose(BS(delay1+1,2*n,0.5)));
        if cltype==1:
            return M1D;            
            
        elif(len(sys.argv)<3):
            #2D cluster
            M2D= np.matmul(np.matmul(BS(delay2+1,2*n,0.5),M1D),np.transpose(BS(delay2+1,2*n,0.5)));
            return M2D;   
        
        elif(cltype==2):
            M2D= np.matmul(np.matmul(BS(delay2+1,2*n,0.5),M1D),np.transpose(BS(delay2+1,2*n,0.5)));
            return M2D;
        
def ChoppedCluster(r,s,delay1,delay2,cltype):
    '''
    

    Parameters
    ----------
    n : integer
        chop a large cluster to a 2n mode smaller cluster.

    Returns
    -------
    covariance matrix of the chopped up cluster oftype cltype

    '''
   
    n=s+9;
    V=MakeCluster(r,n,delay1,delay2,cltype);
        
    ################################FIXED BASIS SETTINGS############################################################
    
    PMM=[[1, 0],[0, 0]]; #projector matrix
    Target=V;
    
    
    def HD_last_mode(Target):
        q=len(Target[0])
        #select and construct a rotator in the SO(2) group using the pattern.
        R=[[np.cos(0), -np.sin(0)],[np.sin(0), np.cos(0)]]; 
              
        #Extract the matrix (2 x 2) for homodyned mode with rotation
        #implemented.
    
            
        B=np.matmul(np.matmul(R,Target[q-2:q,(q-1)-1:q]),np.transpose(R));
        #Extract the matrix (2n-2 x 2n-2) for the rest of the modes
        A=Target[0:q-2,0:q-2];
        #extract the off diagonal block matrix and its transpose with rotation
        #implemented.
        C=np.matmul(Target[0:q-2,(q-1)-1:(q)],np.transpose(R));
        CT=np.transpose(C);
            
                   
        #subtrahend for the homodyne result
        G=np.matmul(np.matmul(C,PMM),CT);
        #The matrix for the Moore-Penrose psuedoinverse
        v=np.matmul(np.matmul(PMM,B),PMM);
        scalar=v[0,0];
    
        if scalar==0:
            Target=A; #Moore penrose inverse of a null matrix is null.
        else:
            Target=A-G*(1/scalar); #Use the (1,1) element of the MP pseudoinverse as in Brask, pg. 5.
            
        return Target
        
        
   
    def HD_first_mode(Target):
        #select and construct a rotator in the SO(2) group using the pattern.
        R=[[np.cos(0), -np.sin(0)],[np.sin(0), np.cos(0)]];
              
        #Extract the matrix (2 x 2) for homodyned mode with rotation
        #implemented.
    
        
        B=np.matmul(np.matmul(R,Target[0:2,0:2]),np.transpose(R));
        #Extract the matrix (2n-2 x 2n-2) for the rest of the modes
        A=Target[2:,2:];
        #extract the off diagonal block matrix and its transpose with rotation
        #implemented.
        C=np.matmul(Target[2:,0:2],np.transpose(R));
        CT=np.transpose(C);
            
                   
        #subtrahend for the homodyne result
        G=np.matmul(np.matmul(C,PMM),CT);
        #The matrix for the Moore-Penrose psuedoinverse
        v=np.matmul(np.matmul(PMM,B),PMM);
        scalar=v[0,0];
    
        if scalar==0:
            Target=A; #Moore penrose inverse of a null matrix is null.
        else:
            Target=A-G*(1/scalar); #Use the (1,1) element of the MP pseudoinverse as in Brask, pg. 5.
        
        return Target 
    for i in range(n-s):
        Target=HD_last_mode(Target);
        Target=HD_first_mode(Target);
    
    
    return Target;
def Permute_XPXP_to_XXPP(M):
        '''The permute function changes a (2d x 2d) d-mode matrix M from [q,p,q,p,q,p] format to [q,q,q,p,p,p].
        '''
        d=len(M[0])
        T1=np.zeros((d,d), dtype=float);
        T2=np.zeros((d,d), dtype=float);
        for i in range(1,d+1):
                for j in range(1,d+1):
                    if j==2*(i)-1:
                        T1[i-1][j-1]=1;
                    else:
                        T1[i-1][j-1]=0;
                  
            
            
        for i in range(1,d+1):
                for j in range(1,d+1):
                    if j+d==2*(i):
                        T2[i-1][j-1]=1;
                    else:
                        T2[i-1][j-1]=0;
                    
            
            
        T=T1+T2;
        
        G=np.matmul(np.matmul((T),M),np.transpose(T));
        return G;

def Permute_XXPP_to_XPXP(M):
        '''The permute function changes a (2d x 2d) d-mode matrix M from [q,p,q,p,q,p] format to [q,q,q,p,p,p].
        '''    
        d=len(M[0])
        T1=np.zeros((d,d), dtype=float);
        T2=np.zeros((d,d), dtype=float);
        for i in range(1,d+1):
                for j in range(1,d+1):
                    if j==2*(i)-1:
                        T1[i-1][j-1]=1;
                    else:
                        T1[i-1][j-1]=0;
                  
            
            
        for i in range(1,d+1):
                for j in range(1,d+1):
                    if j+d==2*(i):
                        T2[i-1][j-1]=1;
                    else:
                        T2[i-1][j-1]=0;
                    
            
            
        T=T1+T2;
        
        G=np.matmul(np.matmul(np.transpose(T),M),(T));
        return G;
    
def covariance_to_unitary(V):
    '''
    
    Parameters
    ----------
    V : 2D array
        A positive definite matrix or covariance matrix in XPXP format.
    Returns
    -------
    complex 2D array
        The Unitary that transforms vacua into this covariance matrix.
    '''
    (S_Eigen,S)=sf.decompositions.williamson(V,tol=1e-13)
    (K,Sr,L)=sf.decompositions.bloch_messiah(S,tol=1e-6)
    A=SymToUni(L);
    
    return A;

def Generate_haar_unitary(N):
    """Generate a Haar-random matrix using the QR decomposition."""
    # Step 1
    A, B = np.random.normal(size=(N, N)), np.random.normal(size=(N, N))
    Z = A + 1j * B

    # Step 2
    Q, R = qr(Z)

    # Step 3
    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])

    # Step 4
    return np.dot(Q, Lambda)

def Generate_random_symplectic(N,passive,mu,sigma):
    '''
    Generate a haar random symplectioc matrix for N modes.
    passive=True gives a passive matrix
    passive=False gives an active matrix with squeeezing sampled from a uniform distribution
    '''
    
    U=Generate_haar_unitary(N);
    O = np.block([[U.real, -U.imag], [U.imag, U.real]])
    
    if passive==True:
        return O
    
    U=Generate_haar_unitary(N);
    P= np.block([[U.real, -U.imag], [U.imag, U.real]])
    
    r = np.random.normal(mu,sigma,N)
    Sq = np.diag(np.concatenate([np.exp(-r), np.exp(r)]))
    
    return np.matmul(O,np.matmul(Sq,P))

def Plot_graph(Z):
    '''

    Parameters
    ----------
    Z : 2D complex matrix
        Adjacency matrix.

    Returns
    -------
    plots a graph given the adjacency matrix.

    '''
    import networkx as nx 
    n=len(Z[0]);
    G = nx.DiGraph()
    for i in range(n): 
        for j in range(n): 
            if np.abs(Z[i][j]) != 0: 
                G.add_edge(i,j) 
      
    
    nx.draw( G, with_labels=True, font_weight='bold',node_color='b', edge_color='k' )
    #edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G)) 
    plt.show()
    

def HD_last_mode(Target,ang):
        PMM=np.array([[1, 0],[0, 0]]); #projector matrix
        q=len(Target[0])
        #select and construct a rotator in the SO(2) group using the pattern.
        R=np.array([[np.cos(ang), -np.sin(ang)],[np.sin(ang), np.cos(ang)]]); 
              
        #Extract the matrix (2 x 2) for homodyned mode with rotation
        #implemented.
    
            
        B=np.matmul(np.matmul(R,Target[q-2:q,(q-1)-1:q]),np.transpose(R));
        #Extract the matrix (2n-2 x 2n-2) for the rest of the modes
        A=Target[0:q-2,0:q-2];
        #extract the off diagonal block matrix and its transpose with rotation
        #implemented.
        C=np.matmul(Target[0:q-2,(q-1)-1:(q)],np.transpose(R));
        CT=np.transpose(C);
            
                   
        #subtrahend for the homodyne result
        G=np.matmul(np.matmul(C,PMM),CT);
        #The matrix for the Moore-Penrose psuedoinverse
        v=np.matmul(np.matmul(PMM,B),PMM);
        scalar=v[0,0];
    
        if scalar==0:
            Target=A; #Moore penrose inverse of a null matrix is null.
        else:
            Target=A-G*(1/scalar); #Use the (1,1) element of the MP pseudoinverse as in Brask, pg. 5.
            
        return Target
        
        
   
def HD_first_mode(Target,ang):
        PMM=np.array([[1, 0],[0, 0]]);
        #select and construct a rotator in the SO(2) group using the pattern.
        R=np.array([[np.cos(ang), -np.sin(ang)],[np.sin(ang), np.cos(ang)]]);
              
        #Extract the matrix (2 x 2) for homodyned mode with rotation
        #implemented.
    
        
        B=np.matmul(np.matmul(R,Target[0:2,0:2]),np.transpose(R));
        #Extract the matrix (2n-2 x 2n-2) for the rest of the modes
        A=Target[2:,2:];
        #extract the off diagonal block matrix and its transpose with rotation
        #implemented.
        C=np.matmul(Target[2:,0:2],np.transpose(R));
        CT=np.transpose(C);
            
                   
        #subtrahend for the homodyne result
        G=np.matmul(np.matmul(C,PMM),CT);
        #The matrix for the Moore-Penrose psuedoinverse
        v=np.matmul(np.matmul(PMM,B),PMM);
        scalar=v[0,0];
    
        if scalar==0:
            Target=A; #Moore penrose inverse of a null matrix is null.
        else:
            Target=A-G*(1/scalar); #Use the (1,1) element of the MP pseudoinverse as in Brask, pg. 5.
        
        return Target 
    
def PureFidelity(Ax,Bx):
    '''
    
    quadrature format: xpxp
    Parameters
    ----------
    A : numpy array
        Covariance matrix.
    B : numpy array
        Covariance matrix.

    Returns
    -------
    The overlap between states corresponding to the covariance matrices Ax and Bx.
    These correspond to both pure states.

    '''
    F=1/np.sqrt(np.sqrt(np.linalg.det(Ax+Bx)))
    
    return F**2

    
def MixedFidelity(Ax,Bx):
    '''
    
    quadrature format: xpxp
    Parameters
    ----------
    A : numpy array
        Covariance matrix.
    B : numpy array
        Covariance matrix.

    Returns
    -------
    The overlap between states corresponding to the covariance matrices Ax and Bx.
    These correspond to both generally mixed states.

    '''
    Ohm=np.array([
        
        [0,1,0,0],
        [-1,0,0,0],
        [0,0,0,1],
        [0,0,-1,0]
        
        ])
    
    V = np.matmul(np.matmul(np.transpose(Ohm),np.linalg.inv((Ax + Bx))),(0.25*Ohm + np.matmul(np.matmul(Bx,Ohm),Ax)))
    W=-2*V*1j*Ohm
    
    D=np.linalg.det(Ax+Bx);
    F= np.divide(np.linalg.det(np.matmul((sp.linalg.sqrtm(np.eye(4)-np.linalg.matrix_power(np.linalg.inv(W),2))+np.eye(4)),np.matmul(W,1j*Ohm))),D, where=D!=0)**(0.25)
    
    F=np.sqrt(1/(np.linalg.det(Ax+Bx)))
    return F**2
 
def Cov_to_Adjacency(C):
    X=Permute_XPXP_to_XXPP(C)
    L=len(X[0]);
    U=np.linalg.inv(2*X[0:int(0.5*L),0:int(0.5*L)])
    V=np.matmul(U,2*X[0:int(0.5*L),int(0.5*L):L])
    Z=V+1j*U
    
    return Z
    
def CycleLastMode(X):
    '''
    

    Parameters
    ----------
    X : 2D array
        Covariance matrix.

    Returns
    -------
    X_KD : 2D array
         same covariance matrix with the last mode cycled to the front.

    '''
    n=len(X[0])
    A1=X[n-2:n,n-2:n]
    A1C=X[0:n-2,n-2:n]
    
    X_KD=np.block([
        
        [A1,np.transpose(A1C)],
        [A1C,X[0:n-2,0:n-2]]
                    ])
    
    return X_KD

def KD(r,N,dynamic):
    '''
    

    Parameters
    ----------
    N : integer
        Number of modes in the resulting state.

    Returns
    -------
    cov : Knight's distance measured covariance matrix of N modes. Measurement parameters are randomly picked. Change it to set them manually.  
        2d array of floating points.

    '''
    if dynamic==False:
        Cov_1D=MakeCluster(r, 2*N, 1, 8, 1)
    elif dynamic==True:        
        Cov_1D=MakeDynamicCluster(r, 2*N, np.zeros(2*N), 1, 8, 1)
    
    
    A=chop(HD_last_mode(HD_first_mode(Cov_1D, 0),0))
    
    parameters=np.random.uniform(low=-0.5*np.pi, high=0.5*np.pi, size=(3*N-2,)).tolist()
    
    #parameters=[]  #set parameters manually here and comment the random selection of parameters.
    
    
    #even N
    if np.mod(N,2)==0:
        cov=A
        mc=0
        while mc<N-1:
               
            if np.mod(mc,2)==0:
                i=0
                cov=CycleLastMode(cov)
                while i<4:        
                    cov=HD_last_mode(cov, parameters.pop(0))
                    i=i+1;                
                mc=mc+1;
                    
            elif np.mod(mc,2)!=0:
                i=0
                cov=CycleLastMode(cov)
                while i<2:        
                    cov=HD_last_mode(cov, parameters.pop(0))
                    i=i+1;
               
                mc=mc+1;
    else:
    #odd N
        cov=A
        mc=0
        
        while mc<N:
            
            if np.mod(mc,2)==0:
                i=0
                if mc==N-1:
                    cov=CycleLastMode(cov)
                    while i<1:        
                        cov=HD_last_mode(cov,  parameters.pop(0))
                        i=i+1; 
                    
                    mc=mc+1
                    
                else:
                    cov=CycleLastMode(cov)
                    while i<4:        
                        cov=HD_last_mode(cov,  parameters.pop(0))
                        i=i+1; 
                    
                    mc=mc+1;
                    
            elif np.mod(mc,2)!=0:
                i=0
                cov=CycleLastMode(cov)
                while i<2:        
                    cov=HD_last_mode(cov,  parameters.pop(0))
                    i=i+1;
                
                mc=mc+1;
            
    return cov
