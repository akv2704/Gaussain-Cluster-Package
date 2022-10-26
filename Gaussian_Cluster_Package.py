# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 14:42:31 2022
@author: abhve
"""
import scipy as sp
import csv
import pandas as pd
import numpy as np
import strawberryfields as sf
import matplotlib.pyplot as plt
import sys
####################################################Functions######################################################
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
        SMSV.append((np.cosh(r[i])*np.eye(2))-np.sinh(r[i])*np.array([[np.cos(angle[i]), np.sin(angle[i])],[-np.sin(angle[i]), np.cos(angle[i])]]))
    
    SingleModeSym=sp.linalg.block_diag(*tuple(SMSV))
    
    return np.array(SingleModeSym)


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
        A positive definite matrix or covariance matrix.

    Returns
    -------
    complex 2D array
        The Unitary that transforms vacua into this covariance matrix.

    '''
    (S_Eigen,S)=sf.decompositions.williamson(Permute_XPXP_to_XXPP(V),tol=1e-13)
    (K,Sr,L)=sf.decompositions.bloch_messiah(S,tol=1e-6)
    A=SymToUni(K);
    
    return A;