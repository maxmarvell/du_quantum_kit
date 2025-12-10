import numpy as np
import quimb.tensor as qtn
from scipy.linalg import expm
import os
from os import mkdir

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def fSim_gate(theta, phi):
    """
    Return a 4-index tensor U_{ijkl} representing the two-qubit fSim(θ, φ) gate in the standard computational basis.
    """
    c = np.cos(theta*np.pi)
    s = np.sin(theta*np.pi)
    ephi = np.exp(-1j * phi*np.pi)

    U = np.array([
        [1,   0,    0,   0],
        [0,   c,  -1j*s, 0],
        [0, -1j*s,  c,   0],
        [0,   0,    0, ephi]
    ], dtype=complex)

    return U.reshape(2, 2, 2, 2)

def random_unitary(q):
    """
    Generate a random unitary matrix of size q^2 x q^2 using the QR decomposition method.
    """
    dim = q**2
    Z = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    D = np.diag(R)
    L = np.diag(D / np.abs(D))
    U = np.dot(Q, L)
    return U.reshape(q, q, q, q)

def FoldedGate(U:np.ndarray):
    """
    Given a rank-4 q x q x q x q tensor U_{ijkl}, return the folded gate, a rank-4 q^2 x q^2 x q^2 x q^2 tensor W_{(ij)(kl)} = U_{ijkl} \otimes U_{ijkl}^*.
    """
    q = U.shape[0]
    W = np.einsum('abcd,efgh->aebfcgdh', U, np.conj(U)).reshape(q**2, q**2, q**2, q**2)
    return W

def PauliBasisW(W:np.ndarray):
    """
    Given a rank-4 q^2 x q^2 x q^2 x q^2 tensor W, return its representation in the Pauli basis as a rank-4 q^2 x q^2 x q^2 x q^2 tensor W_Pauli.
    """
    q = int(np.sqrt(W.shape[0]))
    pauli_matrices = [
        np.array([1, 0, 0, 1]),  # I
        np.array([1, 0, 0, -1]), # Z
        np.array([0, 1, 1, 0]),  # X
        np.array([0, -1j, 1j, 0]) # Y
    ]
    pauli_basis = []
    for i in range(4):
        for j in range(4):
            pauli_basis.append(np.kron(pauli_matrices[i], pauli_matrices[j])/2)
    W_Pauli = np.zeros((q**4, q**4), dtype=complex)
    W = W.reshape(16,16)
    for i in range(q**4):
        for j in range(q**4):
            P_i = pauli_basis[i]
            P_j = pauli_basis[j]
            coeff = np.conj(P_i).T @ W @ P_j
            W_Pauli[i,j] = coeff
    
    return W_Pauli.reshape(q**2, q**2, q**2, q**2)

def main():

    q = 2
    d = 4
    epsilon = 0.25
    theta = 0.5-epsilon
    phi = 2*epsilon

    U = fSim_gate(theta, phi)
    tile_dir = os.path.join(SCRIPT_DIR, f"Tile_Zoo_fSim_theta{theta}_phi{phi}_nparrays")

    W_non_pauli = FoldedGate(U)
    print(W_non_pauli)
    W = PauliBasisW(W_non_pauli)
    get_tiles(W, d, tile_dir)
    
def get_tiles(W:np.ndarray,
              d:int,
              dir:str):

    '''
        gets the four types of tiles corresponding to horizontal defect,
        horizontal direct, vertical defect, vertical direct for a d dimensional
        case

        N.B. convention for index labelling is "kx,y where (x,y) is the bond coordinate"
    '''

    #tiles = dict()

    os.makedirs(dir, exist_ok=True)


    # HORIZONTAL DIRECT TILE

    for d_v in range(1,d+1):
        for d_h in range(1,d+1):
            if (d_v != 1 and d_h != 1) or (d_v !=1 and d_h == 1):

                tensors = np.array([])
                
                if d_v != 1:
                    for i in range(d_h):
                        tensors = np.append(tensors,[qtn.Tensor(W[:,:,:,0],inds=(f'k{2*i+1},2',f'k{2*i+2},1',f'k{2*i},1'))])
                        tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k{2*i+1},{2*j+2}',f'k{2*i+2},{2*j+1}',
                                                                            f'k{2*i},{2*j+1}',f'k{2*i+1},{2*j}')) for j in range(1,d_v-1)])
                        tensors = np.append(tensors,[qtn.Tensor(W[0,:,:,:],inds=(f'k{2*i+2},{2*(d_v-1)+1}',f'k{2*i},{2*(d_v-1)+1}',f'k{2*i+1},{2*(d_v-1)}'))])
                else:
                    for i in range(d_h):
                        tensors = np.append(tensors,[qtn.Tensor(W[0,:,:,0],inds=(f'k{2*i+1},2',f'k{2*i+2},1',f'k{2*i},1'))])


                TN = qtn.TensorNetwork(tensors)
                val = TN.contract()

                reshape = tuple()
            
                for i in range(d_v):
                    reshape = reshape + (f'k{2*(d_h-1)+2},{2*i+1}',)
                for i in range(d_v):
                    reshape = reshape + (f'k0,{2*i+1}',)

                val = val.transpose(*reshape,inplace=True)
                

                h_direct = val.data
                h_direct = h_direct.reshape(-1,*h_direct.shape[-d_v:])
                h_direct = h_direct.reshape(*h_direct.shape[:1],-1)

                #np.savetxt(f"{dir}/h_direct_{d_h}x{d_v}", h_direct, delimiter=",")
                np.save(f"{dir}/h_direct_{d_h}x{d_v}.npy", h_direct)

            else:
                h_direct = W[0,:,:,0]
                for i in range(d_h-1):
                    h_direct = np.einsum('ab,bc->ac',h_direct,W[0,:,:,0])
                #np.savetxt(f"{dir}/h_direct_{d_h}x{d_v}", h_direct, delimiter=",")
                np.save(f"{dir}/h_direct_{d_h}x{d_v}.npy", h_direct)



    # HORIZONTAL DEFECT TILE

    for d_v in range(1,d+1):
        for d_h in range(1,d+1):
            if d_v != 1 or d_h != 1:

                tensors = np.array([])

                tensors = np.append(tensors,[qtn.Tensor(W[:,:,:,0],inds=(f'k{2*i+1},2',f'k{2*i+2},1',f'k{2*i},1')) for i in range(d_h-1)])
                tensors = np.append(tensors,[qtn.Tensor(W[:,0,:,0],inds=(f'k{2*(d_h-1)+1},2',f'k{2*(d_h-1)},1'))])

                for j in range(1,d_v):
                    tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k{2*i+1},{2*j+2}',f'k{2*i+2},{2*j+1}',
                                                                    f'k{2*i},{2*j+1}',f'k{2*i+1},{2*j}')) for i in range(d_h-1)])
                    
                    tensors = np.append(tensors,[qtn.Tensor(W[:,0,:,:],inds=(f'k{2*(d_h-1)+1},{2*j+2}',f'k{2*(d_h-1)},{2*j+1}',f'k{2*(d_h-1)+1},{2*j}'))])


                TN = qtn.TensorNetwork(tensors)
                val = TN.contract()

                reshape = tuple()
                
                for i in range(d_h):
                    reshape = reshape + (f'k{2*i+1},{2*d_v}',)
                for i in range(d_v):
                    reshape = reshape + (f'k0,{2*i+1}',)

                val = val.transpose(*reshape,inplace=True)
                

                h_defect = val.data
                h_defect = h_defect.reshape(-1,*h_defect.shape[-d_v:])
                h_defect = h_defect.reshape(*h_defect.shape[:1],-1)

                #np.savetxt(f"{dir}/h_defect_{d_h}x{d_v}", h_defect, delimiter=",")
                np.save(f"{dir}/h_defect_{d_h}x{d_v}.npy", h_defect)

            else:
                #np.savetxt(f"{dir}/h_defect_{d_h}x{d_v}", W[:,0,:,0], delimiter=",")
                np.save(f"{dir}/h_defect_{d_h}x{d_v}.npy", W[:,0,:,0])



    ### VERTICAL DIRECT TILE

    for d_v in range(1,d+1):
        for d_h in range(1,d+1):
            if (d_v != 1 and d_h != 1) or (d_v == 1 and d_h != 1):

                tensors = np.array([])
                    
                for j in range(d_v):
                    tensors = np.append(tensors,[qtn.Tensor(W[:,:,0,:],inds=(f'k1,{2*j+2}',f'k2,{2*j+1}',f'k1,{2*j}'))])
                    tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k{2*i+1},{2*j+2}',f'k{2*i+2},{2*j+1}',f'k{2*i},{2*j+1}',f'k{2*i+1},{2*j}')) for i in range(1,d_h-1)])
                    tensors = np.append(tensors,[qtn.Tensor(W[:,0,:,:],inds=(f'k{2*(d_h-1)+1},{2*j+2}',f'k{2*(d_h-1)},{2*j+1}',f'k{2*(d_h-1)+1},{2*j}'))])

                TN = qtn.TensorNetwork(tensors)
                val = TN.contract()

                reshape = tuple()
                
                for i in range(d_h):
                    reshape = reshape + (f'k{2*i+1},{2*(d_v-1)+2}',)
                for i in range(d_h):
                    reshape = reshape + (f'k{2*i+1},0',)

                val = val.transpose(*reshape,inplace=True)
                

                v_direct = val.data
                v_direct = v_direct.reshape(-1,*v_direct.shape[-d_h:])
                v_direct = v_direct.reshape(*v_direct.shape[:1],-1)

                #np.savetxt(f"{dir}/v_direct_{d_h}x{d_v}", v_direct, delimiter=",")
                np.save(f"{dir}/v_direct_{d_h}x{d_v}.npy", v_direct)

            else:
                v_direct = W[:,0,0,:]
                for i in range(d_v-1):
                    v_direct = np.einsum('ab,bc->ac',v_direct,W[:,0,0,:])
                #np.savetxt(f"{dir}/v_direct_{d_h}x{d_v}", v_direct, delimiter=",")
                np.save(f"{dir}/v_direct_{d_h}x{d_v}.npy", v_direct)



    ### VERTICAL DEFECT TILE

    for d_v in range(1,d+1):
        for d_h in range(1,d+1):
            if d_v != 1 or d_h != 1:

                tensors = np.array([])

                for j in range(d_v-1):              
                    tensors = np.append(tensors,[qtn.Tensor(W[:,:,0,:],inds=(f'k1,{2*j+2}',f'k2,{2*j+1}',f'k1,{2*j}'))])
                    tensors = np.append(tensors,[qtn.Tensor(W,inds=(f'k{2*i+1},{2*j+2}',f'k{2*i+2},{2*j+1}',
                                                                    f'k{2*i},{2*j+1}',f'k{2*i+1},{2*j}')) for i in range(1,d_h)])

                tensors = np.append(tensors,[qtn.Tensor(W[0,:,0,:],inds=(f'k2,{2*(d_v-1)+1}',f'k1,{2*(d_v-1)}'))])
                tensors = np.append(tensors,[qtn.Tensor(W[0,:,:,:],inds=(f'k{2*i+2},{2*(d_v-1)+1}',f'k{2*i},{2*(d_v-1)+1}',f'k{2*i+1},{2*(d_v-1)}')) for i in range(1,d_h)])

                TN = qtn.TensorNetwork(tensors)

                val = TN.contract()

                reshape = tuple()
                
                for i in range(d_v):
                    reshape = reshape + (f'k{2*d_h},{2*i+1}',)
                for i in range(d_h):
                    reshape = reshape + (f'k{2*i+1},0',)

                val = val.transpose(*reshape,inplace=True)
                

                v_defect = val.data
                v_defect = v_defect.reshape(-1,*v_defect.shape[-d_h:])
                v_defect = v_defect.reshape(*v_defect.shape[:1],-1)

                #np.savetxt(f"{dir}/v_defect_{d_h}x{d_v}", v_defect, delimiter=",")
                np.save(f"{dir}/v_defect_{d_h}x{d_v}.npy", v_defect)

            else:
                #np.savetxt(f"{dir}/v_defect_{d_h}x{d_v}", W[0,:,0,:], delimiter=",")
                np.save(f"{dir}/v_defect_{d_h}x{d_v}.npy", W[0,:,0,:])


if __name__ == '__main__':
    main()