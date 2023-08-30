import numpy as np
import quimb.tensor as qtn
from scipy.linalg import expm 


def main():

    #used e = 0.0000004 as reminder
    q = 2
    e = 0.0000004
    d = 5
    
    W_path = "./Sample_Tensor.csv"
    P_path = "./Sample_Perturbation.csv"
    W = np.loadtxt(W_path,delimiter=',',dtype='complex_')
    G = np.loadtxt(P_path,delimiter=',',dtype='complex_')
    print("\n")
    P = expm(1j*e*G)
    PW = np.einsum('ab,bc->ac',P,W).reshape([q**2,q**2,q**2,q**2])

    get_tiles(PW, d)





def get_tiles(W:np.ndarray,
              d:int):

    '''
        gets the four types of tiles corresponding to horizontal defect,
        horizontal direct, vertical defect, vertical direct for a d dimensional
        case

        N.B. convention for index labelling is "kx,y where (x,y) is the bond coordinate"
    '''

    #tiles = dict()



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

                np.savetxt(f"./Tile_Zoo/h_direct_{d_h}x{d_v}", h_direct, delimiter=",")

            else:
                h_direct = W[0,:,:,0]
                for i in range(d_h-1):
                    h_direct = np.einsum('ab,bc->ac',h_direct,W[0,:,:,0])
                np.savetxt(f"./Tile_Zoo/h_direct_{d_h}x{d_v}", h_direct, delimiter=",")



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

                np.savetxt(f"./Tile_Zoo/h_defect_{d_h}x{d_v}", h_defect, delimiter=",")

            else:
                np.savetxt(f"./Tile_Zoo/h_defect_{d_h}x{d_v}", W[:,0,:,0], delimiter=",")



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

                np.savetxt(f"./Tile_Zoo/v_direct_{d_h}x{d_v}", v_direct, delimiter=",")

            else:
                v_direct = W[:,0,0,:]
                for i in range(d_v-1):
                    v_direct = np.einsum('ab,bc->ac',v_direct,W[:,0,0,:])
                np.savetxt(f"./Tile_Zoo/v_direct_{d_h}x{d_v}", v_direct, delimiter=",")



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

                np.savetxt(f"./Tile_Zoo/v_defect_{d_h}x{d_v}", v_defect, delimiter=",")

            else:
                np.savetxt(f"./Tile_Zoo/v_defect_{d_h}x{d_v}", W[0,:,0,:], delimiter=",")

   


if __name__ == '__main__':
    main()