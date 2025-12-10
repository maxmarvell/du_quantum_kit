import numpy as np
from time import time
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm
import os
from matplotlib import cm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():

    d = 3
    T = 10
    q = 2
    
    epsilon = 0.25
    theta = 0.5-epsilon
    phi = 2*epsilon

    Tile_dir = os.path.join(SCRIPT_DIR, f"Tile_Zoo_fSim_theta{theta}_phi{phi}_nparrays")
    canvas = path_integral(T,d,q,Tile_dir)

    def check_charge_cons(canvas):
        dimt, dimx = canvas.shape
        charges = np.full(shape=[dimt],fill_value=0,dtype="complex_")
        for t in range(dimt):
            sum = 0.0
            for x in range(dimx):
                sum += canvas[t,x]
            charges[t] = sum
        return charges
    
    charges = check_charge_cons(canvas)
    print("Charges at each time step:")
    print(charges)

    # Base larder directory
    larder_base = os.path.join(SCRIPT_DIR, "Larders", f"Larder_fSim_theta{theta}_phi{phi}")
    os.makedirs(larder_base, exist_ok=True)

    # Subdirectory for this (T, d) parameter set
    canvas_dir = os.path.join(larder_base, f"T{T}_d{d}")
    os.makedirs(canvas_dir, exist_ok=True)

    # Save canvas
    filename = f"canvas_theta{theta}_phi{phi}_T{T}_d{d}.npy"
    np.save(os.path.join(canvas_dir, filename), canvas)

    # Build data for plotting (mask zeros)
    data = np.abs(canvas)
    masked = np.ma.masked_where(data == 0, data)

    # Take log *after* masking (masked zeros stay masked)
    log_data = -np.log10(masked)

    # Use a reversed colormap
    cmap = cm.get_cmap("hot_r").copy()

    # Set the color for masked values (zeros) to black
    cmap.set_bad(color='black')
    
    # Plot the canvas
    plt.figure(figsize=(10, 6))
    plt.imshow(log_data, cmap=cmap, interpolation="nearest", aspect="auto")
    plt.colorbar(label="$-\\log_{10}|(Z_x|Z_0(t))|$")
    plt.title(f"Canvas (θ={theta}, φ={phi}, T={T}, d={d})")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()


def path_integral(T:float, d:int, q:int, Tile_dir:str):

    def transfer_matrix(t:float, X:float, Tile_Zoo:dict, a: np.ndarray,
                        l:int, dim_h:int, dim_v:int, d:int,
                        horizontal:bool = True,
                        terminate:bool = False,
                        status:str = "Main"):

        """
            A transfer matrix can either be horizontal or vertical row of contracted
            folded tensors. 
            For the case of a horizontal transfer matrix the row can only be terminated
            either by a defect or by the operator b

            a can be like [0,1,0,0]
        """

        if horizontal:
            direct = Tile_Zoo[status + " h_direct"]
            defect = Tile_Zoo[status + " h_defect"]
        else:
            direct = Tile_Zoo[status + " v_direct"]
            defect = Tile_Zoo[status + " v_defect"]


        if not terminate:
            for i in range(l-1):

                if i == 0 and status == "Low Trim" and (dim_h%d) != 0:
                    a = np.einsum('ab,b->a',Tile_Zoo["Corner h_direct"],a)
                elif i != 0 and status == "Left Trim" and horizontal:
                    a = np.einsum('ab,b->a',Tile_Zoo["Main h_direct"],a)
                else:
                    a = np.einsum('ab,b->a',direct,a)

            if l == 1 and status == "Low Trim" and (dim_h%d) != 0:
                return np.einsum('ab,b->a',Tile_Zoo["Corner h_defect"],a)  
            elif l > 1 and status == "Left Trim" and horizontal:
                return np.einsum('ab,b->a',Tile_Zoo["Main h_defect"],a)
            else:
                return np.einsum('ab,b->a',defect,a) 


        elif (int(2*(t+X))%2 == 0) and horizontal:
            for i in range(l):

                if i == 0 and status == "Low Trim" and (dim_h%d) != 0:
                    a = np.einsum('ab,b->a',Tile_Zoo[f"Corner h_direct"],a)
                elif i != 0 and status == "Left Trim":
                    a = np.einsum('ab,b->a',Tile_Zoo["Main h_direct"],a)
                else:
                    a = np.einsum('ab,b->a',direct,a)
            return a[1]


        elif ((int(2*(t+X))%2 == 0) and not horizontal):
            for _ in range(l-1):

                a = np.einsum('ab,b->a',direct,a)

            a = np.einsum('ab,b->a',defect,a)
            return a[1]


        elif ((int(2*(t+X))%2 != 0) and not horizontal):
            for _ in range(l):

                a = np.einsum('ab,b->a',direct,a)
            return a[1]


        else:
            for i in range(l-1):

                if i == 0 and status == "Low Trim" and (dim_h%d) != 0:
                    a = np.einsum('ab,b->a',Tile_Zoo[f"Corner h_direct"],a)
                elif i != 0 and status == "Left Trim":
                    a = np.einsum('ab,b->a',Tile_Zoo["Main h_direct"],a)
                else:
                    a = np.einsum('ab,b->a',direct,a)

            if l == 1 and status == "Low Trim" and (dim_h%d) != 0:
                a = np.einsum('ab,b->a',Tile_Zoo[f"Corner h_defect"],a)
                return a[1]

            elif l > 1 and status == "Left Trim":
                a = np.einsum('ab,b->a',Tile_Zoo["Main h_defect"],a)
                return a[1]

            else:
                a = np.einsum('ab,b->a',defect,a)
                return a[1]



    def skeleton(t:float, X:float, x_h:np.ndarray, x_v:np.ndarray, Tile_Zoo:dict,
                a:np.ndarray, dim_h:int, dim_v:int, d:int):
        '''
            Computes a contribution to the path integral of the
            input skeleton diagram, only for cases where y is an integer!
        '''

        if x_v == []:

            status = "Low Trim" if (dim_v%d) != 0 else "Left Trim" if ((dim_h%d)!=0 and (dim_v%d)==0) else "Main"        
            return transfer_matrix(t,X,Tile_Zoo,a,x_h[-1],dim_h,dim_v,d,terminate=True,status=status)
        
        elif len(x_v) == len(x_h):
            for i in range(len(x_v)-1):

                status = "Low Trim" if ((dim_v%d)!=0 and i==0) else "Left Trim" if ((dim_h%d)!=0 and (dim_v%d)==0 and i==0) else "Main"
                a = transfer_matrix(t,X,Tile_Zoo,a,x_h[i],dim_h,dim_v,d,status=status)

                status = "Left Trim" if ((dim_h%d)!=0 and x_h[0]==1 and i==0) else "Main"
                a = transfer_matrix(t,X,Tile_Zoo,a,x_v[i],dim_h,dim_v,d,horizontal=False,status=status)

            status = "Low Trim" if ((dim_v%d)!=0 and len(x_v)==1) else "Left Trim" if ((dim_h%d)!=0 and (dim_v%d)==0 and len(x_v)==1) else "Main"
            a = transfer_matrix(t,X,Tile_Zoo,a,x_h[-1],dim_h,dim_v,d,status=status)

            status = "Left Trim" if ((dim_h%d)!=0 and x_h[0]==1 and len(x_v)==1) else "Main"
            return transfer_matrix(t,X,Tile_Zoo,a,x_v[-1],dim_h,dim_v,d,horizontal=False,terminate=True,status=status)

        else:
            for i in range(len(x_v)):

                status = "Low Trim" if ((dim_v%d)!=0 and i==0) else "Left Trim" if ((dim_h%d)!=0 and (dim_v%d)==0 and i==0) else "Main"
                a = transfer_matrix(t,X,Tile_Zoo,a,x_h[i],dim_h,dim_v,d,status=status)

                status = "Left Trim" if ((dim_h%d)!=0 and x_h[0]==1 and i==0) else "Main"
                a = transfer_matrix(t,X,Tile_Zoo,a,x_v[i],dim_h,dim_v,d,horizontal=False,status=status)

            return transfer_matrix(t,X,Tile_Zoo,a,x_h[-1],dim_h,dim_v,d,terminate=True,status="Main")



    def list_generator(x:int,data:dict,k:int=np.inf,
                       lists:np.ndarray=[]):
        '''
            Generates a complete set of possible lists which can
            combine to form a complete set 
        '''

        if x == 0:
            try:
                data[len(lists)].append(lists)
            except:
                data[len(lists)] = [lists]
            return
        elif len(lists) >= k:
            return 

        for i in range(1,x+1):
            sublist = lists.copy()
            sublist.append(i)
            list_generator(x-i,data,k,sublist)



    canvas = np.full(shape=[int(2*T), int(4*T)-2], fill_value=0.0)
    canvas[0,int(2*T)-1] = 1.0

    Tile_Zoo = dict()

    Tile_Zoo["Main h_direct"] = np.load(f"{Tile_dir}/h_direct_{d}x{d}.npy")
    Tile_Zoo["Main h_defect"] = np.load(f"{Tile_dir}/h_defect_{d}x{d}.npy")
    Tile_Zoo["Main v_direct"] = np.load(f"{Tile_dir}/v_direct_{d}x{d}.npy")
    Tile_Zoo["Main v_defect"] = np.load(f"{Tile_dir}/v_defect_{d}x{d}.npy")
    


    for t in range(1, int(2*T)):
        print("t = ", t)
        for x in range(-t, t):
            print("x = ", x)

            vertical_data = {}
            horizontal_data = {}
            dim_h = math.ceil(t/2 - x/2)
            dim_v = math.floor(t/2 + 1 + x/2)
            k = min(dim_h, dim_v)

            if (dim_v%d) != 0:
                Tile_Zoo["Low Trim h_direct"] = np.load(f"{Tile_dir}/h_direct_{d}x{dim_v%d}.npy")
                Tile_Zoo["Low Trim h_defect"] = np.load(f"{Tile_dir}/h_defect_{d}x{dim_v%d}.npy")
                Tile_Zoo["Low Trim v_direct"] = np.load(f"{Tile_dir}/v_direct_{d}x{dim_v%d}.npy")
                Tile_Zoo["Low Trim v_defect"] = np.load(f"{Tile_dir}/v_defect_{d}x{dim_v%d}.npy")

                a = np.zeros(q**(2*(dim_v%d)))
                a[q**(2*(dim_v%d - 1))] = 1.0
                list_generator((dim_v//d + 1)-1,vertical_data)
            else:
                a = np.zeros(q**(2*d))
                a[q**(2*(d - 1))] = 1.0
                list_generator((dim_v//d)-1,vertical_data)
            

            if (dim_h%d) != 0:
                Tile_Zoo["Left Trim h_direct"] = np.load(f"{Tile_dir}/h_direct_{dim_h%d}x{d}.npy")
                Tile_Zoo["Left Trim h_defect"] = np.load(f"{Tile_dir}/h_defect_{dim_h%d}x{d}.npy")
                Tile_Zoo["Left Trim v_direct"] = np.load(f"{Tile_dir}/v_direct_{dim_h%d}x{d}.npy")
                Tile_Zoo["Left Trim v_defect"] = np.load(f"{Tile_dir}/v_defect_{dim_h%d}x{d}.npy")
                list_generator((dim_h//d + 1),horizontal_data,k=k)
            else:
                list_generator((dim_h//d),horizontal_data,k=k)


            if (dim_v%d) != 0 and (dim_h%d) != 0:
                Tile_Zoo["Corner h_direct"] = np.load(f"{Tile_dir}/h_direct_{dim_h%d}x{dim_v%d}.npy")
                Tile_Zoo["Corner h_defect"] = np.load(f"{Tile_dir}/h_defect_{dim_h%d}x{dim_v%d}.npy")



            n = 1
            sum = 0.0
    
            while n <= k:

                try:
                    l1, l2 = horizontal_data[n], vertical_data[n]
                    for h in l1:
                        for v in l2:
                            sum += skeleton(t/2,-x/2,h,v,Tile_Zoo,a,dim_h,dim_v,d)
                #except Exception as e:
                #    # Print the exception message
                #    print(f"An exception occurred: {str(e)}" + f" shape = {dim_h}x{dim_v}")
                #    pass
                except Exception as e:
                    print("\nEXCEPTION:", repr(e))
                    print("At t=", t, "x=", x)
                    print("dim_h=", dim_h, "dim_v=", dim_v)
                    print("horizontal keys:", list(horizontal_data.keys()))
                    print("vertical keys:", list(vertical_data.keys()))
                    print("Tile_Zoo keys:", list(Tile_Zoo.keys()))
                #    raise
                    pass
                    
                try:
                    l1, l2 = horizontal_data[n + 1], vertical_data[n]
                    for h in l1:
                        for v in l2:
                            sum += skeleton(t/2,-x/2,h,v,Tile_Zoo,a,dim_h,dim_v,d)
                #except Exception as e:
                #    # Print the exception message
                #    print(f"An exception occurred: {str(e)}" + f" shape = {dim_h}x{dim_v}")
                #    pass
                except Exception as e:
                    print("\nEXCEPTION:", repr(e))
                    print("At t=", t, "x=", x)
                    print("dim_h=", dim_h, "dim_v=", dim_v)
                    print("horizontal keys =", list(horizontal_data.keys()))
                    print("vertical keys =", list(vertical_data.keys()))
                    print("Tile_Zoo keys =", list(Tile_Zoo.keys()))
                #    raise
                    pass

                if n == 1:               
                    try:
                        l1, v = horizontal_data[n], vertical_data[0]
                        for h in l1:
                            sum += skeleton(t/2,-x/2,h,[],Tile_Zoo,a,dim_h,dim_v,d)
                    #except Exception as e:
                    #    # Print the exception message
                    #    print(f"An exception occurred: {str(e)}" + f" shape = {dim_h}x{dim_v}")
                    #    pass
                    except Exception as e:
                        print("\nEXCEPTION:", repr(e))
                        print("At t=", t, "x=", x)
                        print("dim_h=", dim_h, "dim_v=", dim_v)
                        print("horizontal keys =", list(horizontal_data.keys()))
                        print("vertical keys =", list(vertical_data.keys()))
                        print("Tile_Zoo keys =", list(Tile_Zoo.keys()))
                    #    raise
                        pass

                n += 1
         

            canvas[t,x+int(2*T)-1] = sum
            print("sum = ", sum)

    return canvas


if __name__ == '__main__':
    main()


