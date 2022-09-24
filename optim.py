import numpy as np

def GD(f, df, z0, alpha = 0.1, epsilon=0.1):
    """
        Runs Gradient Descent until accuracy epsilon (||df||<\epsilon^2)
        
        INPUTS
            f: function to minimize
            df: full gradient oracle
            z0: initialization
            alpha: learning rate (constant)
            epsilon: stopping condition 
            
        OUTPUTS
            Z: shape (T,p) sequence of descent
            C: shaphe (T,p) number of oracle calls        
    """
    Tmax = 10000 
    Z = [z0]
    G = [df(z0)] 
    C = [0]
    
    while (len(Z)<Tmax and np.linalg.norm(G[-1])>epsilon**(1/2)): 
        zprev = Z[-1]
        gprev = G[-1]
        znext = zprev-alpha*gprev
        Z.append(znext)
        G.append(df(znext))
        C.append(len(z0))
        
    if len(Z)>=Tmax:
        print("WARNING convergence has failed, max number of interations is reached!")
        
    return (np.array(Z),C)



def LRGD(f, partialdf, z0, alpha = 0.1, rank=1, epsilon=0.1, mixed=0):
    """
        Implementation of Low-Rank Gradient Descent
        
        INPUTS
            f: function to minimize
            deltaf: partial derivative oracle
            z0: initialization
            alpha: learning rate (constant)
            rank: note that rank=p reproduces behaviour of GD
            epsilon: stopping condition (||df||<\epsilon^2)
            mixed: parameter in [0,1]. Allows for mix btw GD and LRGD. 
            
        OUTPUTS
            Z: shape (T,p) sequence of descent
            C: shaphe (T,p) number of oracle calls    
            
        INPUTS
            for arbitrary rank. Note behaviour is the same as GD for full rank. 
    """
    Z = [z0] # point
    G = [] # gradient
    C = [0] # oracle cost
    p = len(z0) # ambiant dimension
    
    def GD_iter():
        #compute
        g = np.sum([partialdf(Z[-1],u)*u for u in np.identity(p)], axis=0)
        #g = deltafU(Z[-1], np.identity(p))
        znext = Z[-1]-alpha*g
        
        #update
        Z.append(znext)
        G.append(g)
        C.append(p)
    
    def LR_iter():
        """Only called after len(G)>=r"""
        U, _ , _ = np.linalg.svd(np.array(G[-rank:]).T,full_matrices=False)
        #g = deltafU(Z[-1],U)
        g = np.sum([partialdf(Z[-1],u)*u for u in U.T], axis=0)
        znext = Z[-1]-alpha*g
        
        #update
        Z.append(znext)
        G.append(g)
        C.append(rank)
        #print("next pos : ", Z[-1])
    
    # INITIALIZATION
    for _ in range(rank):
        GD_iter()
    
    # RUNNING
    while np.linalg.norm(G[-1]) > epsilon**(1/2): 
        #print("hi", np.linalg.norm(G[-1]))
        
        if np.random.uniform()>mixed:
            # Always true for "pure" strategies
            LR_iter()
        else: 
            GD_iter()
        
        if np.linalg.norm(G[-1]) < epsilon**(1/2): 
    
            GD_iter()
    
    return (np.array(Z),C)