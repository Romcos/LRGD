import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

def show_trajs(Z1, Z2 = None, f= lambda x:0, title = "a plot", num = 5, savename = None, colorbar = True):
    """
    Displays one (or two if Z2 is not None) trajectories in 2D [-0.5,2]x[-0.5,2]
    
    INPUTS
        Z1 / Z2: shape (T,2) with T: length of trajectories
        f: function as a heatmap
        num: number of dots with number assigned on the plot
    """
    ## Function as background
    xmin = -0.5
    xmax = 2
    N=100

    x1, x2 = np.meshgrid(np.linspace(xmin,xmax, num=N), np.linspace(xmin,xmax, num=N))
    coords = np.stack([x1,x2],axis=2)
    f_coords = np.apply_along_axis(f,2,coords)
    plt.figure(figsize=(4,4))
    plt.imshow(f_coords,extent=[xmin,xmax,xmin,xmax], cmap="jet", origin='lower') #cmap = cm.jet
    if colorbar : 
        plt.colorbar(shrink =0.7, ticks = [0,4,8,12,16], pad = -0.15)
        
    plt.xticks([0,1,2])
    plt.yticks([0,1,2])
        
    ## Scatter trajectory
    markers = ["x","o"]
    
    if Z2 is None : 
        Zs = [Z1]
    else:
        Zs = [Z1, Z2]
    
    for k in range(len(Zs)):
        Z = Zs[k]
        x = Z[:,0]
        y = Z[:,1]
        T = len(x)
        plt.scatter(x=x,y=y,color="white",marker=markers[k],s=10)
        for i, txt in enumerate(x):
            if i< num :
                plt.annotate(str(i), (x[i]+(-1)**k*0.1, y[i]),color="white",fontsize=10)
        #adjust_text(texts) --> see https://blog.finxter.com/matplotlib-text-and-annotate-a-simple-guide/

    plt.title(title)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    if not(savename is None):
        plt.savefig(savename, bbox_inches='tight')

    plt.show()

    
def plot_convergence(Zs, Cs, f, labels=["","",""], verticals = None,colors = None, savename=None,xmax=None,ymin=None):
    """
    Displays the decay of function value f for multiple decents
    colors should contain at least as many entries as Zs.
    """
    fig, ax = plt.subplots(1, 1, figsize = (5,5))
    
    if (colors is None):
        colors = pl.cm.jet(np.linspace(0,1,len(Zs)))
    
    if (verticals is None):
        verticals= [False for _ in range(len(Zs))]
    
    for k in range(len(Zs)): 
        ax.plot(np.cumsum(Cs[k]),[f(Zs[k][i]) for i in range(len(Zs[k]))], label = labels[k], color=colors[k])
        if verticals[k]:
            for i,c in enumerate(Cs[k]):
                if c==np.max(Cs[k]):
                    ax.axvline(x = np.sum(Cs[k][:i]), color = colors[k], linestyle = "--", linewidth = 0.5)
                
    ax.set_yscale("log")
    ax.set_ylabel(r"$f(\theta)$ (log scale)")
    ax.set_xlabel("oracle calls")
    if not(xmax is None):
        ax.set_xlim(left=0,right=xmax)
    else:
        ax.set_xlim(left=0)
        
    if (ymin is None):
        ymin = np.min([f(Zs[i][-1]) for i in range(len(Zs))])
    else:
        ax.set_ylim(bottom=ymin)
    ax.legend()
    if not(savename is None):
        fig.savefig(savename, bbox_inches='tight')
    plt.show()
    
def plot_convergence2(Zs, Cs, f,df, labels=["","",""], verticals = None,colors = None, savename=None,xmax=None,ymin=None):
    """
    Displays the decay of function value f for multiple decents
    colors should contain at least as many entries as Zs.
    """
    fig, ax = plt.subplots(1, 1, figsize = (5,5))
    
    if (colors is None):
        colors = pl.cm.jet(np.linspace(0,1,len(Zs)))
    
    if (verticals is None):
        verticals = [False for _ in range(len(Zs))]
    
    for k in range(len(Zs)): 
        ax.plot(np.cumsum(Cs[k]),[np.linalg.norm(df(Zs[k][i])) for i in range(len(Zs[k]))], label = labels[k], color=colors[k])
        if verticals[k]:
            for i,c in enumerate(Cs[k]):
                if c==np.max(Cs[k]):
                    ax.axvline(x = np.sum(Cs[k][:i]), color = colors[k], linestyle = "--", linewidth = 0.5)
                
    ax.set_yscale("log")
    ax.set_ylabel(r"$|| ~\nabla f(\theta) ~||$ (log scale)")
    ax.set_xlabel("oracle calls")
    if not(xmax is None):
        ax.set_xlim(left=0,right=xmax)
    else:
        ax.set_xlim(left=0)
        
    if (ymin is not None):
        ax.set_ylim(bottom=ymin)
        
    ax.legend()
    if not(savename is None):
        fig.savefig(savename, bbox_inches='tight')
    plt.show()