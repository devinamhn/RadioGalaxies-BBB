import pylab as pl
import pickle
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

epoch = 501
#%% density
path = "./model10_GMM_T3_init/4/model10_3"

def create_frame(step,ax):
    ax.cla()
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(0, 30)
    sns.set_style("whitegrid", {'axes.grid' : True})
    
    with open(path + "density"+str(step)+".txt", "rb") as fp:
#    with open("./bbb_plots/density/bbb29/density"+str(step)+".txt", "rb") as fp:
        a = pickle.load(fp)
    #plt.xlabel('Signal-to-Noise (dB)')
    sns.kdeplot(a, ax=ax,legend=False, fill=True, alpha=0.5)#.set_title('epoch'+str(step))
    ax.legend(['epoch'+str(step)])
    
fig = plt.figure(dpi=200)  
ax = fig.gca()

animation = FuncAnimation(fig, create_frame, frames=np.arange(0,epoch,80), fargs=(ax,),
                          interval=2000)
animation.save('animation_density.mp4', writer='ffmpeg', fps=2);
#%%
#signal to noise
def create_frame(step,ax):
    ax.cla()
    ax.set_xlim(-70, 30)
    ax.set_ylim(0, 0.15)
    sns.set_style("whitegrid", {'axes.grid' : True})

    with open(path+"snr"+str(step)+".txt", "rb") as fp:
        a = pickle.load(fp)
    sns.kdeplot(a, ax=ax, fill=True, alpha=0.5)#.set_title('epoch'+str(step))
    ax.legend(['epoch'+str(step)])
    plt.xlabel('Signal-to-Noise (dB)')
    
fig = plt.figure(dpi=200) 
ax = fig.gca()
animation = FuncAnimation(fig, create_frame, frames=np.arange(0,epoch,80), fargs=(ax,),
                          interval=2000)
animation.save('animation_snr.mp4', writer='ffmpeg', fps=2);

