import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

def entropy_MI(softmax, samples_iter):

    class_0 = softmax[:,0]
    class_1 = softmax[:,1]
    #biased estimator of predictive entropy, bias will reduce as samples_iter is increased
    entropy = -((np.sum(class_0)/samples_iter) * np.log(np.sum(class_0)/samples_iter) + (np.sum(class_1)/samples_iter) * np.log(np.sum(class_1)/samples_iter))
    
    mutual_info = entropy + np.sum(class_0*np.log(class_0))/samples_iter +  np.sum(class_1*np.log(class_1))/samples_iter
    
    #entropy of a single pass
    class0 = softmax[:,0][0]
    class1 = softmax[:,1][0]
    entropy_singlepass = -(class0 * np.log(class0) + class1 * np.log(class1))
    #print("Entropy of a single pass:", entropy_singlepass)
    return entropy, mutual_info, entropy_singlepass

def overlapping(x, y, beta=0.1):

    n_z = 100 #100
    z = np.linspace(0,1,n_z)
    dz = 1./n_z
    
    norm = 1./(beta*np.sqrt(2*np.pi))
    
    n_x = len(x)
    f_x = np.zeros(n_z)
    for i in range(n_z):
        for j in range(n_x):
            f_x[i] += norm*np.exp(-0.5*(z[i] - x[j])**2/beta**2)
        f_x[i] /= n_x
    
    
    n_y = len(y)
    f_y = np.zeros(n_z)
    for i in range(n_z):
        for j in range(n_y):
            f_y[i] += norm*np.exp(-0.5*(z[i] - y[j])**2/beta**2)
            
        f_y[i] /= n_y
    
    
    eta_z = np.zeros(n_z)
    eta_z = np.minimum(f_x, f_y)
    
    #pl.subplot(111)
    #pl.plot(z, f_x, label=r"$f_x$")
    #pl.plot(z, f_y, label=r"$f_y$")
    #pl.plot(z, eta_z, label=r"$\eta_z$")
    #pl.legend()
    #pl.show()
    
    return np.sum(eta_z)*dz

def GMM_logits(y_logits, n_components):
    
    def draw_ellipse(position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()
        
        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)
        
        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                 angle, **kwargs))
            
    def plot_gmm(gmm, X, label=True, ax=None):
        ax = ax or plt.gca()
        labels = gmm.fit_predict(X)
        if label:
            #ax.set_facecolor('white')
    
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap='viridis', zorder=2)
            ax.set_xlabel("class0 (FR I) logits")
            ax.set_ylabel("class1 (FR II) logits")
            #ax.set_title("Uncertain Classification")
        else:
            ax.scatter(X[:, 0], X[:, 1], s=10, zorder=2)
            
        ax.axis('equal')
        #print("Means:", gmm.means_)
        print("Covariances:", gmm.covariances_)
        covs = gmm.covariances_
        w_factor = 0.2 / gmm.weights_.max()
        for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
            draw_ellipse(pos, covar, alpha=w * w_factor)
            
        return covs
    #BayesianGaussianMixture?
    gmm = GaussianMixture(n_components, covariance_type='full', random_state=None)
    covs = plot_gmm(gmm, y_logits)
    return covs