from sklearn import tree,preprocessing,grid_search,cross_validation,ensemble,svm,learning_curve,metrics
from math import *
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import markers
#import seaborn as sns
from scipy import stats

# Input data file format:
#0  filename 				ignore this
#1  log_L 					log-luminosity
#2  z 						redshift
#3  grid_id 				convert to (r,phi) with python func below
#4  bin_size_emit 			source time bin size
#5  alpha 					Band func param
#6  beta 					Band func param
#7  E_peak 					energy peak (log this)
#8  background_name 		name - ignore
#9  bgd_15-25keV 			bkg in band (log this)
#10 bgd_15-50keV 			bkg in band (log this)
#11 bgd25-100keV 			bkg in band (log this)
#12 bgd50-350keV 			bkg in band (log this)
#13 theta 					incoming angle
#14 flux 					flux of burst (log this)
#15 burst_shape 			name - ignore this or turn into feature(s)
#16 ndet 					number of detectors active
#17 rate_GRB_0_global
#18 z1_global
#19 n1_global
#20 n2_global
#21 lum_star_global
#22 x_global
#23 y_global
#24 Epeak_type
#25 Lum_evo_type
#26 trigger_index			0 if not detected, 1 if detected

class lightcurves:
	names = []
	enc = []
	n = 0
	def read(self,Nenc):
		ifp=open('GRBLCsmooth2_encoded'+repr(Nenc)+'.txt','r')
		enc1=ifp.readlines()
		for i in range(len(enc1)):
			self.names.append(enc1[i].split()[0])
			self.enc.append(enc1[i].split()[1:])
		ifp.close()
		self.n = len(self.names)
	def findidx(self,lcname):
		if self.n>0:
			lcid=self.names.index(lcname)
		else:
			lcid=-1
		return lcid
	def features(self,idx):
		f = np.array(self.enc[int(idx)])
		return f

def id2xy(id):
    x=(id%7-3.0)/2.0
    y=(int(id/7)-2.0)/3.0
    r=hypot(x,y)
    phi=atan2(y,x)
    return (r,phi)

def readdata(filepath,Nenc=0,ndet=True):
    LCs = lightcurves()
    if Nenc>0:
        LCs.read(Nenc)
    flog=lambda x:log10(float(x))
    conv1={7:flog, 8:len, 9:flog, 10:flog, 11:flog, 12:flog, 14:flog, 15:LCs.findidx}
    conv2={7:flog, 8:len, 9:flog, 10:flog, 11:flog, 12:flog, 14:flog, 15:LCs.findidx, 24:len, 25:len}
    ifp=open(filepath,'r')
    linesplit=ifp.readline().split()
    if len(linesplit)>19:
        conv=conv2
        tidx=26
    elif len(linesplit)>18:
        conv=conv1
        tidx=17
    else:
        conv=conv1
        tidx=16
    ifp.close()
    filedata=np.loadtxt(filepath,converters=conv,comments='##')
    if ndet:
        x=np.zeros((len(filedata),15+Nenc))
    else:
        x=np.zeros((len(filedata),14+Nenc))
    y=np.zeros((len(filedata),))
    x[:,0:2]=filedata[:,1:3]
    x[:,2:4]=map(id2xy,filedata[:,3])
    x[:,4:8]=filedata[:,4:8]
    x[:,8:14]=filedata[:,9:15]
    if ndet:
        x[:,14]=filedata[:,16]
    if Nenc>0:
        x[:,15:(15+Nenc)]=map(LCs.features,filedata[:,15])
    y=filedata[:,tidx]
    return x,y[:,None]

def PrintPredictions(filename,x,yt,yp,method='forest',sep='\t'):
    (nd,nin) = x.shape
    ofp=open(filename,'w')
    for i in range(nd):
        for j in range(nin):
            ofp.write(repr(x[i][j])+sep)
        if method=='forest':
            ofp.write(repr(yt[i])+sep+repr(yp[i][1])+'\n')
        if method=='svm':
            ofp.write(repr(yt[i])+sep+repr(yp[i])+'\n')
    ofp.close()

def PrintPredictionsKS(filename,z,yt,yp,sep=' '):
    nd = z.shape[0]
    ofp=open(filename,'w')
    for i in range(nd):
        ofp.write(repr(z[i])+sep+repr(yt[i])+sep+repr(yp[i][1])+'\n')
    ofp.close()

def PrintPredictionsKSNN(filename,z,yt,yp,sep=' '):
    nd = z.shape[0]
    ofp=open(filename,'w')
    for i in range(nd):
        ofp.write(repr(z[i])+sep+repr(yt[i])+sep+repr(yp[i])+'\n')
    ofp.close()

def getNames(N):
    names = ['log_L', 'z', 'det_r', 'det_phi', 'bin_size_emit', 'alpha', 'beta', 'E_peak', 'bgd_15-25keV', 'bgd_15-50keV', \
    'bgd_25-100keV', 'bgd_50-350keV', 'theta', 'flux', 'ndet', 'encLC1', 'encLC2', 'encLC3', 'encLC4', 'encLC5', 'encLC6', \
    'encLC7', 'encLC8', 'encLC9', 'encLC10']
    return names[0:N]

def PrintData(fname,x,y):
    ofp = open(fname,'w')
    (nd,np) = x.shape
    ofp.write(repr(np)+',\n2,\n')
    for i in range(nd):
        for j in range(np):
            ofp.write(repr(x[i,j])+',')
        ofp.write('\n'+repr(int(y[i]))+',\n')
    ofp.close()

# Obtained from https://jmetzen.github.io/2015-01-29/ml_advice.html
# Modified from http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html
def plot_learning_curve(estimator, X, y, title=None, xlim=None, ylim=None, cv=5, train_sizes=np.linspace(.1, 1.0, 10), verbose=0, jobs=4, fname=None):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects
    """

    train_sizes, train_scores, test_scores = learning_curve.learning_curve(estimator, X, y, cv=cv, n_jobs=jobs, train_sizes=train_sizes, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(1)
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.25, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.25, color="b")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Train")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="b", label="Test")

    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right", prop={'size':12})
    ax.set_xscale("log")
    ax.xaxis.set_ticks([1e3,3e3,1e4,3e4,1e5])
    ax.xaxis.set_ticklabels([r'$10^3$',r'$3\times10^3$',r'$10^4$',r'$3\times10^4$',r'$10^5$'])
    ax.grid("on")
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    if title:
        ax.set_title(title)
    #plt.show()
    if fname:
        fig.savefig(fname, bbox_inches='tight', pad_inches=0.05, dpi=200)

# plots the ROC curve
def plot_ROC(ytrue,ypred,title=None,fname=None):
    unif = np.logspace(-5,0,num=100)

    fig, ax = plt.subplots(1)
    fpr, tpr, pth = metrics.roc_curve(ytrue, ypred, pos_label=1)
    ax.semilogx(fpr, tpr, '-', color="k")
    ax.semilogx(unif,unif,'--',color="r")
    ax.grid("on")
    if title:
        ax.set_title(title)
    plt.show()
    if fname:
        fig.savefig(fname)

def scoreModel(ytrue,ypred,pth=0.5):
    correct = 0.0
    nd = ytrue.shape[0]
    for i in range(nd):
        if ytrue[i]==1.0 and ypred[i]>=pth:
            correct += 1.0
        elif ytrue[i]==0.0 and ypred[i]<pth:
            correct += 1.0
    return correct/nd
