import numpy as np
import random as ran
import triangle
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def Rz(z,n):
    n0 = n[0]
    n1 = n[1]
    n2 = n[2]
    n3 = n[3]
    z1 = n[4]
    z2 = n[5]
    R = np.piecewise(z, [z<=z1,(z<=z2) & (z>z1), z>z2], [lambda x: n0*np.power((1+x),n1),
                                                  lambda x: n0*np.power((1+x),n2)*np.power((1+z1),n1-n2),
                                                  lambda x: n0*np.power((1+x),n3)*np.power((1+z1),n1-n2)*np.power((1+z2),n2-n3)])
    return R

def createCube():
    n0 = np.linspace(.01,2)
    n1 = np.linspace(0,4)
    n2 = np.linspace(-6,6)
    n3 = np.linspace(-10,0)
    z1 = np.linspace(0,10)
    z2 = np.linspace(0,10)
    
    sample = []
    for x in range(0,50):
        x0 = ran.randrange(0,50)
        x1 = ran.randrange(0,50)
        x2 = ran.randrange(0,50)
        x3 = ran.randrange(0,50)
        x4 = ran.randrange(0,50)
        x5 = ran.randrange(0,50)
        
        sample.append([n0[x0], n1[x1], n2[x2], n3[x3], z1[x4], z2[x5]])

    return sample

def plotRzPost(data, N, maxlike, zdatafile, outfile, annotate = False, title = None):
    zdata = np.loadtxt(zdatafile)
    factor = 4.0 * np.pi / 6.0 * 0.8 * dV / (1.0 + z)
    fig, ax = plt.subplots(2,1, figsize=(6,8))
    fig.subplots_adjust(hspace=0.05)
    hist_n, hist_bin, hist_patch = ax[1].hist(zdata, bins=Nbins, range=[0,10])
    dz = 10.0 / Nbins
    ax[1].cla()
    for i in range(N):
        n = data[i]
        if i==0:
            ax[0].plot(z, Rz(z,n), '-b', lw=1, alpha=0.1, label='posterior sample')
        else:
            ax[0].plot(z, Rz(z,n), '-b', lw=1, alpha=0.1)
        ax[1].plot(z, Rz(z,n) * factor * df, '-b', lw=1, alpha=0.1)
    ax[0].plot(z, Rz(z,maxlike), '-k', lw=2, label='max likelihood')
    ax[1].plot(z, Rz(z,maxlike) * factor, '--k', lw=2, label=r'$N_{\mathrm{int}}$')
    ax[1].plot(z, Rz(z,maxlike) * factor * df, '-k', lw=2, label=r'$N_{\mathrm{exp}}$')
    ax[1].bar(hist_bin[:-1], hist_n/dz, width=dz, color='red', alpha=0.5, label=r'$N_{\mathrm{obs}}$')
    #ax[0].set_xlabel('Redshift')
    ax[1].set_xlabel('Redshift')
    ax[0].set_ylabel(r'$R_{\mathrm{GRB}}(z)$')
    ax[1].set_ylabel(r'$N(z)/dz$')
    ax[0].grid('on')
    ax[1].grid('on')
    ax[0].legend(loc='upper left', prop={'size':10})
    ax[1].legend(loc='upper right')
    ax[0].set_xlim([0,10])
    ax[1].set_xlim([0,10])
    ax[0].set_ylim([0,40])
    #ax[1].set_ylim([0,40])
    ax[1].set_ylim([0,(int(np.max(Rz(z,maxlike) * factor))/10 + 1)*10])
    ax[0].xaxis.set_major_locator(MultipleLocator(1))
    ax[1].xaxis.set_major_locator(MultipleLocator(1))
    ax[0].xaxis.set_minor_locator(MultipleLocator(0.25))
    ax[1].xaxis.set_minor_locator(MultipleLocator(0.25))
    ax[0].yaxis.set_major_locator(MultipleLocator(5))
    #ax[1].yaxis.set_major_locator(MultipleLocator(5))
    ax[1].yaxis.set_major_locator(MultipleLocator(20))
    ax[0].yaxis.set_minor_locator(MultipleLocator(1))
    #ax[1].yaxis.set_minor_locator(MultipleLocator(1))
    ax[1].yaxis.set_minor_locator(MultipleLocator(5))
    ax[0].set_xticklabels([])
    if annotate:
        ax[0].annotate(r'$n_0$', xy = (0.05,0.5), xytext = (0.25,7), xycoords = 'data', textcoords = 'data',
            arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        ax[0].annotate(r'$n_1$', xy = (2.5,6), xytext = (1.5,12), xycoords = 'data', textcoords = 'data',
            arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        ax[0].annotate(r'$z_1$', xy = (5.2,18), xytext = (4.25,22), xycoords = 'data', textcoords = 'data',
            arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        ax[0].annotate(r'$n_2$', xy = (6.25,7), xytext = (7.25,16), xycoords = 'data', textcoords = 'data',
            arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    if title:
        ax[0].set_title(title)
    fig.savefig(outfile, bbox_inches='tight', pad_inches=0.05, dpi=200)
    #plt.show()

def maxlikeplot(maxlike):
    factor = 4.0 * np.pi / 6.0 * 0.8 * dV / (1.0 + z)
    fig, ax = plt.subplots(2,1, figsize=(6,8))
    fig.subplots_adjust(hspace=0.05)
    dz = 10.0 / Nbins
    ax[1].cla()
    ax[0].plot(z, Rz(z,maxlike), '-k', lw=2, label='max likelihood')
    ax[1].plot(z, Rz(z,maxlike) * factor, '--k', lw=2, label=r'$N_{\mathrm{int}}$')
    ax[1].plot(z, Rz(z,maxlike) * factor * df, '-k', lw=2, label=r'$N_{\mathrm{exp}}$')
    ax[1].set_xlabel('Redshift')
    ax[0].set_ylabel(r'$R_{\mathrm{GRB}}(z)$')
    ax[1].set_ylabel(r'$N(z)/dz$')
    ax[0].grid('on')
    ax[1].grid('on')
    ax[0].legend(loc='upper left', prop={'size':10})
    ax[1].legend(loc='upper right')
    ax[0].set_xlim([0,10])
    ax[1].set_xlim([0,10])
    ax[0].set_ylim([0,40])
    fig.savefig('./Demo_max', bbox_inches='tight', pad_inches=0.05, dpi=200)


detfrac_data = np.loadtxt('support_data/splines_detection_fraction_z_RF.txt')
detfrac = interp1d(detfrac_data[:,0], detfrac_data[:,1], kind='linear')
Ez_data = np.loadtxt('support_data/splines_Ez.txt', usecols=(0,3))
Ez = interp1d(Ez_data[:,0], Ez_data[:,1], kind='linear')
    
maxlike = [1,1.68,-2.7,-.5,3.5,6,9966.27]
N = 50
Nbins = 20
z = np.linspace(0,10,num=2001)
df = gaussian_filter1d(detfrac(z), 10)
dV = 75.28129176631614 * Ez(z)

sample = createCube()

plotRzPost(sample, N, maxlike,
           'support_data/FynboGRB_lum_z_Zonly.txt',
           './Demo_redshift_distribution_posterior_RF_bestfit.png',
           title = 'Real Data')

#maxlikeplot(maxlike)
