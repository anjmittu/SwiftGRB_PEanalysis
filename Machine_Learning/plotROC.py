from MLutils import *
import sys
import pickle
import numpy as np
from sklearn import tree,ensemble, cross_validation, metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

Nenc = 0
Ntrees = 500
colors = ["k","b","g","m","y"]
unif = np.logspace(-5,0,num=100)

fig, ax = plt.subplots(1)

d = np.loadtxt("RandomForest_predictionsKSold.txt")
fpr, tpr, pth = metrics.roc_curve(d[:,1], d[:,2], pos_label=1)
ax.plot(fpr, tpr, '-', color=colors[0], label="Just old", lw=2)

d = np.loadtxt("RandomForest_predictionsKSnew.txt")
fpr, tpr, pth = metrics.roc_curve(d[:,1], d[:,2], pos_label=1)
ax.plot(fpr, tpr, '-', color=colors[1], label="Just new", lw=2)

d = np.loadtxt("RandomForest_predictionsKSmix.txt")
fpr, tpr, pth = metrics.roc_curve(d[:,1], d[:,2], pos_label=1)
ax.plot(fpr, tpr, '--', color=colors[2], label="Mixed", lw=2)

ipf = np.argmin(np.abs(pth-0.5))
#ax1.plot(fpr[ipf], tpr[ipf], 'o', color=colors[i], ms=8)

precision, recall, pth2 = metrics.precision_recall_curve(d[:,1], d[:,2], pos_label=1)
f1score = 2.0 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1])
#ax.plot(pth2, f1score, '-', color=colors[1])

imx = np.argmax(f1score)

#ax.plot(fpr[imx], tpr[imx], 'o', color=colors[2], ms=8)

AUC = metrics.roc_auc_score(d[:,1],d[:,2])
acc = metrics.accuracy_score(d[:,1]==1.0,d[:,2]>=pth2[imx])
#ksd, ksp = stats.ks_2samp(d[d[:,1]==1.0,0],d[d[:,2]>=pth2[imx],0])

#print "%s\tAUC=%0.4f\t(TPR,FPR)=(%0.4f,%0.4f) at pth=0.5\tF1=%0.4f at pth=%0.4f"%(clf,AUC,tpr[ipf],fpr[ipf],f1score[imx],pth2[imx])
#print "AUC=%0.4f\t(TPR,FPR,F1)=(%0.4f,%0.4f,%0.4f) at pth=%0.4f"%(AUC,tpr[imx],fpr[imx],f1score[imx],pth2[imx])
#print "AUC=%0.3f\tAcc=%0.3f\tF1-score=%0.3f\n"%(AUC,acc,f1score[imx])

ax.plot(unif,unif,'--',color="r",lw=2)

ax.set_xscale("log")
ax.legend(loc="best", prop={'size':12})
ax.grid("on")
ax.set_xlabel("False Positive Rate", fontsize=16)
ax.set_ylabel("True Positive Rate", fontsize=16)
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.025))

plt.show()
fig.savefig("modelROCs.png", bbox_inches='tight', pad_inches=0.05, dpi=200)
