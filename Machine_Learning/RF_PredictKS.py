from MLutils import *
import sys
from sklearn import tree,ensemble, cross_validation, metrics
import random as r
import numpy as np

Nenc = 0
Ntrees = 500

filenameold = "newdata/Swift_train_all.txt"
xallold,yallold = readdata(filenameold,0)

filenamenew = "newdata/log_summary_info_Swiftlc_z360_lum5205_n0042_n1144_n2084_alpha065_beta300_Yonetoku_mod18_noevo_ndet26884_Machine_Learning.txt"
xallnew,yallnew = readdata(filenamenew,0)

filenamemix = "newdata/Swift_new_data3.txt"
xallmix,yallmix = readdata(filenamemix,0)

ndata = xallmix.shape[0]
nidx = np.linspace(0,ndata-1,num=ndata,dtype=int)
np.random.seed(65048)
np.random.shuffle(nidx)
xallmixR = xallmix[nidx,:]
yallmixR = yallmix[nidx]
ndata = xallold.shape[0]
nidx = np.linspace(0,ndata-1,num=ndata,dtype=int)
np.random.seed(65048)
np.random.shuffle(nidx)
xalloldR = xallold[nidx,:]
yalloldR = yallold[nidx]
ndata = xallnew.shape[0]
nidx = np.linspace(0,ndata-1,num=ndata,dtype=int)
np.random.seed(65048)
np.random.shuffle(nidx)
xallnewR = xallnew[nidx,:]
yallnewR = yallnew[nidx]

RF1 = ensemble.RandomForestClassifier(n_estimators=500,min_samples_split=8,max_features=9)
RF1.fit(xallmixR[10000:,:],yallmixR[10000:])
print 'Score of mix: ', RF1.score(xallmixR[10000:,:],yallmixR[10000:]),'\n'

RF2 = ensemble.RandomForestClassifier(n_estimators=500,min_samples_split=8,max_features=3)
RF2.fit(xalloldR[10000:,:],yalloldR[10000:])
print 'Score of old: ', RF2.score(xalloldR[10000:,:],yalloldR[10000:]),'\n'

RF3 = ensemble.RandomForestClassifier(n_estimators=500,min_samples_split=16,max_features=9)
RF3.fit(xallnewR[1000:,:],yallnewR[1000:])
print 'Score of new: ', RF3.score(xallnewR[1000:,:],yallnewR[1000:]),'\n'


ypredold1 = RF1.predict_proba(xalloldR[0:10000,:])
yprednew1 = RF1.predict_proba(xallnewR[0:1000,:])
ypredmix1 = RF1.predict_proba(xallmixR[0:10000,:])

ypredold2 = RF2.predict_proba(xalloldR[0:10000,:])
yprednew2 = RF2.predict_proba(xallnewR[0:1000,:])
ypredmix2 = RF2.predict_proba(xallmixR[0:10000,:])

ypredold3 = RF3.predict_proba(xalloldR[0:10000,:])
yprednew3 = RF3.predict_proba(xallnewR[0:1000,:])
ypredmix3 = RF3.predict_proba(xallmixR[0:10000,:])

PrintPredictionsKS('RandomForest_predictionsKS_trainmix3_testold.txt',xalloldR[0:10000,1],yalloldR[0:10000],ypredold1)
PrintPredictionsKS('RandomForest_predictionsKS_trainmix3_testnew.txt',xallnewR[0:1000,1],yallnewR[0:1000],yprednew1)
PrintPredictionsKS('RandomForest_predictionsKS_trainmix3_testmix3.txt',xallmixR[0:10000,1],yallmixR[0:10000],ypredmix1)

PrintPredictionsKS('RandomForest_predictionsKS_trainold_testold.txt',xalloldR[0:10000,1],yalloldR[0:10000],ypredold2)
PrintPredictionsKS('RandomForest_predictionsKS_trainold_testnew.txt',xallnewR[0:1000,1],yallnewR[0:1000],yprednew2)
PrintPredictionsKS('RandomForest_predictionsKS_trainold_testmix3.txt',xallmixR[0:10000,1],yallmixR[0:10000],ypredmix2)

PrintPredictionsKS('RandomForest_predictionsKS_trainnew_testold.txt',xalloldR[0:10000,1],yalloldR[0:10000],ypredold3)
PrintPredictionsKS('RandomForest_predictionsKS_trainnew_testnew.txt',xallnewR[0:1000,1],yallnewR[0:1000],yprednew3)
PrintPredictionsKS('RandomForest_predictionsKS_trainnew_testmix3.txt',xallmixR[0:10000,1],yallmixR[0:10000],ypredmix3)
