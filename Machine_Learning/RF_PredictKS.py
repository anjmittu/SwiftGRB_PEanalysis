from MLutils import *
import sys
from sklearn import tree,ensemble, cross_validation, metrics
import random as r
import numpy as np

Nenc = 0
Ntrees = 500

filenameold = "newdata/Swift_train_all.txt"
xallold,yallold = readdata(filenameold,0)

#filenamenew = "newdata/log_summary_info_Swiftlc_z360_lum5205_n0042_n1144_n2084_alpha065_beta300_Yonetoku_mod18_noevo_ndet26884_Machine_Learning.txt"
filenamenew = "newdata/newDataCombined.txt"
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

xnew_train = xallnewR[1000:,:]
ynew_train = yallnewR[1000:]
xnew_test = xallnewR[0:1000,:]
ynew_test = yallnewR[0:1000]
xold_train = xalloldR[10000:,:]
yold_train = yalloldR[10000:]
xold_test = xalloldR[0:10000,:]
yold_test = yalloldR[0:10000]
xmix_train = np.concatenate((xnew_train, xold_train))
ymix_train = np.concatenate((ynew_train, yold_train))
xmix_test = np.concatenate((xnew_test, xold_test))
ymix_test = np.concatenate((ynew_test, yold_test))

RF1 = ensemble.RandomForestClassifier(n_estimators=500,min_samples_split=8,max_features=9)
RF1.fit(xmix_train,ymix_train)
print 'Score of mix: ', RF1.score(xmix_test,ymix_test),'\n'

RF2 = ensemble.RandomForestClassifier(n_estimators=500,min_samples_split=8,max_features=3)
RF2.fit(xold_train,yold_train)
print 'Score of old: ', RF2.score(xold_test,yold_test),'\n'

RF3 = ensemble.RandomForestClassifier(n_estimators=500,min_samples_split=16,max_features=9)
RF3.fit(xnew_train,ynew_train)
print 'Score of new: ', RF3.score(xnew_test,ynew_test),'\n'


ypredold1 = RF1.predict_proba(xold_test)
yprednew1 = RF1.predict_proba(xnew_test)
ypredmix1 = RF1.predict_proba(xmix_test)

ypredold2 = RF2.predict_proba(xold_test)
yprednew2 = RF2.predict_proba(xnew_test)
ypredmix2 = RF2.predict_proba(xmix_test)

ypredold3 = RF3.predict_proba(xold_test)
yprednew3 = RF3.predict_proba(xnew_test)
ypredmix3 = RF3.predict_proba(xmix_test)

PrintPredictionsKS('RandomForest_predictionsKS_trainmix3_testold.txt', xold_test[:,1],yold_test,ypredold1)
PrintPredictionsKS('RandomForest_predictionsKS_trainmix3_testnew.txt', xnew_test[:,1],ynew_test,yprednew1)
PrintPredictionsKS('RandomForest_predictionsKS_trainmix3_testmix3.txt', xmix_test[:,1],ymix_test,ypredmix1)

PrintPredictionsKS('RandomForest_predictionsKS_trainold_testold.txt',xold_test[:,1],yold_test,ypredold2)
PrintPredictionsKS('RandomForest_predictionsKS_trainold_testnew.txt',xnew_test[:,1],ynew_test,yprednew2)
PrintPredictionsKS('RandomForest_predictionsKS_trainold_testmix3.txt',xmix_test[:,1],ymix_test,ypredmix2)

PrintPredictionsKS('RandomForest_predictionsKS_trainnew_testold.txt',xold_test[:,1],yold_test,ypredold3)
PrintPredictionsKS('RandomForest_predictionsKS_trainnew_testnew.txt',xnew_test[:,1],ynew_test,yprednew3)
PrintPredictionsKS('RandomForest_predictionsKS_trainnew_testmix3.txt',xmix_test[:,1],ymix_test,ypredmix3)
