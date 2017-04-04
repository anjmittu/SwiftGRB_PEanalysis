from MLutils import *
import sys
from sklearn import tree,ensemble, cross_validation, metrics

Nenc = 0
Ntrees = 500

filenameold = "newdata/Swift_train_all.txt"
xallold,yallold = readdata(filenameold,0)

filenamenew = "newdata/log_summary_info_Swiftlc_z360_lum5205_n0042_n1144_n2084_alpha065_beta300_Yonetoku_mod18_noevo_ndet26884_Machine_Learning.txt"
xallnew,yallnew = readdata(filenamenew,0)

filenamemix = "newdata/Swift_new_data.txt"
xallmix,yallmix = readdata(filenamemix,0)

xtrain,xtest,ytrain,ytest = cross_validation.train_test_split(xallmix,yallmix,test_size=0.20)

RF = ensemble.RandomForestClassifier(n_estimators=500,min_samples_split=8,max_features=13)
RF.fit(xtrain,ytrain)

score = RF.score(xtest,ytest)
ypredold = RF.predict_proba(xallold)
yprednew = RF.predict_proba(xallnew)
ypredmix = RF.predict_proba(xallmix)

print 'Random Forest testing = ',score
PrintPredictionsKS('RandomForest_predictionsKSold.txt',xallold[:,1],yallold,ypredold)
PrintPredictionsKS('RandomForest_predictionsKSnew.txt',xallnew[:,1],yallnew,yprednew)
PrintPredictionsKS('RandomForest_predictionsKSmix.txt',xallmix[:,1],yallmix,ypredmix)
