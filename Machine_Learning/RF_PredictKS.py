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

RF1 = ensemble.RandomForestClassifier(n_estimators=500,min_samples_split=8,max_features=13)
RF1.fit(xallmix,yallmix)

RF2 = ensemble.RandomForestClassifier(n_estimators=500,min_samples_split=8,max_features=3)
RF2.fit(xallold,yallold)

RF3 = ensemble.RandomForestClassifier(n_estimators=500,min_samples_split=16,max_features=9)
RF3.fit(xallnew,yallnew)

ypredold1 = RF1.predict_proba(xallold)
yprednew1 = RF1.predict_proba(xallnew)
ypredmix1 = RF1.predict_proba(xallmix)

ypredold2 = RF2.predict_proba(xallold)
yprednew2 = RF2.predict_proba(xallnew)
ypredmix2 = RF2.predict_proba(xallmix)

ypredold3 = RF3.predict_proba(xallold)
yprednew3 = RF3.predict_proba(xallnew)
ypredmix3 = RF3.predict_proba(xallmix)

PrintPredictionsKS('RandomForest_predictionsKS_trainmix_testold.txt',xallold[:,1],yallold,ypredold1)
PrintPredictionsKS('RandomForest_predictionsKS_trainmix_testnew.txt',xallnew[:,1],yallnew,yprednew1)
PrintPredictionsKS('RandomForest_predictionsKS_trainmix_testmix.txt',xallmix[:,1],yallmix,ypredmix1)

PrintPredictionsKS('RandomForest_predictionsKS_trainold_testold.txt',xallold[:,1],yallold,ypredold2)
PrintPredictionsKS('RandomForest_predictionsKS_trainold_testnew.txt',xallnew[:,1],yallnew,yprednew2)
PrintPredictionsKS('RandomForest_predictionsKS_trainold_testmix.txt',xallmix[:,1],yallmix,ypredmix2)

PrintPredictionsKS('RandomForest_predictionsKS_trainnew_testold.txt',xallold[:,1],yallold,ypredold3)
PrintPredictionsKS('RandomForest_predictionsKS_trainnew_testnew.txt',xallnew[:,1],yallnew,yprednew3)
PrintPredictionsKS('RandomForest_predictionsKS_trainnew_testmix.txt',xallmix[:,1],yallmix,ypredmix3)
