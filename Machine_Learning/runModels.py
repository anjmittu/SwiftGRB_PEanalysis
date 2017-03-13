from MLutils import *

from optparse import OptionParser
import numpy as np
from sklearn import tree,preprocessing,grid_search,cross_validation,ensemble
import os
import pickle
import math

parser=OptionParser()

parser.add_option("--RF",action="store_true",default=False,help="train Random Forest classifier")
parser.add_option("--RF2",action="store_true",default=False,help="train Random Forest classifier")
parser.add_option("--RF3",action="store_true",default=False,help="train Random Forest classifier")
parser.add_option("--AB",action="store_true",default=False,help="train AdaBoost classifier")
parser.add_option("--DT",action="store_true",default=False,help="train Decision Tree classifier")
parser.add_option("--SVM",action="store_true",default=False,help="train Support Vector Machines classifier")
parser.add_option("--LC",action="store_true",default=False,help="plot learning curves")
parser.add_option("--NN",action="store_true",default=False,help="analyze NN predictions for best settings")
parser.add_option("--jobs",action="store",type=int,default=4,help="number of parallel jobs to run")
parser.add_option("--verbose",action="store",type=int,default=1,help="verbose level")
(opts,args)=parser.parse_args()

def writeNetwork(nh,act):
	if len(nh)==1:
		nhid = repr(nh[0])
	else:
		nhid = repr(nh[0])+'-'+repr(nh[1])
	activ = repr(act)*len(nh)+'0'
	return [nhid,activ]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Universal data reading
#---------------------------------------------------------------

# read in prior training data
#filename = "newdata/Swift_train_all.txt"
filename = "newdata/Swift_new_data.txt"
xall,yall = readdata(filename,0)

# shuffle the data
ndata = xall.shape[0]
nidx = np.linspace(0,ndata-1,num=ndata,dtype=int)
np.random.seed(657948)
np.random.shuffle(nidx)
xall = xall[nidx,:]
yall = yall[nidx]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Random Forests
#---------------------------------------------------------------

if opts.RF:
	# setup RF grid search parameters
	nsplits = [8,16,32,64]
	nfeats = [3,4,5,6,7,8,9,10,11,12,13,14]
	grid_pars1 = {'min_samples_split':nsplits, 'max_features':nfeats}

	# setup RF classifier and train and evaluate on test
	Ntrees = 500
	basemod1 = ensemble.RandomForestClassifier(n_estimators=Ntrees)
	RF = grid_search.GridSearchCV(basemod1,grid_pars1,verbose=opts.verbose,cv=5,n_jobs=opts.jobs)
	RF.fit(xall,yall)
	print '\nRandom Forest cross-validation best params: ',RF.best_params_
	print 'Score = ',RF.score(xall,yall),'\n'

	savefile1a = open('RandomForest_grid.pkl','w')
	savefile1b = open('RandomForest_best.pkl','w')
	pickle.dump(RF,savefile1a)
	pickle.dump(RF.best_estimator_,savefile1b)
	savefile1a.close()
	savefile1b.close()

# finish the grid in nsplits
if opts.RF2:
	# setup RF grid search parameters
	nsplits = [2, 4]
	nfeats = [3,4,5,6,7,8,9,10,11,12,13,14]
	grid_pars1 = {'min_samples_split':nsplits, 'max_features':nfeats}

	# setup RF classifier and train and evaluate on test
	Ntrees = 500
	basemod1 = ensemble.RandomForestClassifier(n_estimators=Ntrees)
	RF = grid_search.GridSearchCV(basemod1,grid_pars1,verbose=opts.verbose,cv=5,n_jobs=opts.jobs)
	RF.fit(xall,yall)
	print '\nRandom Forest cross-validation best params: ',RF.best_params_
	print 'Score = ',RF.score(xall,yall),'\n'

	#savefile1a = open('RandomForest_grid2.pkl','w')
	#savefile1b = open('RandomForest_best2.pkl','w')
	#pickle.dump(RF,savefile1a)
	#pickle.dump(RF.best_estimator_,savefile1b)
	#savefile1a.close()
	#savefile1b.close()

# use optimal nsplits and nfeats from before
if opts.RF3:
	# setup RF grid search parameters
	ntrees = range(50,501,50)
	grid_pars1 = {'n_estimators':ntrees}

	# setup RF classifier and train and evaluate on test
	basemod1 = ensemble.RandomForestClassifier(min_samples_split=4, max_features=5)
	RF = grid_search.GridSearchCV(basemod1,grid_pars1,verbose=opts.verbose,cv=5,n_jobs=opts.jobs,
		pre_dispatch='2*n_jobs')
	RF.fit(xall,yall)
	print '\nRandom Forest cross-validation best params: ',RF.best_params_
	print 'Score = ',RF.score(xall,yall),'\n'

	#savefile1a = open('RandomForest_grid3.pkl','w')
	#savefile1b = open('RandomForest_best3.pkl','w')
	#pickle.dump(RF,savefile1a)
	#pickle.dump(RF.best_estimator_,savefile1b)
	#savefile1a.close()
	#savefile1b.close()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Decision Tree (for example)
#---------------------------------------------------------------

if opts.DT:
	np.random.seed(None)
	# train and print example tree
	DT = tree.DecisionTreeClassifier(min_samples_split=4,max_features=5,max_depth=3)
	DT.fit(xall,yall)
	print '\nExample tree score = ',DT.score(xall,yall),'\n'
	tree_out = tree.export_graphviz(DT,out_file="tree.dot",feature_names=getNames(15))
	os.system('dot -Tpng tree.dot -o tree.png')

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# AdaBoost
#---------------------------------------------------------------

if opts.AB:
	# setup AB grid search parameters
	nestim = [100,200,300,400,500]
	learnrate = np.logspace(-3.0,0.0,num=7,base=10.0)
	grid_pars2 = {'n_estimators':nestim, 'learning_rate':learnrate}

	# setup AB classifier and train and evaluate on test
	basemod2 = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_features=5,
		min_samples_split=4))
	AB = grid_search.GridSearchCV(basemod2,grid_pars2,verbose=opts.verbose,cv=5,n_jobs=opts.jobs,
		pre_dispatch='2*n_jobs')
	AB.fit(xall,yall)
	print '\nAdaBoost cross-validation best params: ',AB.best_params_
	print 'Score = ',AB.score(xall,yall),'\n'

	#savefile2a = open('AdaBoost_grid.pkl','w')
	#savefile2b = open('AdaBoost_best.pkl','w')
	#pickle.dump(AB,savefile2a)
	#pickle.dump(AB.best_estimator_,savefile2b)
	#savefile2a.close()
	#savefile2b.close()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Support Vector Machines
#---------------------------------------------------------------

if opts.SVM:
	# scale data (for SVM)
    if os.path.exists('SVM_scaler.pkl'):
        fp3 = open('SVM_scaler.pkl','r')
        scaler = pickle.load(fp3)
        fp3.close()
        xall2 = scaler.transform(xall)
    else:
        scaler = preprocessing.StandardScaler()
        xall2 = scaler.fit_transform(xall)

    # setup hyperparameter search
    csroot = np.logspace(-2.0,8.0,num=11,base=10.0)
    gsroot = np.logspace(-5.0,3.0,num=9,base=10.0)
    if opts.part==None:
    	cs = csroot
    	gs = gsroot
    else:
    	nc = np.mod(opts.part,11)
    	ng = (opts.part - nc)/11
    	cs = csroot[nc:nc+1]
    	gs = gsroot[3*ng:3*ng+3]
    grid_pars3 = {'C':cs, 'gamma':gs}
    print cs,gs,grid_pars3

    # train SVM classifier
    basemod3 = svm.SVC(kernel='rbf', probability=True)
    SVM = grid_search.GridSearchCV(basemod3,grid_pars3,verbose=opts.verbose,cv=5,n_jobs=opts.jobs,
    	pre_dispatch='2*n_jobs')
    SVM.fit(xall2, yall)
    print '\nSVM cross-validation best params: ',SVM.best_params_
    print 'Score = ',SVM.score(xall2,yall),'\n'

    if opts.part==None:
        savefile3a = open('SVM_grid.pkl','w')
        savefile3b = open('SVM_best.pkl','w')
    else:
        savefile3a = open('SVM_grid'+repr(opts.part)+'.pkl','w')
        savefile3b = open('SVM_best'+repr(opts.part)+'.pkl','w')
    #pickle.dump(SVM,savefile3a)
    #pickle.dump(SVM.best_estimator_,savefile3b)
    #savefile3a.close()
    #savefile3b.close()
    #if not(os.path.exists('SVM_scaler.pkl')):
    #    savefile3c = open('SVM_scaler.pkl','w')
    #    pickle.dump(scaler,savefile3c)
    #    savefile3c.close()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Learning Curves
#---------------------------------------------------------------

if opts.LC:
	rfclf = ensemble.RandomForestClassifier(n_estimators=500, max_features=5, min_samples_split=4)
	#abclf = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_features=8,
	#	min_samples_split=8),n_estimators=300, learning_rate=0.0021544346900318843)
	plot_learning_curve(rfclf, xall, np.squeeze(yall), "Random Forest Learning Curve", xlim=(1e3,1.2e5),
		train_sizes=np.logspace(-2,0,num=10), cv=5, verbose=opts.verbose, jobs=opts.jobs,
		fname="RFlearncurve_new.png")
	#plot_learning_curve(abclf, "AdaBoost Learning Curve", xall, yall, xlim=(1e3,1.2e5),
	#	train_sizes=np.logspace(-2,0,num=10), verbose=opts.verbose, fname="ABlearncurve.png")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Neural Networks
#---------------------------------------------------------------

if opts.NN:
	nhid=[[25],[50],[100],[1000],[25,25],[50,50],[100,30],[100,50],[100,100]]
	activ=[1,3]
	bestnet = None
	bestscore = 0
	nets = []
	for nh in nhid:
		for act in activ:
			nets.append(writeNetwork(nh,act))
	for net in nets:
		filename = 'NNmodels/predictions/Swift_NN_CVall_nhid-'+net[0]+'_act'+net[1]+'_eval_pred.txt'
		data = np.loadtxt(filename,usecols=(16,18))
		score = scoreNN(data[:0],data[:,1])
		print net[0],'\t',net[1],'\t',score
		if score > bestscore:
			bestnet = net
			bestscore = score
	print 'Best NN settings: nhid=',bestnet[0],'\tact=',bestnet[1],'\taccuracy=',bestscore
