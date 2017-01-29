from MLutils import *

from optparse import OptionParser
parser=OptionParser()

parser.add_option("--RF",action="store_true",default=False,help="train Random Forest classifier grid")
parser.add_option("--RF2",action="store_true",default=False,help="train Random Forest classifier")
parser.add_option("--AB",action="store_true",default=False,help="train AdaBoost classifier")
parser.add_option("--SVM",action="store_true",default=False,help="train Support Vector Machines classifier")
parser.add_option("--NN",action="store_true",default=False,help="analyze neural network predictions for best settings")
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
filename = "newdata/Swiftv2_train_all.txt"
dall = np.loadtxt(filename)
xall = dall[:,:-1]
yall = dall[:,-1]
filename = "newdata/Swiftv2_validate_all.txt"
dval = np.loadtxt(filename)
xval = dval[:,:-1]
yval = dval[:,-1]
filename = "newdata/Swift_validate_all.txt"
xval2,yval2 = readdata(filename, ndet=False)
print 'Data loaded.'

# shuffle the data
ndata = xall.shape[0]
nidx = np.linspace(0,ndata-1,num=ndata,dtype=int)
np.random.seed(65048)
np.random.shuffle(nidx)
xall = xall[nidx,:]
yall = yall[nidx]
print 'Data shuffled.'

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Random Forests
#---------------------------------------------------------------

if opts.RF:
	# setup grid search RF parameters
	nsplits = [4, 8, 12, 16, 20, 24, 28, 32]
	depth = [6, 8, 10, 12, 14, 16]
	nfeats = 5
	Ntrees = 500
	grid_pars = {'max_depth': depth, 'min_samples_split': nsplits}

	# create and fit the model
	basemod = ensemble.RandomForestClassifier(n_estimators=Ntrees, max_features=nfeats)
	RF = grid_search.GridSearchCV(basemod, grid_pars, verbose=2, cv=5, n_jobs=-1, pre_dispatch='2*n_jobs')
	RF.fit(xall,yall)
	for estim in RF.grid_scores_:
		print estim.parameters, estim.mean_validation_score
	print '\nRandom Forest cross-validation best params: ',RF.best_params_
	print 'RF score on train data           =', RF.best_estimator_.score(xall,yall)
	print 'RF score on validation data      =', RF.best_estimator_.score(xval,yval)
	print 'RF score on full validation data =', RF.best_estimator_.score(xval2,yval2)

	#savefile1a = open('RandomForest_v2_fit.pkl','w')
	#pickle.dump(RF.best_estimator_,savefile1a)
	#savefile1a.close()
	#savefile1b = open('RandomForest_v2_grid.pkl','w')
	#pickle.dump(RF,savefile1b)
	#savefile1b.close()

if opts.RF2:
	# setup optimal RF parameters
	nsplits = 4
	depth = None
	nfeats = 5
	Ntrees = 500

	# create and fit the model
	RF = ensemble.RandomForestClassifier(n_estimators=Ntrees, max_features=nfeats, max_depth=depth,
		min_samples_split=nsplits, verbose=2, n_jobs=-1)
	RF.fit(xall,yall)
	print 'RF score on train data           =', RF.score(xall,yall)
	print 'RF score on validation data      =', RF.score(xval,yval)
	print 'RF score on full validation data =', RF.score(xval2,yval2)

	#savefile1a = open('RandomForest_v2_fit.pkl','w')
	#pickle.dump(RF,savefile1a)
	#savefile1a.close()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# AdaBoost
#---------------------------------------------------------------

if opts.AB:
	# setup optimal AB parameters
	nestim = 500
	learnrate = 0.001
	max_features = 5
	min_samples_split = 4
	depth = None

	# create and fit the model
	AB = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_features=max_features,
		min_samples_split=min_samples_split, max_depth=depth), n_estimators=nestim, learning_rate=learnrate)
	AB.fit(xall,yall)
	print 'AdaBoost score on train data           =', AB.score(xall,yall)
	print 'AdaBoost score on validation data      =', AB.score(xval,yval)
	print 'AdaBoost score on full validation data =', AB.score(xval2,yval2)

	#savefile2 = open('AdaBoost_v2_fit.pkl','w')
	#pickle.dump(AB,savefile2)
	#savefile2.close()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Support Vector Machines
#---------------------------------------------------------------

if opts.SVM:
	# scale data (for SVM)
    if os.path.exists('SVM_v2_scaler.pkl'):
        fp3 = open('SVM_v2_scaler.pkl','r')
        scaler = pickle.load(fp3)
        fp3.close()
        xall2 = scaler.transform(xall)
    else:
        scaler = preprocessing.StandardScaler()
        xall2 = scaler.fit_transform(xall)
        savefile3c = open('SVM_v2_scaler.pkl','w')
        pickle.dump(scaler,savefile3c)
        savefile3c.close()

    # setup optimal hyperparameters
    C = 10.0**2.25
    gamma = 1.0

    # train SVM classifier
    SVM = svm.SVC(kernel='rbf', probability=True, C=C, gamma=gamma, verbose=True)
    SVM.fit(xall2, yall)
    print 'SVM score on train data           =', SVM.score(xall2,yall)
    print 'SVM score on validation data      =', SVM.score(scaler.transform(xval),yval)
    print 'SVM score on full validation data =', SVM.score(scaler.transform(xval2),yval2)

    #savefile3 = open('SVM_v2_fit.pkl','w')
    #pickle.dump(SVM,savefile3)
    #savefile3.close()

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
