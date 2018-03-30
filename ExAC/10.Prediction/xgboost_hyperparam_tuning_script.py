#Basic imports
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from os import environ,getcwd

#Classifier imports
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier

#ML framework imports
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, average_precision_score, make_scorer
from sklearn.model_selection import StratifiedKFold,train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import scale
import xgboost as xgb

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
from sklearn import metrics   #Additional scklearn functions
#from sklearn.grid_search import 

ABSOLUTE_NEGATIVES = False
FILTER_DOMAIN = False

def filter_to_ligand_binding_domains(use_max_binding_score):
    
    ligands_negatives_df = {}
    for ligand in ligands:
        
        ligands_negatives_df[ligand] = pd.DataFrame()
        for domain in negatives_dict[ligand].keys():
            if domain == 'negatives' or domain == 'domains':
                continue
            domain_all = features_all.loc[features_all.loc[:,"domain_name"] == domain,:]
            
            #In case this domain was previously filtered
            if len(domain_all) == 0:
                continue
            
            if (use_max_binding_score):
                ligands_negatives_df[ligand] = pd.concat([ligands_negatives_df[ligand],domain_all.loc[domain_all.loc[:,"max_binding_score"] == 0,:]])
            else:
                ligand_bind_str = ligand+"_binding_score"
                ligands_negatives_df[ligand] = pd.concat([ligands_negatives_df[ligand],domain_all.loc[domain_all.loc[:,ligand_bind_str] == 0,:]])
        
    #Handeling the ligand "all_ligands"
    all_ligands_negatives_df = pd.concat([ligands_negatives_df["dna"], ligands_negatives_df["dnabase"], ligands_negatives_df["dnabackbone"], ligands_negatives_df["rna"], ligands_negatives_df["rnabase"], 
                                 ligands_negatives_df["rnabackbone"], ligands_negatives_df["ion"], ligands_negatives_df["peptide"], ligands_negatives_df["metabolite"]])
    all_ligands_negatives_df = all_ligands_negatives_df.drop_duplicates()
    #Filter to just positions with max. binding score = 0
    all_ligands_negatives_df = all_ligands_negatives_df[all_ligands_negatives_df["max_binding_score"] == 0]
    ligands_negatives_df["all_ligands"] = all_ligands_negatives_df
    
    #Leaving just the features columns
    for ligand in ligands_negatives_df.keys():   
        ligands_negatives_df[ligand] = ligands_negatives_df[ligand][features_cols]
        print(ligand+" non-binding #:"+str(len(ligands_negatives_df[ligand])))
    
    return ligands_negatives_df

def negatives_by_binding_score(use_max_binding_score):
    
    ligands_negatives_df = {}
    for ligand in ligands:
        
        if use_max_binding_score:
            ligand_bind_str = "max_binding_score"
        else:
            ligand_bind_str = ligand+"_binding_score"
        
        ligands_negatives_df[ligand] = features_all[features_all[ligand_bind_str] == 0]
        ligands_negatives_df[ligand] = ligands_negatives_df[ligand].loc[:,features_cols]
        print(ligand+" non-binding #:"+str(len(ligands_negatives_df[ligand])))
        
    #Handeling the ligand "all_ligands"
    ligands_negatives_df["all_ligands"] = features_all[features_all["max_binding_score"] == 0]
    ligands_negatives_df["all_ligands"] = ligands_negatives_df["all_ligands"].loc[:,features_cols]
    print("all_ligands non-binding #:"+str(len(ligands_negatives_df["all_ligands"])))
    
    return ligands_negatives_df 

def modelfit(alg, ligand_bind_features, ligand_negatives_features, ligand_name, useTrainCV=True, cv_folds=10, early_stopping_rounds=50):
    
    features = np.ones([ligand_bind_features.shape[1],]).astype(bool)
    X = pd.concat([ligand_bind_features.iloc[:,features], ligand_negatives_features.iloc[:,features]])

    y = [1] * ligand_bind_features.shape[0]
    y.extend([0] * ligand_negatives_features.shape[0])
    y = np.array(y)
    
    print "modelfit"
    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(X, label=y)
    #print alg.get_params()['n_estimators']
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, 
                      metrics='auc', early_stopping_rounds=early_stopping_rounds)
    alg.set_params(n_estimators=cvresult.shape[0])
    print "Optimal n_estimators: " + str(cvresult.shape[0])
    
    #Fit the algorithm on the data
    #print "fitting"
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,test_size=0.25)
    #print X_train
    alg.fit(X_train, y_train,eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:,1]
    
    #Predict test set:
    #probs = alg.predict_proba(X_test)
    
    #Print model report:
    #print "\nModel Report"
    #auc_score = roc_auc_score(y_test, probs[:, 1])
    #print y_test
    #print probs[:, 1]
    #precision , recall, _ = precision_recall_curve(y_test, probs[:, 1])
    #auprc = auc(recall, precision)    

    #Print model report:
    print "\nModel Report"
    print "Accuracy(Train): %.4g" % metrics.accuracy_score(y_train, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob)
    #print "AUC (Test) = "+str(auc_score)
    #print "AUPRC (Test) = "+str(auprc)
    """               
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    """
    return alg,cvresult#,dtrain_predictions,dtrain_predprob,alg         


curr_dir = getcwd()
print curr_dir
input_path = curr_dir+"/domains_similarity/filtered_features_table/"
filename = "positions_features_mediode_filter_01.25.18.csv"

#input_path = curr_dir[0]+"/../9.Features_exploration/binding_df/10/"
#filename = "positions_features_01.25.18.csv"

bind_scores_num = 10

#Features table
features_all = pd.read_csv(input_path+filename, sep='\t', index_col=0)
features_cols = features_all.columns[1:-bind_scores_num] #removing binding scores and domain name
ligands = ["dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone", "peptide", "ion", "metabolite"]
print "all samples positions #: "+str(features_all.shape[0])

#lignd binding domains dictionary
with open(curr_dir+"/ligands_negatives_domains_dict.pik", 'rb') as handle:
        negatives_dict = pickle.load(handle)

#Create negatives datasets
if FILTER_DOMAIN:
    if ABSOLUTE_NEGATIVES:
        ligands_negatives_df = filter_to_ligand_binding_domains(True)
    else:
        ligands_negatives_df = filter_to_ligand_binding_domains(False)
else:
    if ABSOLUTE_NEGATIVES:
        ligands_negatives_df = negatives_by_binding_score(True)
    else:
        ligands_negatives_df = negatives_by_binding_score(False)

bind_th = 0.1
ligands_features_df = {}
    
for ligand in ligands:
    score_col_str = ligand+"_binding_score"
    ligand_binding_df = features_all[features_all[score_col_str] >= bind_th]
    print ligand+" #: "+str(ligand_binding_df.shape[0])
    ligands_features_df[ligand] = ligand_binding_df.loc[:,features_cols]

all_ligands_features_df = pd.concat([ligands_features_df["dna"], ligands_features_df["dnabase"], ligands_features_df["dnabackbone"], ligands_features_df["rna"], ligands_features_df["rnabase"], 
                                     ligands_features_df["rnabackbone"], ligands_features_df["ion"], ligands_features_df["peptide"], ligands_features_df["metabolite"]])
all_ligands_features_df = all_ligands_features_df.drop_duplicates()
print "all_ligands #: "+str(all_ligands_features_df.shape[0])
ligands_features_df["all_ligands"] = all_ligands_features_df

#Reading the ligand input
try:
    ligand = environ['ligand']
except:
    ligand = "dnabase"
print "ligand = "+ligand
    
#Reading the downsampler input
try: 
    downsample_method = environ['down']
except:
    downsample_method = "NoDown"
print "downsample_method = "+downsample_method

#Reading the classifier input
try: 
    classifier_method = environ['classifier']
except:
    classifier_method = "XGB"
print "classifier_method = "+classifier_method

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
#print "about to run"
returns = modelfit(xgb1, ligands_features_df[ligand], ligands_negatives_df[ligand], ligand)
print "Optimal n_estimators: "+str(returns[1].shape[0]) 
optimized_n_est = returns[1].shape[0]

optimized_n_est = returns[1].shape[0] 
ligand_bind_features = ligands_features_df[ligand]
ligand_negatives_features = ligands_negatives_df[ligand]
features = features = np.ones([ligand_bind_features.shape[1],]).astype(bool)
X = pd.concat([ligand_bind_features.iloc[:,features], ligand_negatives_features.iloc[:,features]])
y = [1] * ligand_bind_features.shape[0]
y.extend([0] * ligand_negatives_features.shape[0])
y = np.array(y)

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
print "Making GridSearchCV object"
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=optimized_n_est, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27, verbose=10), 
 param_grid = param_test1, scoring='average_precision',n_jobs=4,iid=False, cv=5)
print "Fitting"
gsearch1.fit(X,y)
print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
