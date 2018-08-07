import numpy as np
import pickle
from collections import defaultdict
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.utils import shuffle

# Neural Net imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR



# define the network 
class Net(nn.Module):
    def __init__(self, dropout_parameter = 0.5, hidden_units_1 = 200, 
                 hidden_units_2 = 400, batch_size = 75, 
                 learning_rate = 1e-5, beta = 0.9, 
                 weight_decay = 1e-4, epoch_count = 15, weight="balanced", input_size = 750):
        torch.manual_seed(0)
        super(Net, self).__init__()
        self.input = nn.Linear(input_size, hidden_units_1) 
        self.hidden1 = nn.Linear(hidden_units_1, hidden_units_2)
        self.hidden1_bn = nn.BatchNorm1d(hidden_units_2)
        self.hidden2 = nn.Linear(hidden_units_2, hidden_units_2)
        self.hidden2_bn = nn.BatchNorm1d(hidden_units_2)
        self.hidden3 = nn.Linear(hidden_units_2, hidden_units_1)
        self.hidden3_bn = nn.BatchNorm1d(hidden_units_1)
        self.dropout = nn.Dropout(p = dropout_parameter)
        self.output = nn.Linear(hidden_units_1,2)
        self.learning_rate = learning_rate
        self.beta = beta
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.epoch_count = epoch_count
        self.weight = weight
        

    def forward(self, x):
        sf = nn.Softmax()
        x = F.rrelu(self.input(x))
        x = self.dropout(F.rrelu(self.hidden1_bn(self.hidden1(x))))
        x = self.dropout(F.rrelu(self.hidden2_bn(self.hidden2(x))))
        x = self.dropout(F.rrelu(self.hidden3_bn(self.hidden3(x))))
        x = self.output(x)
        return x
    
    def fit(self, train_valid_data, train_valid_labels, weight):
        # set in training mode
        self.train()
        
        # set random seed
        torch.manual_seed(0)
          
        trainset = pd.concat([train_valid_data,train_valid_labels],axis=1)
        trainset = shuffle(trainset, random_state = 0)

        train_valid_data = trainset.iloc[:,:trainset.shape[1]-1]
        train_valid_labels = trainset.iloc[:,trainset.shape[1]-1]

        # create loss function
        loss = nn.BCEWithLogitsLoss(weight = weight)
        # mini-batching
        batch_size = self.batch_size
        
        BETA_2 = 0.999        
        no_batch_minus_1 = train_valid_data.shape[0] / batch_size 

        skf_2 = RepeatedStratifiedKFold(n_splits=no_batch_minus_1,n_repeats = self.epoch_count,random_state=0)

        # create adam optimizer for Phase 2
        optimizer_2 = optim.Adam(self.parameters(), lr=self.learning_rate,betas = (self.beta,BETA_2), 
                                 weight_decay = self.weight_decay)
        
        lambda1 = lambda epoch_count: 0.99 ** epoch_count 
        scheduler = LambdaLR(optimizer_2, lr_lambda=lambda1)
        
        count = 0
        epoch_count = 0
        
        for train,test in skf_2.split(train_valid_data,train_valid_labels):
            data = train_valid_data.iloc[test,:]
            data = torch.Tensor(data.values.astype(np.float32))
            # forward pass          
            output = self.forward(data)
            output.data = output.data.view(data.shape[0],2)

            labels = train_valid_labels[test]
            labels = labels.astype(int)
            labels = torch.Tensor(np.eye(2)[labels])
            labels = torch.autograd.Variable(labels, requires_grad = False)

            # zero the gradient buffers
            optimizer_2.zero_grad()
            # compute loss and gradients
            loss_output = loss(output,labels)
            loss_output.backward()
            # Does the update
            optimizer_2.step()
            
            count = count + 1

            # Early Stopping
            if count == no_batch_minus_1 + 1:
                count = 0
                epoch_count = epoch_count + 1
                scheduler.step()
            
    #prediction probabilities array
    def predict_proba(self, X_test):
        self.eval()
        #forward pass
        test = torch.Tensor(X_test.values.astype(np.float32))
        output = self.forward(test)
        sf = nn.Softmax()
        probs = sf(output.data)
        probs_list = []
        for i in range(len(probs)):
            probs_list.append(probs[i][1].item())          
        return probs_list


def generate_models_dict(ligand, ligands, ligands_positives_df, ligands_negatives_df, folds_num, no_features):
    """
    Create a dictionary of models with specific hyperparameters values for every ligand and fold combination.
    The function return the dictionary as output and also saves it in a pickle format.
    """
    models = ["XGB", "SVM", "RF", "ADA", "KNN", "Logistic", "NN"]
    ligand_pos = ligands_positives_df[ligand].shape[0]
    ligand_neg = ligands_negatives_df[ligand].shape[0]

    #Initialize classifier dict
    classifiers = defaultdict(dict)
    for ligand in ligands:
        for model in models:
            classifiers[model][ligand] = dict.fromkeys(np.arange(1,folds_num+1))


    #Update the specific hyperparameters values

    ###XGB###
    #==dna==#
    classifiers["XGB"]["dna"][1] = XGBClassifier(n_estimators=169, n_jobs=-1, random_state=0, max_depth=45, min_child_weight=1.185689, colsample_bytree=0.883199, 
                                               scale_pos_weight=0.1, gamma=0.374832, learning_rate=0.131259)
    classifiers["XGB"]["dna"][2] = XGBClassifier(n_estimators=1129, n_jobs=-1, random_state=0, max_depth=54, min_child_weight=1.56085330221, colsample_bytree=0.988774209162, 
                                               scale_pos_weight=0.1, gamma=0.1822673250862, learning_rate=0.00102357640439)
    classifiers["XGB"]["dna"][3] = XGBClassifier(n_estimators=821, n_jobs=-1, random_state=0, max_depth=26, min_child_weight=0.142072116396, colsample_bytree=0.315346974776, 
                                               scale_pos_weight=0.1, gamma=0.00114988707471, learning_rate=0.131259)
    classifiers["XGB"]["dna"][4] = XGBClassifier(n_estimators=1250, n_jobs=-1, random_state=0, max_depth=70, min_child_weight=0.721091120972, colsample_bytree=0.871492685917, 
                                               scale_pos_weight=0.1, gamma=0.595526012087, learning_rate=0.00130322162319)
    classifiers["XGB"]["dna"][5] = XGBClassifier(n_estimators=892, n_jobs=-1, random_state=0, max_depth=81, min_child_weight=1.91796544373, colsample_bytree=0.516526636354, 
                                               scale_pos_weight=0.1, gamma=0.0117522405231, learning_rate=0.00109855361287)

    #==rna==#
    classifiers["XGB"]["rna"][1] = XGBClassifier(n_estimators=169, n_jobs=-1, random_state=0, max_depth=45, min_child_weight=1.185689, colsample_bytree=0.883199, 
                                               scale_pos_weight=0.1, gamma=0.374832, learning_rate=0.131259)
    classifiers["XGB"]["rna"][2] = XGBClassifier(n_estimators=1129, n_jobs=-1, random_state=0, max_depth=54, min_child_weight=1.56085330221, colsample_bytree=0.988774209162, 
                                               scale_pos_weight=0.1, gamma=0.1822673250862, learning_rate=0.00102357640439)
    classifiers["XGB"]["rna"][3] = XGBClassifier(n_estimators=821, n_jobs=-1, random_state=0, max_depth=26, min_child_weight=0.142072116396, colsample_bytree=0.315346974776, 
                                               scale_pos_weight=0.1, gamma=0.00114988707471, learning_rate=0.131259)
    classifiers["XGB"]["rna"][4] = XGBClassifier(n_estimators=1250, n_jobs=-1, random_state=0, max_depth=70, min_child_weight=0.721091120972, colsample_bytree=0.871492685917, 
                                               scale_pos_weight=0.1, gamma=0.595526012087, learning_rate=0.00130322162319)
    classifiers["XGB"]["rna"][5] = XGBClassifier(n_estimators=892, n_jobs=-1, random_state=0, max_depth=81, min_child_weight=1.91796544373, colsample_bytree=0.516526636354, 
                                               scale_pos_weight=0.1, gamma=0.0117522405231, learning_rate=0.00109855361287)
    #==ion==#
    classifiers["XGB"]["ion"][1] = XGBClassifier(n_estimators=169, n_jobs=-1, random_state=0, max_depth=45, min_child_weight=1.185689, colsample_bytree=0.883199, 
                                               scale_pos_weight=0.1, gamma=0.374832, learning_rate=0.131259)
    classifiers["XGB"]["ion"][2] = XGBClassifier(n_estimators=1129, n_jobs=-1, random_state=0, max_depth=54, min_child_weight=1.56085330221, colsample_bytree=0.988774209162, 
                                               scale_pos_weight=0.1, gamma=0.1822673250862, learning_rate=0.00102357640439)
    classifiers["XGB"]["ion"][3] = XGBClassifier(n_estimators=821, n_jobs=-1, random_state=0, max_depth=26, min_child_weight=0.142072116396, colsample_bytree=0.315346974776, 
                                               scale_pos_weight=0.1, gamma=0.00114988707471, learning_rate=0.131259)
    classifiers["XGB"]["ion"][4] = XGBClassifier(n_estimators=1250, n_jobs=-1, random_state=0, max_depth=70, min_child_weight=0.721091120972, colsample_bytree=0.871492685917, 
                                               scale_pos_weight=0.1, gamma=0.595526012087, learning_rate=0.00130322162319)
    classifiers["XGB"]["ion"][5] = XGBClassifier(n_estimators=892, n_jobs=-1, random_state=0, max_depth=81, min_child_weight=1.91796544373, colsample_bytree=0.516526636354, 
                                               scale_pos_weight=0.1, gamma=0.0117522405231, learning_rate=0.00109855361287)
    #==peptide==#
    classifiers["XGB"]["peptide"][1] = XGBClassifier(n_estimators=169, n_jobs=-1, random_state=0, max_depth=45, min_child_weight=1.185689, colsample_bytree=0.883199, 
                                               scale_pos_weight=0.1, gamma=0.374832, learning_rate=0.131259)
    classifiers["XGB"]["peptide"][2] = XGBClassifier(n_estimators=1129, n_jobs=-1, random_state=0, max_depth=54, min_child_weight=1.56085330221, colsample_bytree=0.988774209162, 
                                               scale_pos_weight=0.1, gamma=0.1822673250862, learning_rate=0.00102357640439)
    classifiers["XGB"]["peptide"][3] = XGBClassifier(n_estimators=821, n_jobs=-1, random_state=0, max_depth=26, min_child_weight=0.142072116396, colsample_bytree=0.315346974776, 
                                               scale_pos_weight=0.1, gamma=0.00114988707471, learning_rate=0.131259)
    classifiers["XGB"]["peptide"][4] = XGBClassifier(n_estimators=1250, n_jobs=-1, random_state=0, max_depth=70, min_child_weight=0.721091120972, colsample_bytree=0.871492685917, 
                                               scale_pos_weight=0.1, gamma=0.595526012087, learning_rate=0.00130322162319)
    classifiers["XGB"]["peptide"][5] = XGBClassifier(n_estimators=892, n_jobs=-1, random_state=0, max_depth=81, min_child_weight=1.91796544373, colsample_bytree=0.516526636354, 
                                               scale_pos_weight=0.1, gamma=0.0117522405231, learning_rate=0.00109855361287)
    #==metabolite==#
    classifiers["XGB"]["metabolite"][1] = XGBClassifier(n_estimators=169, n_jobs=-1, random_state=0, max_depth=45, min_child_weight=1.185689, colsample_bytree=0.883199, 
                                               scale_pos_weight=0.1, gamma=0.374832, learning_rate=0.131259)
    classifiers["XGB"]["metabolite"][2] = XGBClassifier(n_estimators=1129, n_jobs=-1, random_state=0, max_depth=54, min_child_weight=1.56085330221, colsample_bytree=0.988774209162, 
                                               scale_pos_weight=0.1, gamma=0.1822673250862, learning_rate=0.00102357640439)
    classifiers["XGB"]["metabolite"][3] = XGBClassifier(n_estimators=821, n_jobs=-1, random_state=0, max_depth=26, min_child_weight=0.142072116396, colsample_bytree=0.315346974776, 
                                               scale_pos_weight=0.1, gamma=0.00114988707471, learning_rate=0.131259)
    classifiers["XGB"]["metabolite"][4] = XGBClassifier(n_estimators=1250, n_jobs=-1, random_state=0, max_depth=70, min_child_weight=0.721091120972, colsample_bytree=0.871492685917, 
                                               scale_pos_weight=0.1, gamma=0.595526012087, learning_rate=0.00130322162319)
    classifiers["XGB"]["metabolite"][5] = XGBClassifier(n_estimators=892, n_jobs=-1, random_state=0, max_depth=81, min_child_weight=1.91796544373, colsample_bytree=0.516526636354, 
                                               scale_pos_weight=0.1, gamma=0.0117522405231, learning_rate=0.00109855361287)
    #==sm==#
    classifiers["XGB"]["sm"][1] = XGBClassifier(n_estimators=169, n_jobs=-1, random_state=0, max_depth=45, min_child_weight=1.185689, colsample_bytree=0.883199, 
                                               scale_pos_weight=0.1, gamma=0.374832, learning_rate=0.131259)
    classifiers["XGB"]["sm"][2] = XGBClassifier(n_estimators=1129, n_jobs=-1, random_state=0, max_depth=54, min_child_weight=1.56085330221, colsample_bytree=0.988774209162, 
                                               scale_pos_weight=0.1, gamma=0.1822673250862, learning_rate=0.00102357640439)
    classifiers["XGB"]["sm"][3] = XGBClassifier(n_estimators=821, n_jobs=-1, random_state=0, max_depth=26, min_child_weight=0.142072116396, colsample_bytree=0.315346974776, 
                                               scale_pos_weight=0.1, gamma=0.00114988707471, learning_rate=0.131259)
    classifiers["XGB"]["sm"][4] = XGBClassifier(n_estimators=1250, n_jobs=-1, random_state=0, max_depth=70, min_child_weight=0.721091120972, colsample_bytree=0.871492685917, 
                                               scale_pos_weight=0.1, gamma=0.595526012087, learning_rate=0.00130322162319)
    classifiers["XGB"]["sm"][5] = XGBClassifier(n_estimators=892, n_jobs=-1, random_state=0, max_depth=81, min_child_weight=1.91796544373, colsample_bytree=0.516526636354, 
                                               scale_pos_weight=0.1, gamma=0.0117522405231, learning_rate=0.00109855361287)
    
    ###RF###
    #==dna==#
    classifiers["XGB"]["dna"][1] = RandomForestClassifier(n_estimators=940, max_depth=19, min_samples_leaf=24, min_samples_split=16, class_weight="balanced",
                                                   n_jobs=-1, random_state=0) 
    classifiers["XGB"]["dna"][2] = RandomForestClassifier(n_estimators=940, max_depth=19, min_samples_leaf=24, min_samples_split=16, class_weight="balanced",
                                                   n_jobs=-1, random_state=0) 
    classifiers["XGB"]["dna"][3] = RandomForestClassifier(n_estimators=940, max_depth=19, min_samples_leaf=24, min_samples_split=16, class_weight="balanced",
                                                   n_jobs=-1, random_state=0) 
    classifiers["XGB"]["dna"][4] = RandomForestClassifier(n_estimators=940, max_depth=19, min_samples_leaf=24, min_samples_split=16, class_weight="balanced",
                                                   n_jobs=-1, random_state=0) 
    classifiers["XGB"]["dna"][5] = RandomForestClassifier(n_estimators=940, max_depth=19, min_samples_leaf=24, min_samples_split=16, class_weight="balanced",
                                                   n_jobs=-1, random_state=0) 
    #==rna==#
    classifiers["XGB"]["rna"][1] = RandomForestClassifier(n_estimators=561, max_depth=14, min_samples_leaf=25, min_samples_split=26, class_weight="balanced",
                                                   n_jobs=-1, random_state=0) 
    classifiers["XGB"]["rna"][2] = RandomForestClassifier(n_estimators=561, max_depth=14, min_samples_leaf=25, min_samples_split=26, class_weight="balanced",
                                                   n_jobs=-1, random_state=0) 
    classifiers["XGB"]["rna"][3] = RandomForestClassifier(n_estimators=561, max_depth=14, min_samples_leaf=25, min_samples_split=26, class_weight="balanced",
                                                   n_jobs=-1, random_state=0) 
    classifiers["XGB"]["rna"][4] = RandomForestClassifier(n_estimators=561, max_depth=14, min_samples_leaf=25, min_samples_split=26, class_weight="balanced",
                                                   n_jobs=-1, random_state=0) 
    classifiers["XGB"]["rna"][5] = RandomForestClassifier(n_estimators=561, max_depth=14, min_samples_leaf=25, min_samples_split=26, class_weight="balanced",
                                                   n_jobs=-1, random_state=0) 
    #==ion==#
    classifiers["XGB"]["ion"][1] = RandomForestClassifier(n_estimators=734, max_depth=17, min_samples_leaf=12, min_samples_split=13, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    classifiers["XGB"]["ion"][2] = RandomForestClassifier(n_estimators=734, max_depth=17, min_samples_leaf=12, min_samples_split=13, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    classifiers["XGB"]["ion"][3] = RandomForestClassifier(n_estimators=734, max_depth=17, min_samples_leaf=12, min_samples_split=13, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    classifiers["XGB"]["ion"][4] = RandomForestClassifier(n_estimators=734, max_depth=17, min_samples_leaf=12, min_samples_split=13, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    classifiers["XGB"]["ion"][5] = RandomForestClassifier(n_estimators=734, max_depth=17, min_samples_leaf=12, min_samples_split=13, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    #==peptide==#
    classifiers["XGB"]["peptide"][1] = RandomForestClassifier(n_estimators=659, max_depth=18, min_samples_leaf=28, min_samples_split=12, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    classifiers["XGB"]["peptide"][2] = RandomForestClassifier(n_estimators=659, max_depth=18, min_samples_leaf=28, min_samples_split=12, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    classifiers["XGB"]["peptide"][3] = RandomForestClassifier(n_estimators=659, max_depth=18, min_samples_leaf=28, min_samples_split=12, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    classifiers["XGB"]["peptide"][4] = RandomForestClassifier(n_estimators=659, max_depth=18, min_samples_leaf=28, min_samples_split=12, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    classifiers["XGB"]["peptide"][5] = RandomForestClassifier(n_estimators=659, max_depth=18, min_samples_leaf=28, min_samples_split=12, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    #==metabolite==#
    classifiers["XGB"]["metabolite"][1] = RandomForestClassifier(n_estimators=928, max_depth=19, min_samples_leaf=35, min_samples_split=26, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    classifiers["XGB"]["metabolite"][2] = RandomForestClassifier(n_estimators=928, max_depth=19, min_samples_leaf=35, min_samples_split=26, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    classifiers["XGB"]["metabolite"][3] = RandomForestClassifier(n_estimators=928, max_depth=19, min_samples_leaf=35, min_samples_split=26, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    classifiers["XGB"]["metabolite"][4] = RandomForestClassifier(n_estimators=928, max_depth=19, min_samples_leaf=35, min_samples_split=26, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    classifiers["XGB"]["metabolite"][5] = RandomForestClassifier(n_estimators=928, max_depth=19, min_samples_leaf=35, min_samples_split=26, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    #==sm==#
    classifiers["XGB"]["sm"][1] = RandomForestClassifier(n_estimators=746, max_depth=18, min_samples_leaf=28, min_samples_split=18, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    classifiers["XGB"]["sm"][2] = RandomForestClassifier(n_estimators=746, max_depth=18, min_samples_leaf=28, min_samples_split=18, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    classifiers["XGB"]["sm"][3] = RandomForestClassifier(n_estimators=746, max_depth=18, min_samples_leaf=28, min_samples_split=18, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    classifiers["XGB"]["sm"][4] = RandomForestClassifier(n_estimators=746, max_depth=18, min_samples_leaf=28, min_samples_split=18, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    classifiers["XGB"]["sm"][5] = RandomForestClassifier(n_estimators=746, max_depth=18, min_samples_leaf=28, min_samples_split=18, class_weight="balanced",
                                                   n_jobs=-1, random_state=0)
    
    ###ADA###
    #==dna==#
    classifiers["ADA"]["dna"][1] = AdaBoostClassifier(n_estimators=870, learning_rate=0.045646, random_state=0)
    classifiers["ADA"]["dna"][2] = AdaBoostClassifier(n_estimators=870, learning_rate=0.045646, random_state=0)
    classifiers["ADA"]["dna"][3] = AdaBoostClassifier(n_estimators=870, learning_rate=0.045646, random_state=0)
    classifiers["ADA"]["dna"][4] = AdaBoostClassifier(n_estimators=870, learning_rate=0.045646, random_state=0)
    classifiers["ADA"]["dna"][5] = AdaBoostClassifier(n_estimators=870, learning_rate=0.045646, random_state=0)
    #==rna==#
    classifiers["ADA"]["rna"][1] = AdaBoostClassifier(n_estimators=890, learning_rate=0.011096, random_state=0)
    classifiers["ADA"]["rna"][2] = AdaBoostClassifier(n_estimators=890, learning_rate=0.011096, random_state=0)
    classifiers["ADA"]["rna"][3] = AdaBoostClassifier(n_estimators=890, learning_rate=0.011096, random_state=0)
    classifiers["ADA"]["rna"][4] = AdaBoostClassifier(n_estimators=890, learning_rate=0.011096, random_state=0)
    classifiers["ADA"]["rna"][5] = AdaBoostClassifier(n_estimators=890, learning_rate=0.011096, random_state=0)
    #==ion==#
    classifiers["ADA"]["ion"][1] = AdaBoostClassifier(n_estimators=839, learning_rate=0.04376, random_state=0)
    classifiers["ADA"]["ion"][2] = AdaBoostClassifier(n_estimators=839, learning_rate=0.04376, random_state=0)
    classifiers["ADA"]["ion"][3] = AdaBoostClassifier(n_estimators=839, learning_rate=0.04376, random_state=0)
    classifiers["ADA"]["ion"][4] = AdaBoostClassifier(n_estimators=839, learning_rate=0.04376, random_state=0)
    classifiers["ADA"]["ion"][5] = AdaBoostClassifier(n_estimators=839, learning_rate=0.04376, random_state=0)
    #==peptide==#
    classifiers["ADA"]["peptide"][1] = AdaBoostClassifier(n_estimators=371, learning_rate=0.01638, random_state=0)
    classifiers["ADA"]["peptide"][2] = AdaBoostClassifier(n_estimators=371, learning_rate=0.01638, random_state=0)
    classifiers["ADA"]["peptide"][3] = AdaBoostClassifier(n_estimators=371, learning_rate=0.01638, random_state=0)
    classifiers["ADA"]["peptide"][4] = AdaBoostClassifier(n_estimators=371, learning_rate=0.01638, random_state=0)
    classifiers["ADA"]["peptide"][5] = AdaBoostClassifier(n_estimators=371, learning_rate=0.01638, random_state=0)
    #==metabolite==#
    classifiers["ADA"]["metabolite"][1] = AdaBoostClassifier(n_estimators=670, learning_rate=0.065686, random_state=0)
    classifiers["ADA"]["metabolite"][2] = AdaBoostClassifier(n_estimators=670, learning_rate=0.065686, random_state=0)
    classifiers["ADA"]["metabolite"][3] = AdaBoostClassifier(n_estimators=670, learning_rate=0.065686, random_state=0)
    classifiers["ADA"]["metabolite"][4] = AdaBoostClassifier(n_estimators=670, learning_rate=0.065686, random_state=0)
    classifiers["ADA"]["metabolite"][5] = AdaBoostClassifier(n_estimators=670, learning_rate=0.065686, random_state=0)
    #==sm==#
    classifiers["ADA"]["sm"][1] = AdaBoostClassifier(n_estimators=916, learning_rate=0.04588, random_state=0) 
    classifiers["ADA"]["sm"][2] = AdaBoostClassifier(n_estimators=916, learning_rate=0.04588, random_state=0) 
    classifiers["ADA"]["sm"][3] = AdaBoostClassifier(n_estimators=916, learning_rate=0.04588, random_state=0) 
    classifiers["ADA"]["sm"][4] = AdaBoostClassifier(n_estimators=916, learning_rate=0.04588, random_state=0) 
    classifiers["ADA"]["sm"][5] = AdaBoostClassifier(n_estimators=916, learning_rate=0.04588, random_state=0) 
    
    ###SVM###
    #==dna==#
    classifiers["SVM"]["dna"][1] = SVC(kernel='rbf', class_weight=None, C=1.1269, gamma=0.0125, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["dna"][2] = SVC(kernel='rbf', class_weight=None, C=1.1269, gamma=0.0125, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["dna"][3] = SVC(kernel='rbf', class_weight=None, C=1.1269, gamma=0.0125, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["dna"][4] = SVC(kernel='rbf', class_weight=None, C=1.1269, gamma=0.0125, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["dna"][5] = SVC(kernel='rbf', class_weight=None, C=1.1269, gamma=0.0125, probability=True, random_state=0, cache_size=200)
    #==rna==#
    classifiers["SVM"]["rna"][1] = SVC(kernel='rbf', class_weight=None, C=0.00158, gamma=0.00001186, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["rna"][2] = SVC(kernel='rbf', class_weight=None, C=0.00158, gamma=0.00001186, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["rna"][3] = SVC(kernel='rbf', class_weight=None, C=0.00158, gamma=0.00001186, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["rna"][4] = SVC(kernel='rbf', class_weight=None, C=0.00158, gamma=0.00001186, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["rna"][5] = SVC(kernel='rbf', class_weight=None, C=0.00158, gamma=0.00001186, probability=True, random_state=0, cache_size=200)
    #==ion==#
    classifiers["SVM"]["ion"][1] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=6.4298, gamma=0.000934, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["ion"][2] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=6.4298, gamma=0.000934, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["ion"][3] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=6.4298, gamma=0.000934, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["ion"][4] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=6.4298, gamma=0.000934, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["ion"][5] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=6.4298, gamma=0.000934, probability=True, random_state=0, cache_size=200)
    #==peptide==#
    classifiers["SVM"]["peptide"][1] = SVC(kernel='poly', class_weight=None, C=0.001828, gamma=0.01238, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["peptide"][2] = SVC(kernel='poly', class_weight=None, C=0.001828, gamma=0.01238, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["peptide"][3] = SVC(kernel='poly', class_weight=None, C=0.001828, gamma=0.01238, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["peptide"][4] = SVC(kernel='poly', class_weight=None, C=0.001828, gamma=0.01238, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["peptide"][5] = SVC(kernel='poly', class_weight=None, C=0.001828, gamma=0.01238, probability=True, random_state=0, cache_size=200)
    #==metabolite==#
    classifiers["SVM"]["metabolite"][1] = SVC(kernel='rbf', class_weight=None, C=1.182, gamma=0.025, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["metabolite"][2] = SVC(kernel='rbf', class_weight=None, C=1.182, gamma=0.025, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["metabolite"][3] = SVC(kernel='rbf', class_weight=None, C=1.182, gamma=0.025, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["metabolite"][4] = SVC(kernel='rbf', class_weight=None, C=1.182, gamma=0.025, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["metabolite"][5] = SVC(kernel='rbf', class_weight=None, C=1.182, gamma=0.025, probability=True, random_state=0, cache_size=200)
    #==sm==#
    classifiers["SVM"]["sm"][1] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=24.8123, gamma=0.001616, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["sm"][2] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=24.8123, gamma=0.001616, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["sm"][3] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=24.8123, gamma=0.001616, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["sm"][4] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=24.8123, gamma=0.001616, probability=True, random_state=0, cache_size=200)
    classifiers["SVM"]["sm"][5] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=24.8123, gamma=0.001616, probability=True, random_state=0, cache_size=200)
    
    ###Logistic###
    #==dna==#
    classifiers["Logistic"]["dna"][1] = LogisticRegression(C=0.006834, random_state=0, n_jobs=-1, class_weight=None)
    classifiers["Logistic"]["dna"][2] = LogisticRegression(C=0.006834, random_state=0, n_jobs=-1, class_weight=None)
    classifiers["Logistic"]["dna"][3] = LogisticRegression(C=0.006834, random_state=0, n_jobs=-1, class_weight=None)
    classifiers["Logistic"]["dna"][4] = LogisticRegression(C=0.006834, random_state=0, n_jobs=-1, class_weight=None)
    classifiers["Logistic"]["dna"][5] = LogisticRegression(C=0.006834, random_state=0, n_jobs=-1, class_weight=None)
    #==rna==#
    classifiers["Logistic"]["rna"][1] = LogisticRegression(C=0.028598, random_state=0, n_jobs=-1, class_weight=None)
    classifiers["Logistic"]["rna"][2] = LogisticRegression(C=0.028598, random_state=0, n_jobs=-1, class_weight=None)
    classifiers["Logistic"]["rna"][3] = LogisticRegression(C=0.028598, random_state=0, n_jobs=-1, class_weight=None)
    classifiers["Logistic"]["rna"][4] = LogisticRegression(C=0.028598, random_state=0, n_jobs=-1, class_weight=None)
    classifiers["Logistic"]["rna"][5] = LogisticRegression(C=0.028598, random_state=0, n_jobs=-1, class_weight=None)
    #==ion==#
    classifiers["Logistic"]["ion"][1] = LogisticRegression(C=0.014664, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1})
    classifiers["Logistic"]["ion"][2] = LogisticRegression(C=0.014664, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1})
    classifiers["Logistic"]["ion"][3] = LogisticRegression(C=0.014664, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1})
    classifiers["Logistic"]["ion"][4] = LogisticRegression(C=0.014664, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1})
    classifiers["Logistic"]["ion"][5] = LogisticRegression(C=0.014664, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1})
    #==peptide==#
    classifiers["Logistic"]["peptide"][1] = LogisticRegression(C=0.00598, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1})
    classifiers["Logistic"]["peptide"][2] = LogisticRegression(C=0.00598, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1})
    classifiers["Logistic"]["peptide"][3] = LogisticRegression(C=0.00598, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1})
    classifiers["Logistic"]["peptide"][4] = LogisticRegression(C=0.00598, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1})
    classifiers["Logistic"]["peptide"][5] = LogisticRegression(C=0.00598, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1})
    #==metabolite==#
    classifiers["Logistic"]["metabolite"][1] = LogisticRegression(C=0.004859, random_state=0, n_jobs=-1, class_weight=None)
    classifiers["Logistic"]["metabolite"][2] = LogisticRegression(C=0.004859, random_state=0, n_jobs=-1, class_weight=None)
    classifiers["Logistic"]["metabolite"][3] = LogisticRegression(C=0.004859, random_state=0, n_jobs=-1, class_weight=None)
    classifiers["Logistic"]["metabolite"][4] = LogisticRegression(C=0.004859, random_state=0, n_jobs=-1, class_weight=None)
    classifiers["Logistic"]["metabolite"][5] = LogisticRegression(C=0.004859, random_state=0, n_jobs=-1, class_weight=None)
    #==sm==#
    classifiers["Logistic"]["sm"][1] = LogisticRegression(C=0.002812, random_state=0, n_jobs=-1, class_weight=None)
    classifiers["Logistic"]["sm"][2] = LogisticRegression(C=0.002812, random_state=0, n_jobs=-1, class_weight=None)
    classifiers["Logistic"]["sm"][3] = LogisticRegression(C=0.002812, random_state=0, n_jobs=-1, class_weight=None)
    classifiers["Logistic"]["sm"][4] = LogisticRegression(C=0.002812, random_state=0, n_jobs=-1, class_weight=None)
    classifiers["Logistic"]["sm"][5] = LogisticRegression(C=0.002812, random_state=0, n_jobs=-1, class_weight=None)
    
    ###Logistic###
    #==dna==#
    classifiers["KNN"]["dna"][1] = KNeighborsClassifier(n_neighbors=394, n_jobs=-1, weights="distance")
    classifiers["KNN"]["dna"][2] = KNeighborsClassifier(n_neighbors=394, n_jobs=-1, weights="distance")
    classifiers["KNN"]["dna"][3] = KNeighborsClassifier(n_neighbors=394, n_jobs=-1, weights="distance")
    classifiers["KNN"]["dna"][4] = KNeighborsClassifier(n_neighbors=394, n_jobs=-1, weights="distance")
    classifiers["KNN"]["dna"][5] = KNeighborsClassifier(n_neighbors=394, n_jobs=-1, weights="distance")
    #==rna==#
    classifiers["KNN"]["rna"][1] = KNeighborsClassifier(n_neighbors=205, n_jobs=-1, weights="uniform")
    classifiers["KNN"]["rna"][2] = KNeighborsClassifier(n_neighbors=205, n_jobs=-1, weights="uniform")
    classifiers["KNN"]["rna"][3] = KNeighborsClassifier(n_neighbors=205, n_jobs=-1, weights="uniform")
    classifiers["KNN"]["rna"][4] = KNeighborsClassifier(n_neighbors=205, n_jobs=-1, weights="uniform")
    classifiers["KNN"]["rna"][5] = KNeighborsClassifier(n_neighbors=205, n_jobs=-1, weights="uniform")
    #==ion==#
    classifiers["KNN"]["ion"][1] = KNeighborsClassifier(n_neighbors=19, n_jobs=-1, weights="uniform")
    classifiers["KNN"]["ion"][2] = KNeighborsClassifier(n_neighbors=19, n_jobs=-1, weights="uniform")
    classifiers["KNN"]["ion"][3] = KNeighborsClassifier(n_neighbors=19, n_jobs=-1, weights="uniform")
    classifiers["KNN"]["ion"][4] = KNeighborsClassifier(n_neighbors=19, n_jobs=-1, weights="uniform")
    classifiers["KNN"]["ion"][5] = KNeighborsClassifier(n_neighbors=19, n_jobs=-1, weights="uniform")
    #==peptide==#
    classifiers["KNN"]["peptide"][1] = KNeighborsClassifier(n_neighbors=74, n_jobs=-1, weights="distance")
    classifiers["KNN"]["peptide"][2] = KNeighborsClassifier(n_neighbors=74, n_jobs=-1, weights="distance")
    classifiers["KNN"]["peptide"][3] = KNeighborsClassifier(n_neighbors=74, n_jobs=-1, weights="distance")
    classifiers["KNN"]["peptide"][4] = KNeighborsClassifier(n_neighbors=74, n_jobs=-1, weights="distance")
    classifiers["KNN"]["peptide"][5] = KNeighborsClassifier(n_neighbors=74, n_jobs=-1, weights="distance")
    #==metabolite==#
    classifiers["KNN"]["metabolite"][1] = KNeighborsClassifier(n_neighbors=161, n_jobs=-1, weights="distance")
    classifiers["KNN"]["metabolite"][2] = KNeighborsClassifier(n_neighbors=161, n_jobs=-1, weights="distance")
    classifiers["KNN"]["metabolite"][3] = KNeighborsClassifier(n_neighbors=161, n_jobs=-1, weights="distance")
    classifiers["KNN"]["metabolite"][4] = KNeighborsClassifier(n_neighbors=161, n_jobs=-1, weights="distance")
    classifiers["KNN"]["metabolite"][5] = KNeighborsClassifier(n_neighbors=161, n_jobs=-1, weights="distance")
    #==sm==#
    classifiers["KNN"]["sm"][1] = KNeighborsClassifier(n_neighbors=117, n_jobs=-1, weights="distance")
    classifiers["KNN"]["sm"][2] = KNeighborsClassifier(n_neighbors=117, n_jobs=-1, weights="distance")
    classifiers["KNN"]["sm"][3] = KNeighborsClassifier(n_neighbors=117, n_jobs=-1, weights="distance")
    classifiers["KNN"]["sm"][4] = KNeighborsClassifier(n_neighbors=117, n_jobs=-1, weights="distance")
    classifiers["KNN"]["sm"][5] = KNeighborsClassifier(n_neighbors=117, n_jobs=-1, weights="distance")
    
    ###NN###
    #==dna==#
    classifiers["NN"]["dna"][1] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11, input_size=no_features)
    classifiers["NN"]["dna"][2] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)    
    classifiers["NN"]["dna"][3] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["dna"][4] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["dna"][5] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)    
    #==rna==#
    classifiers["NN"]["rna"][1] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["rna"][2] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)    
    classifiers["NN"]["rna"][3] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["rna"][4] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["rna"][5] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)    
    
    #==ion==#    
    classifiers["NN"]["ion"][1] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["ion"][2] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)    
    classifiers["NN"]["ion"][3] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["ion"][4] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["ion"][5] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)    
    
    #==peptide==#
    classifiers["NN"]["peptide"][1] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["peptide"][2] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)    
    classifiers["NN"]["peptide"][3] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["peptide"][4] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["peptide"][5] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)  
    #==metabolite==#
    classifiers["NN"]["metabolite"][1] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["metabolite"][2] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)    
    classifiers["NN"]["metabolite"][3] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["metabolite"][4] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["metabolite"][5] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)  
    
    #==sm==#
    classifiers["NN"]["sm"][1] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["sm"][2] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)    
    classifiers["NN"]["sm"][3] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["sm"][4] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    classifiers["NN"]["sm"][5] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)  
    
    with open("hyperparams_dict.pik", 'wb') as handle:
        pickle.dump(classifiers, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return classifiers