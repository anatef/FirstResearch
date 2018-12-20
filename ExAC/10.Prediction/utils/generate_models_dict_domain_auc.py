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


def generate_models_dict(ligand, classifier_method, ligands, ligands_positives_df, ligands_negatives_df, folds_num, no_features):
    """
    Create a dictionary of models with specific hyperparameters values for every ligand and fold combination.
    The function return the dictionary as output and also saves it in a pickle format.
    """
    models = ["XGB", "SVM", "RF", "ADA", "KNN", "Logistic", "NN"]
    
    #For balanced weight of XGB
    ligand_pos = ligands_positives_df[ligand].shape[0]
    ligand_neg = ligands_negatives_df[ligand].shape[0]
    scale_weight = ligand_neg/float(ligand_pos) 

    #Initialize classifier dict
    classifiers = defaultdict(dict)
    classifiers[classifier_method][ligand] = dict.fromkeys(np.arange(1,folds_num+1))
            
    #For input size of NN
    features_num = ligands_positives_df[ligand].shape[1]


    #Update the specific hyperparameters values

    ###XGB###
    if (classifier_method == "XGB"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["XGB"]["dna"][1] = XGBClassifier(n_estimators=1242, n_jobs=-1, random_state=0, max_depth=24, min_child_weight=1.690038, colsample_bytree= 0.756064, 
                                                       scale_pos_weight=scale_weight, gamma=0.008946, learning_rate=0.052992) #
            classifiers["XGB"]["dna"][2] = XGBClassifier(n_estimators=4224, n_jobs=-1, random_state=0, max_depth=46, min_child_weight=2.045270, colsample_bytree=0.372216, 
                                                       scale_pos_weight=scale_weight, gamma=0.082468, learning_rate=0.005201) #
            classifiers["XGB"]["dna"][3] = XGBClassifier(n_estimators=4999, n_jobs=-1, random_state=0, max_depth=18, min_child_weight=4.363253, colsample_bytree=0.761365, 
                                                       scale_pos_weight=scale_weight, gamma=0.011982, learning_rate=0.0033856) #
            classifiers["XGB"]["dna"][4] = XGBClassifier(n_estimators=1802, n_jobs=-1, random_state=0, max_depth=67, min_child_weight=4.718740, colsample_bytree=0.313751, 
                                                       scale_pos_weight=1, gamma=0.001920, learning_rate=0.016505) #
            classifiers["XGB"]["dna"][5] = XGBClassifier(n_estimators=1659, n_jobs=-1, random_state=0, max_depth=26, min_child_weight=4.962148, colsample_bytree=0.315347, 
                                                       scale_pos_weight=scale_weight, gamma=0.001150, learning_rate=0.082067) #

        #==rna==#
        elif (ligand == "rna"):  
            ## for all the features
            classifiers["XGB"]["rna"][1] = XGBClassifier(n_estimators=138, n_jobs=-1, random_state=0, max_depth=3, min_child_weight=2.541577, colsample_bytree=0.375136, 
                                                       scale_pos_weight=scale_weight, gamma=0.217347, learning_rate=0.106480) #
            classifiers["XGB"]["rna"][2] = XGBClassifier(n_estimators=195, n_jobs=-1, random_state=0, max_depth=76, min_child_weight=1.541431, colsample_bytree=0.424649, 
                                                       scale_pos_weight=scale_weight, gamma=0.032380, learning_rate=0.217783) #
            classifiers["XGB"]["rna"][3] = XGBClassifier(n_estimators=1266, n_jobs=-1, random_state=0, max_depth=24, min_child_weight=1.690038, colsample_bytree=0.756064, 
                                                      scale_pos_weight=scale_weight, gamma=0.008946, learning_rate=0.052992) #
            classifiers["XGB"]["rna"][4] = XGBClassifier(n_estimators=3882, n_jobs=-1, random_state=0, max_depth=83, min_child_weight=0.644631, colsample_bytree=0.486571, 
                                                       scale_pos_weight=scale_weight, gamma=0.012335, learning_rate=0.009901) #
            classifiers["XGB"]["rna"][5] = XGBClassifier(n_estimators=1323, n_jobs=-1, random_state=0, max_depth=20, min_child_weight=3.394398, colsample_bytree=0.790474, 
                                                       scale_pos_weight=0.1, gamma=0.055726, learning_rate=0.007600) #trying to improve auprc_ratio
            #classifiers["XGB"]["rna"][5] = XGBClassifier(n_estimators=396, n_jobs=-1, random_state=0, max_depth=20, min_child_weight=1.980491, colsample_bytree=0.922029, 
                                                       #scale_pos_weight=scale_weight, gamma=0.082559, learning_rate=0.131959) #
            
        #==ion==#
        elif (ligand == "ion"):
            classifiers["XGB"]["ion"][1] = XGBClassifier(n_estimators=1809, n_jobs=-1, random_state=0, max_depth=85, min_child_weight=3.171370, colsample_bytree=0.969212, 
                                                       scale_pos_weight=scale_weight, gamma=0.090860, learning_rate=0.016700) #
            classifiers["XGB"]["ion"][2] = XGBClassifier(n_estimators=1076, n_jobs=-1, random_state=0, max_depth=56, min_child_weight=3.437441, colsample_bytree=0.411631, 
                                                      scale_pos_weight=0.1, gamma=0.695203, learning_rate=0.036141) #
            classifiers["XGB"]["ion"][3] = XGBClassifier(n_estimators=4134, n_jobs=-1, random_state=0, max_depth=46, min_child_weight=2.045270, colsample_bytree=0.372216, 
                                                       scale_pos_weight=scale_weight, gamma=0.082468, learning_rate=0.005201) #
            classifiers["XGB"]["ion"][4] = XGBClassifier(n_estimators=234, n_jobs=-1, random_state=0, max_depth=40, min_child_weight=2.964223, colsample_bytree=0.883199, 
                                                       scale_pos_weight=0.1, gamma=0.374832, learning_rate=0.092338) #
            classifiers["XGB"]["ion"][5] = XGBClassifier(n_estimators=1184, n_jobs=-1, random_state=0, max_depth=56, min_child_weight=3.437441, colsample_bytree=0.411631, 
                                                       scale_pos_weight=0.1, gamma=0.695203, learning_rate=0.036141) #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["XGB"]["peptide"][1] = XGBClassifier(n_estimators=665, n_jobs=-1, random_state=0, max_depth=35, min_child_weight=1.119623, colsample_bytree=0.509014, 
                                                       scale_pos_weight=0.1, gamma=0.608477, learning_rate=0.029205) #
            classifiers["XGB"]["peptide"][2] = XGBClassifier(n_estimators=480, n_jobs=-1, random_state=0, max_depth=88, min_child_weight=0.462981, colsample_bytree=0.413088, 
                                                       scale_pos_weight=1, gamma=0.176914, learning_rate=0.036385) #
            classifiers["XGB"]["peptide"][3] = XGBClassifier(n_estimators=948, n_jobs=-1, random_state=0, max_depth=24, min_child_weight=1.690038, colsample_bytree=0.756064, 
                                                       scale_pos_weight=scale_weight, gamma=0.008946, learning_rate=0.052992) #
            classifiers["XGB"]["peptide"][4] = XGBClassifier(n_estimators=2049, n_jobs=-1, random_state=0, max_depth=87, min_child_weight=1.568462, colsample_bytree=0.968088, 
                                                       scale_pos_weight=scale_weight, gamma=0.026017, learning_rate=0.030942) #
            classifiers["XGB"]["peptide"][5] = XGBClassifier(n_estimators=2790, n_jobs=-1, random_state=0, max_depth=29, min_child_weight=4.197565, colsample_bytree=0.641778, 
                                                       scale_pos_weight=0.1, gamma=0.034612, learning_rate=0.018811) #
 
        #==sm==#
        elif (ligand == "sm"):
            classifiers["XGB"]["sm"][1] = XGBClassifier(n_estimators=494, n_jobs=-1, random_state=0, max_depth=35, min_child_weight=1.119623, colsample_bytree=0.509014, 
                                                       scale_pos_weight=0.1, gamma=0.608477, learning_rate=0.029205) #
            classifiers["XGB"]["sm"][2] = XGBClassifier(n_estimators=3166, n_jobs=-1, random_state=0, max_depth=25, min_child_weight=3.577806, colsample_bytree=0.614620, 
                                                       scale_pos_weight=scale_weight, gamma=0.133550, learning_rate=0.005540) #
            classifiers["XGB"]["sm"][3] = XGBClassifier(n_estimators=3858, n_jobs=-1, random_state=0, max_depth=46, min_child_weight=2.045270476865308, colsample_bytree=0.372216, 
                                                       scale_pos_weight=scale_weight, gamma=0.082468, learning_rate=0.005201) #
            classifiers["XGB"]["sm"][4] = XGBClassifier(n_estimators=3625, n_jobs=-1, random_state=0, max_depth=83, min_child_weight=0.644631, colsample_bytree=0.486571, 
                                                       scale_pos_weight=scale_weight, gamma=0.012335, learning_rate=0.009901) #
            classifiers["XGB"]["sm"][5] = XGBClassifier(n_estimators=3174, n_jobs=-1, random_state=0, max_depth=83, min_child_weight=0.644631, colsample_bytree=0.486571, 
                                                       scale_pos_weight=scale_weight, gamma=0.012335, learning_rate=0.009901) #
    
    ###RF###
    elif (classifier_method == "RF"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["RF"]["dna"][1] = RandomForestClassifier(n_estimators=1415, max_depth=26, min_samples_leaf=4, min_samples_split=20, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["dna"][2] = RandomForestClassifier(n_estimators=920, max_depth=41, min_samples_leaf=33, min_samples_split=3, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["dna"][3] = RandomForestClassifier(n_estimators=1310, max_depth=84, min_samples_leaf=5, min_samples_split=24, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["dna"][4] = RandomForestClassifier(n_estimators=1310, max_depth=84, min_samples_leaf=5, min_samples_split=24, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["dna"][5] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
        #==rna==#
        elif (ligand == "rna"):
            classifiers["RF"]["rna"][1] = RandomForestClassifier(n_estimators=951, max_depth=89, min_samples_leaf=2, min_samples_split=8, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["rna"][2] = RandomForestClassifier(n_estimators=1154, max_depth=57, min_samples_leaf=1, min_samples_split=13, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["rna"][3] = RandomForestClassifier(n_estimators=1154, max_depth=57, min_samples_leaf=1, min_samples_split=13, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["rna"][4] = RandomForestClassifier(n_estimators=1310, max_depth=84, min_samples_leaf=5, min_samples_split=24, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["rna"][5] = RandomForestClassifier(n_estimators=951, max_depth=89, min_samples_leaf=2, min_samples_split=8, class_weight="balanced", n_jobs=-1, random_state=0) #
        #==ion==#
        elif (ligand == "ion"):
            classifiers["RF"]["ion"][1] = RandomForestClassifier(n_estimators=1154, max_depth=57, min_samples_leaf=1, min_samples_split=13, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["ion"][2] = RandomForestClassifier(n_estimators=1154, max_depth=57, min_samples_leaf=1, min_samples_split=13, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["ion"][3] = RandomForestClassifier(n_estimators=610, max_depth=14, min_samples_leaf=2, min_samples_split=40, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["ion"][4] = RandomForestClassifier(n_estimators=749, max_depth=22, min_samples_leaf=18, min_samples_split=29, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["ion"][5] = RandomForestClassifier(n_estimators=959, max_depth=31, min_samples_leaf=4, min_samples_split=37, class_weight="balanced", n_jobs=-1, random_state=0) #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["RF"]["peptide"][1] = RandomForestClassifier(n_estimators=829, max_depth=20, min_samples_leaf=35, min_samples_split=32, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["peptide"][2] = RandomForestClassifier(n_estimators=959, max_depth=31, min_samples_leaf=4, min_samples_split=37, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["peptide"][3] = RandomForestClassifier(n_estimators=565, max_depth=34, min_samples_leaf=12, min_samples_split=42, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["peptide"][4] = RandomForestClassifier(n_estimators=765, max_depth=71, min_samples_leaf=16, min_samples_split=49, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["peptide"][5] = RandomForestClassifier(n_estimators=610, max_depth=14, min_samples_leaf=2, min_samples_split=40, class_weight="balanced", n_jobs=-1, random_state=0) #
        #==sm==#
        elif (ligand == "sm"):
            classifiers["RF"]["sm"][1] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["sm"][2] = RandomForestClassifier(n_estimators=565, max_depth=34, min_samples_leaf=12, min_samples_split=42, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["sm"][3] = RandomForestClassifier(n_estimators=518, max_depth=37, min_samples_leaf=31, min_samples_split=31, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["sm"][4] = RandomForestClassifier(n_estimators=1310, max_depth=84, min_samples_leaf=5, min_samples_split=24, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["sm"][5] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
    
    ###ADA###
    elif (classifier_method == "ADA"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["ADA"]["dna"][1] = AdaBoostClassifier(n_estimators=1204, learning_rate=0.03214735265313265, random_state=0) #
            classifiers["ADA"]["dna"][2] = AdaBoostClassifier(n_estimators=1204, learning_rate=0.03214735265313265, random_state=0) #
            classifiers["ADA"]["dna"][3] = AdaBoostClassifier(n_estimators=111, learning_rate=0.08875590953616813, random_state=0) #
            classifiers["ADA"]["dna"][4] = AdaBoostClassifier(n_estimators=759, learning_rate=0.0237726915486892, random_state=0) # 
            classifiers["ADA"]["dna"][5] = AdaBoostClassifier(n_estimators=1298, learning_rate=0.011292634498495331, random_state=0) #
        #==rna==#
        elif (ligand == "rna"):
            classifiers["ADA"]["rna"][1] = AdaBoostClassifier(n_estimators=801, learning_rate=0.007578422693958828, random_state=0) #
            classifiers["ADA"]["rna"][2] = AdaBoostClassifier(n_estimators=1389, learning_rate=0.007599872442483752, random_state=0) #
            classifiers["ADA"]["rna"][3] = AdaBoostClassifier(n_estimators=1307, learning_rate=0.0005726987569667398, random_state=0) #
            classifiers["ADA"]["rna"][4] = AdaBoostClassifier(n_estimators=1350, learning_rate=0.0010225099190844706, random_state=0) #
            classifiers["ADA"]["rna"][5] = AdaBoostClassifier(n_estimators=1267, learning_rate=0.004586076110566589, random_state=0) #
        #==ion==#
        elif (ligand == "ion"):
            classifiers["ADA"]["ion"][1] = AdaBoostClassifier(n_estimators=1204, learning_rate=0.03214735265313265, random_state=0) #
            classifiers["ADA"]["ion"][2] = AdaBoostClassifier(n_estimators=1009, learning_rate=0.022994984856708464, random_state=0) #
            classifiers["ADA"]["ion"][3] = AdaBoostClassifier(n_estimators=1298, learning_rate=0.011292634498495331, random_state=0) #
            classifiers["ADA"]["ion"][4] = AdaBoostClassifier(n_estimators=1204, learning_rate=0.03214735265313265, random_state=0) #
            classifiers["ADA"]["ion"][5] = AdaBoostClassifier(n_estimators=1267, learning_rate=0.004586076110566589, random_state=0) #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["ADA"]["peptide"][1] = AdaBoostClassifier(n_estimators=1054, learning_rate= 0.0003600754754068794, random_state=0) #
            classifiers["ADA"]["peptide"][2] = AdaBoostClassifier(n_estimators=187, learning_rate=0.004785420473599381, random_state=0) #
            classifiers["ADA"]["peptide"][3] = AdaBoostClassifier(n_estimators=949, learning_rate=0.17361539551031385, random_state=0) #
            classifiers["ADA"]["peptide"][4] = AdaBoostClassifier(n_estimators=1219, learning_rate=0.07472678614126843, random_state=0) #
            classifiers["ADA"]["peptide"][5] = AdaBoostClassifier(n_estimators=1072, learning_rate=0.01931430739645049, random_state=0) #
        #==sm==#
        elif (ligand == "sm"):
            classifiers["ADA"]["sm"][1] = AdaBoostClassifier(n_estimators=269, learning_rate=0.21061955317091516, random_state=0) #
            classifiers["ADA"]["sm"][2] = AdaBoostClassifier(n_estimators=1009, learning_rate=0.022994984856708464, random_state=0) #
            classifiers["ADA"]["sm"][3] = AdaBoostClassifier(n_estimators=1204, learning_rate=0.03214735265313265, random_state=0) #
            classifiers["ADA"]["sm"][4] = AdaBoostClassifier(n_estimators=1017, learning_rate=0.038674305888702416, random_state=0) #
            classifiers["ADA"]["sm"][5] = AdaBoostClassifier(n_estimators=1031, learning_rate=0.16361538260797742, random_state=0) #
    
    ###SVM###
    elif (classifier_method == "SVM"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["SVM"]["dna"][1] = SVC(kernel='rbf', class_weight=None, C=0.062786, gamma=0.001267, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["dna"][2] = SVC(kernel='rbf', class_weight="balanced", C=0.510614, gamma=0.001933, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["dna"][3] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["dna"][4] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["dna"][5] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
        #==rna==#
        elif (ligand == "rna"):
            classifiers["SVM"]["rna"][1] = SVC(kernel='rbf', class_weight=None, C=10.657846, gamma=0.000194, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["rna"][2] = SVC(kernel='rbf', class_weight={0: 10, 1: 1}, C=2.402881, gamma=0.000863, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["rna"][3] = SVC(kernel='poly', class_weight={0: 10, 1: 1}, C=1.533865, gamma=0.000152, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["rna"][4] = SVC(kernel='rbf', class_weight={0: 10, 1: 1}, C=2.402881, gamma=0.000863, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["rna"][5] = SVC(kernel='rbf', class_weight={0: 10, 1: 1}, C=0.024509, gamma=0.000426, probability=True, random_state=0, cache_size=200) #
        #==ion==#
        elif (ligand == "ion"):
            classifiers["SVM"]["ion"][1] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["ion"][2] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["ion"][3] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["ion"][4] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["ion"][5] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["SVM"]["peptide"][1] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C= 2.726349, gamma=0.000131, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["peptide"][2] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["peptide"][3] = SVC(kernel='rbf', class_weight="balanced", C=85.161446, gamma=0.000202, probability=True, random_state=0, cache_size=200) # 
            classifiers["SVM"]["peptide"][4] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) # 
            classifiers["SVM"]["peptide"][5] = SVC(kernel='rbf', class_weight="balanced", C=8.432559, gamma=0.000196, probability=True, random_state=0, cache_size=200) #
        #==sm==#
        elif (ligand == "sm"):
            classifiers["SVM"]["sm"][1] = SVC(kernel='rbf', class_weight="balanced", C=0.076938, gamma=0.000517, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["sm"][2] = SVC(kernel='rbf', class_weight="balanced", C=0.076938, gamma=0.000517, probability=True, random_state=0, cache_size=200) # 
            classifiers["SVM"]["sm"][3] = SVC(kernel='rbf', class_weight="balanced", C=0.076938, gamma=0.000517, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["sm"][4] = SVC(kernel='rbf', class_weight="balanced", C=1.273404, gamma=0.000196, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["sm"][5] = SVC(kernel='rbf', class_weight="balanced", C=0.076938, gamma=0.000517, probability=True, random_state=0, cache_size=200) # 

    ###Logistic###
    elif (classifier_method == "Logistic"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["Logistic"]["dna"][1] = LogisticRegression(C=0.001741, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["dna"][2] = LogisticRegression(C=0.001189, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["dna"][3] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") # 
            classifiers["Logistic"]["dna"][4] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") # 
            classifiers["Logistic"]["dna"][5] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") # 
        #==rna==#
        elif (ligand == "rna"):
            classifiers["Logistic"]["rna"][1] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["rna"][2] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["rna"][3] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["rna"][4] = LogisticRegression(C=0.001203, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["rna"][5] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
        #==ion==#
        elif (ligand == "ion"):
            classifiers["Logistic"]["ion"][1] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["ion"][2] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["ion"][3] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["ion"][4] = LogisticRegression(C=0.001203, random_state=0, n_jobs=-1, class_weight="balanced") # 
            classifiers["Logistic"]["ion"][5] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["Logistic"]["peptide"][1] = LogisticRegression(C=0.003279, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["peptide"][2] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["peptide"][3] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["peptide"][4] = LogisticRegression(C=0.001805, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["peptide"][5] = LogisticRegression(C=0.001805, random_state=0, n_jobs=-1, class_weight="balanced") #
        #==sm==#
        elif (ligand == "sm"):
            classifiers["Logistic"]["sm"][1] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["sm"][2] = LogisticRegression(C=0.002423, random_state=0, n_jobs=-1, class_weight=None) # 
            classifiers["Logistic"]["sm"][3] = LogisticRegression(C=0.002231, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["sm"][4] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["sm"][5] = LogisticRegression(C=0.002423, random_state=0, n_jobs=-1, class_weight=None) #
    
    ###KNN###
    elif (classifier_method == "KNN"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["KNN"]["dna"][1] = KNeighborsClassifier(n_neighbors=1030, n_jobs=-1, weights="uniform") ##
            classifiers["KNN"]["dna"][2] = KNeighborsClassifier(n_neighbors=1028, n_jobs=-1, weights="distance") ##
            classifiers["KNN"]["dna"][3] = KNeighborsClassifier(n_neighbors=880, n_jobs=-1, weights="uniform") ##
            classifiers["KNN"]["dna"][4] = KNeighborsClassifier(n_neighbors=820, n_jobs=-1, weights="uniform") ##
            classifiers["KNN"]["dna"][5] = KNeighborsClassifier(n_neighbors=818, n_jobs=-1, weights="uniform") ##
        #==rna==#
        elif (ligand == "rna"):
            classifiers["KNN"]["rna"][1] = KNeighborsClassifier(n_neighbors=685, n_jobs=-1, weights="distance") ##
            classifiers["KNN"]["rna"][2] = KNeighborsClassifier(n_neighbors=865, n_jobs=-1, weights="distance") ##
            classifiers["KNN"]["rna"][3] = KNeighborsClassifier(n_neighbors=838, n_jobs=-1, weights="uniform") ##
            classifiers["KNN"]["rna"][4] = KNeighborsClassifier(n_neighbors=748, n_jobs=-1, weights="distance") ##
            classifiers["KNN"]["rna"][5] = KNeighborsClassifier(n_neighbors=361, n_jobs=-1, weights="distance") ##
        #==ion==#
        elif (ligand == "ion"):
            classifiers["KNN"]["ion"][1] = KNeighborsClassifier(n_neighbors=35, n_jobs=-1, weights="distance") ##
            classifiers["KNN"]["ion"][2] = KNeighborsClassifier(n_neighbors=19, n_jobs=-1, weights="distance") ## 
            classifiers["KNN"]["ion"][3] = KNeighborsClassifier(n_neighbors=41, n_jobs=-1, weights="uniform") ##
            classifiers["KNN"]["ion"][4] = KNeighborsClassifier(n_neighbors=35, n_jobs=-1, weights="distance") ##
            classifiers["KNN"]["ion"][5] = KNeighborsClassifier(n_neighbors=42, n_jobs=-1, weights="distance") ##
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["KNN"]["peptide"][1] = KNeighborsClassifier(n_neighbors=68, n_jobs=-1, weights="distance") ##
            classifiers["KNN"]["peptide"][2] = KNeighborsClassifier(n_neighbors=110, n_jobs=-1, weights="distance") ##
            classifiers["KNN"]["peptide"][3] = KNeighborsClassifier(n_neighbors=15, n_jobs=-1, weights="distance") ##
            classifiers["KNN"]["peptide"][4] = KNeighborsClassifier(n_neighbors=38, n_jobs=-1, weights="distance") ##
            classifiers["KNN"]["peptide"][5] = KNeighborsClassifier(n_neighbors=258, n_jobs=-1, weights="distance") ##
        #==sm==#
        elif (ligand == "sm"):
            classifiers["KNN"]["sm"][1] = KNeighborsClassifier(n_neighbors=364, n_jobs=-1, weights="distance") #
            classifiers["KNN"]["sm"][2] = KNeighborsClassifier(n_neighbors=228, n_jobs=-1, weights="distance") #
            classifiers["KNN"]["sm"][3] = KNeighborsClassifier(n_neighbors=277, n_jobs=-1, weights="distance") #
            classifiers["KNN"]["sm"][4] = KNeighborsClassifier(n_neighbors=138, n_jobs=-1, weights="distance") #
            classifiers["KNN"]["sm"][5] = KNeighborsClassifier(n_neighbors=59, n_jobs=-1, weights="distance") #
    ###NN###
    elif (classifier_method == "NN"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["NN"]["dna"][1] = Net(dropout_parameter = 0.5, hidden_units_1=539, hidden_units_2=624, batch_size=295, learning_rate=0.000006, beta=0.975980,
                                              weight_decay=3.194446e-10, epoch_count=907, weight="balanced", input_size=no_features) #
            classifiers["NN"]["dna"][2] = Net(dropout_parameter = 0.5, hidden_units_1=457, hidden_units_2=671, batch_size=135, learning_rate=0.000003, beta=0.916482,
                                              weight_decay=9.976996e-17, epoch_count=1223, weight=0.1, input_size=no_features) #
            classifiers["NN"]["dna"][3] = Net(dropout_parameter = 0.5, hidden_units_1=539, hidden_units_2=624, batch_size=295, learning_rate=0.000006, beta=0.975980,
                                              weight_decay=3.194446e-10, epoch_count=372, weight="balanced", input_size=no_features) #
            classifiers["NN"]["dna"][4] = Net(dropout_parameter = 0.5, hidden_units_1=457, hidden_units_2=671, batch_size=135, learning_rate=0.000003, beta=0.916482,
                                              weight_decay=9.976996e-17, epoch_count=973, weight=0.1, input_size=no_features) #
            classifiers["NN"]["dna"][5] = Net(dropout_parameter = 0.5, hidden_units_1=939, hidden_units_2=498, batch_size=252, learning_rate=0.000004, beta=0.931570,
                                              weight_decay=1.918448e-24, epoch_count=822, weight=None, input_size=no_features) #  
        #==rna==#
        elif (ligand == "rna"):
            classifiers["NN"]["rna"][1] = Net(dropout_parameter = 0.5, hidden_units_1=539, hidden_units_2=624, batch_size=295, learning_rate=0.000006, beta=0.975980, 
                                              weight_decay=3.194446e-10, epoch_count=488, weight="balanced", input_size=no_features) #
            classifiers["NN"]["rna"][2] = Net(dropout_parameter = 0.5, hidden_units_1=930, hidden_units_2=744, batch_size=103, learning_rate=0.000002, beta=0.886173, 
                                              weight_decay=8.883323e-12, epoch_count=507, weight=None, input_size=no_features) #
            classifiers["NN"]["rna"][3] = Net(dropout_parameter = 0.5, hidden_units_1=821, hidden_units_2=878, batch_size=169, learning_rate=0.000091, beta=0.954622, 
                                              weight_decay=8.886669e-20, epoch_count=18, weight=None, input_size=no_features) #
            classifiers["NN"]["rna"][4] = Net(dropout_parameter = 0.5, hidden_units_1=784, hidden_units_2=549, batch_size=104, learning_rate=0.000001, beta=0.845870, 
                                              weight_decay=5.860976e-20, epoch_count=480, weight=None, input_size=no_features) #
            classifiers["NN"]["rna"][5] = Net(dropout_parameter = 0.5, hidden_units_1=982, hidden_units_2=610, batch_size=278, learning_rate=0.000014, beta=0.981022, 
                                              weight_decay=2.894894e-21, epoch_count=63, weight=None, input_size=no_features) #

        #==ion==#
        elif (ligand == "ion"):
            classifiers["NN"]["ion"][1] = Net(dropout_parameter = 0.5, hidden_units_1=347, hidden_units_2=934, batch_size=123, learning_rate=0.001957, beta=0.919507,
                                              weight_decay=4.178006e-09, epoch_count=7, weight=None, input_size=no_features) #
            classifiers["NN"]["ion"][2] = Net(dropout_parameter = 0.5, hidden_units_1=732, hidden_units_2=558, batch_size=33, learning_rate=0.001095, beta=0.932919,
                                              weight_decay=1.035851e-10, epoch_count=5, weight=0.1, input_size=no_features) #
            classifiers["NN"]["ion"][3] = Net(dropout_parameter = 0.5, hidden_units_1=250, hidden_units_2=923, batch_size=52, learning_rate=0.00028766277717993586, beta=0.8707470075441109,
                                              weight_decay=3.2602759556271237e-25, epoch_count=18, weight="balanced", input_size=no_features)
            classifiers["NN"]["ion"][4] = Net(dropout_parameter = 0.5, hidden_units_1=140, hidden_units_2=418, batch_size=263, learning_rate=0.0006838687356454338, beta=0.9460256784374906,
                                              weight_decay=1.0014770140328751e-23, epoch_count=42, weight=None, input_size=no_features) 
            classifiers["NN"]["ion"][5] = Net(dropout_parameter = 0.5, hidden_units_1=94, hidden_units_2=438, batch_size=229, learning_rate=0.00028939076697090865, beta=0.9229990562424056,
                                              weight_decay=1.5832929922838409e-18, epoch_count=8, weight=None, input_size=no_features) 

        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["NN"]["peptide"][1] = Net(dropout_parameter = 0.5, hidden_units_1=866, hidden_units_2=986, batch_size=83, learning_rate=0.0006418587699324594, beta=0.9231693756464692, 
                                                  weight_decay=9.662474065546658e-11, epoch_count=28, weight="balanced", input_size=no_features) 
            classifiers["NN"]["peptide"][2] = Net(dropout_parameter = 0.5, hidden_units_1=559, hidden_units_2=359, batch_size=147, learning_rate=0.0001252065381499946, beta=0.9357945617622756, 
                                                  weight_decay=7.679182193416932e-09, epoch_count=82, weight=None, input_size=no_features)  
            classifiers["NN"]["peptide"][3] = Net(dropout_parameter = 0.5, hidden_units_1=866, hidden_units_2=986, batch_size=83, learning_rate=0.0006418587699324594, beta=0.9231693756464692, 
                                                  weight_decay=9.662474065546658e-11, epoch_count=62, weight="balanced", input_size=no_features) 
            classifiers["NN"]["peptide"][4] = Net(dropout_parameter = 0.5, hidden_units_1=915, hidden_units_2=631, batch_size=43, learning_rate=3.5243908871012604e-05, beta=0.9179879456406768, 
                                                  weight_decay=5.0904150260183585e-08, epoch_count=34, weight="balanced", input_size=no_features) 
            classifiers["NN"]["peptide"][5] = Net(dropout_parameter = 0.5, hidden_units_1=923, hidden_units_2=932, batch_size=33, learning_rate=0.00011818138865906221, beta=0.8592027585611957, 
                                                  weight_decay=9.323094091596805e-24, epoch_count=50, weight="balanced", input_size=no_features) 
        #==sm==#
        elif (ligand == "sm"):
            classifiers["NN"]["sm"][1] = Net(dropout_parameter = 0.5, hidden_units_1=248, hidden_units_2=426, batch_size=103, learning_rate=0.00022920868800464477, beta=0.8949291648687738, 
                                             weight_decay=2.735659121919167e-22, epoch_count=24, weight=None, input_size=no_features) 
            classifiers["NN"]["sm"][2] = Net(dropout_parameter = 0.5, hidden_units_1=750, hidden_units_2=439, batch_size=201, learning_rate=1.0456259228180203e-05, beta=0.8959855883756007, 
                                             weight_decay=8.731434565883134e-22, epoch_count=67, weight=0.1, input_size=no_features) 
            classifiers["NN"]["sm"][3] = Net(dropout_parameter = 0.5, hidden_units_1=641, hidden_units_2=512, batch_size=30, learning_rate=4.645312461081778e-05, beta=0.8548690295975528, 
                                             weight_decay=1.7634816165605853e-08, epoch_count=44, weight=None, input_size=no_features) 
            classifiers["NN"]["sm"][4] = Net(dropout_parameter = 0.5, hidden_units_1=307, hidden_units_2=869, batch_size=113, learning_rate=0.0002089679702998887, beta=0.9277407561848752, 
                                             weight_decay=3.7489350711707085e-10, epoch_count=3, weight=None, input_size=no_features) 
            classifiers["NN"]["sm"][5] = Net(dropout_parameter = 0.5, hidden_units_1=523, hidden_units_2=858, batch_size=198, learning_rate=2.134952972083124e-05, beta=0.9492396398888632, 
                                             weight_decay=9.593155724387094e-12, epoch_count=27, weight="balanced", input_size=no_features) 

    with open("hyperparams_dict.pik", 'wb') as handle:
        pickle.dump(classifiers, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return classifiers