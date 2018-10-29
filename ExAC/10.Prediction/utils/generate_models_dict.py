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
            classifiers["XGB"]["dna"][1] = XGBClassifier(n_estimators=925, n_jobs=-1, random_state=0, max_depth=67, min_child_weight=4.962147616428278, colsample_bytree=0.3137508006361189, 
                                                       scale_pos_weight=1, gamma=0.0019203206073890719, learning_rate=0.016505459297554252) #
            classifiers["XGB"]["dna"][2] = XGBClassifier(n_estimators=371, n_jobs=-1, random_state=0, max_depth=84, min_child_weight=3.2294705653332807, colsample_bytree=0.5781904084470194, 
                                                       scale_pos_weight=scale_weight, gamma=0.4734989304499474, learning_rate=0.2359498261862276) #
            classifiers["XGB"]["dna"][3] = XGBClassifier(n_estimators=1054, n_jobs=-1, random_state=0, max_depth=26, min_child_weight=0.3551802909894347, colsample_bytree=0.31534697477615553, 
                                                       scale_pos_weight=scale_weight, gamma=0.0011498870747119448, learning_rate=0.08206717036367267) #
            classifiers["XGB"]["dna"][4] = XGBClassifier(n_estimators=4109, n_jobs=-1, random_state=0, max_depth=46, min_child_weight=2.045270476865308, colsample_bytree=0.372215819534954, 
                                                       scale_pos_weight=scale_weight, gamma=0.08246796389157786, learning_rate=0.005200782754291454) #
            classifiers["XGB"]["dna"][5] = XGBClassifier(n_estimators=2783, n_jobs=-1, random_state=0, max_depth=87, min_child_weight=3.315391015500504, colsample_bytree=0.447491782552863, 
                                                       scale_pos_weight=scale_weight, gamma=0.0011533284318779018, learning_rate=0.04511544946707904) #

        #==rna==#
        elif (ligand == "rna"):
            classifiers["XGB"]["rna"][1] = XGBClassifier(n_estimators=2177, n_jobs=-1, random_state=0, max_depth=46, min_child_weight=2.045270476865308, colsample_bytree=0.372215819534954, 
                                                       scale_pos_weight=scale_weight, gamma=0.08246796389157786, learning_rate=0.005200782754291454) #
            classifiers["XGB"]["rna"][2] = XGBClassifier(n_estimators=169, n_jobs=-1, random_state=0, max_depth=67, min_child_weight=4.962147616428278, colsample_bytree=0.3137508006361189, 
                                                       scale_pos_weight=1, gamma=0.0019203206073890719, learning_rate=0.016505459297554252) #
            classifiers["XGB"]["rna"][3] = XGBClassifier(n_estimators=1081, n_jobs=-1, random_state=0, max_depth=56, min_child_weight=3.437441381939077, colsample_bytree=0.4116307578351688, 
                                                       scale_pos_weight=0.1, gamma=0.6952032143696817, learning_rate=0.03614059711477347) #
            classifiers["XGB"]["rna"][4] = XGBClassifier(n_estimators=271, n_jobs=-1, random_state=0, max_depth=11, min_child_weight=2.488389892678253, colsample_bytree=0.5708757841236043, 
                                                       scale_pos_weight=0.1, gamma=0.2935474783881804, learning_rate=0.03835244294748661) #
            classifiers["XGB"]["rna"][5] = XGBClassifier(n_estimators=1364, n_jobs=-1, random_state=0, max_depth=67, min_child_weight=4.962147616428278, colsample_bytree=0.3137508006361189, 
                                                       scale_pos_weight=1, gamma=0.0019203206073890719, learning_rate=0.016505459297554252) #
        #==ion==#
        elif (ligand == "ion"):
            classifiers["XGB"]["ion"][1] = XGBClassifier(n_estimators=1689, n_jobs=-1, random_state=0, max_depth=33, min_child_weight=2.6092416087503585, colsample_bytree=0.5609964549928927, 
                                                       scale_pos_weight=1, gamma=0.00621823078201685, learning_rate=0.05126464012770091) #
            classifiers["XGB"]["ion"][2] = XGBClassifier(n_estimators=1406, n_jobs=-1, random_state=0, max_depth=40, min_child_weight=0.2901458016193781, colsample_bytree=0.5758124691685906, 
                                                       scale_pos_weight=1, gamma=0.008617626078400921, learning_rate=0.027365425850188276) #
            classifiers["XGB"]["ion"][3] = XGBClassifier(n_estimators=2588, n_jobs=-1, random_state=0, max_depth=59, min_child_weight=1.253113600969427, colsample_bytree=0.7580588721039582, 
                                                       scale_pos_weight=1, gamma=0.2699431266072476, learning_rate=0.0032547540399356784) #
            classifiers["XGB"]["ion"][4] = XGBClassifier(n_estimators=3540, n_jobs=-1, random_state=0, max_depth=70, min_child_weight=1.253113600969427, colsample_bytree=0.7580588721039582, 
                                                       scale_pos_weight=1, gamma=0.2699431266072476, learning_rate=0.0032547540399356784) #
            classifiers["XGB"]["ion"][5] = XGBClassifier(n_estimators=2643, n_jobs=-1, random_state=0, max_depth=35, min_child_weight=2.9596757909422484, colsample_bytree=0.6073579004123729, 
                                                       scale_pos_weight=1, gamma=0.029094018252643763, learning_rate=0.004025324835794404) #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["XGB"]["peptide"][1] = XGBClassifier(n_estimators=1365, n_jobs=-1, random_state=0, max_depth=40, min_child_weight=0.2901458016193781, colsample_bytree=0.5758124691685906, 
                                                       scale_pos_weight=1, gamma=0.008617626078400921, learning_rate=0.027365425850188276) #
            classifiers["XGB"]["peptide"][2] = XGBClassifier(n_estimators=3195, n_jobs=-1, random_state=0, max_depth=65, min_child_weight=3.31252285766338, colsample_bytree=0.43454738874322335, 
                                                       scale_pos_weight=0.1, gamma=0.09944371785022237, learning_rate=0.006465167933420622) #
            classifiers["XGB"]["peptide"][3] = XGBClassifier(n_estimators=758, n_jobs=-1, random_state=0, max_depth=94, min_child_weight=0.3135647601167285, colsample_bytree=0.5680241889173815, 
                                                       scale_pos_weight=0.1, gamma=0.0059710705243693095, learning_rate=0.0936772384485262) #
            classifiers["XGB"]["peptide"][4] = XGBClassifier(n_estimators=647, n_jobs=-1, random_state=0, max_depth=45, min_child_weight=2.9642230911250915, colsample_bytree=0.8831993114357629, 
                                                       scale_pos_weight=0.1, gamma=0.37483216628479255, learning_rate=0.09233813225851784) #
            classifiers["XGB"]["peptide"][5] = XGBClassifier(n_estimators=2044, n_jobs=-1, random_state=0, max_depth=75, min_child_weight=2.4540930642543897, colsample_bytree=0.6021237271959063, 
                                                       scale_pos_weight=0.1, gamma=0.4447621983542005, learning_rate=0.005121948527914932) #
        #==metabolite==#
    #     classifiers["XGB"]["metabolite"][1] = XGBClassifier(n_estimators=169, n_jobs=-1, random_state=0, max_depth=45, min_child_weight=1.185689, colsample_bytree=0.883199, 
    #                                                scale_pos_weight=0.1, gamma=0.374832, learning_rate=0.131259)
    #     classifiers["XGB"]["metabolite"][2] = XGBClassifier(n_estimators=1129, n_jobs=-1, random_state=0, max_depth=54, min_child_weight=1.56085330221, colsample_bytree=0.988774209162, 
    #                                                scale_pos_weight=0.1, gamma=0.1822673250862, learning_rate=0.00102357640439)
    #     classifiers["XGB"]["metabolite"][3] = XGBClassifier(n_estimators=821, n_jobs=-1, random_state=0, max_depth=26, min_child_weight=0.142072116396, colsample_bytree=0.315346974776, 
    #                                                scale_pos_weight=0.1, gamma=0.00114988707471, learning_rate=0.131259)
    #     classifiers["XGB"]["metabolite"][4] = XGBClassifier(n_estimators=1250, n_jobs=-1, random_state=0, max_depth=70, min_child_weight=0.721091120972, colsample_bytree=0.871492685917, 
    #                                                scale_pos_weight=0.1, gamma=0.595526012087, learning_rate=0.00130322162319)
    #     classifiers["XGB"]["metabolite"][5] = XGBClassifier(n_estimators=892, n_jobs=-1, random_state=0, max_depth=81, min_child_weight=1.91796544373, colsample_bytree=0.516526636354, 
    #                                                scale_pos_weight=0.1, gamma=0.0117522405231, learning_rate=0.00109855361287)
        #==sm==#
        elif (ligand == "sm"):
            classifiers["XGB"]["sm"][1] = XGBClassifier(n_estimators=2404, n_jobs=-1, random_state=0, max_depth=58, min_child_weight=0.7350694744440062, colsample_bytree=0.36990255035719394, 
                                                       scale_pos_weight=0.1, gamma=0.0024447643966464053, learning_rate=0.010622539541433905) #
            classifiers["XGB"]["sm"][2] = XGBClassifier(n_estimators=757, n_jobs=-1, random_state=0, max_depth=67, min_child_weight=4.962147616428278, colsample_bytree=0.3137508006361189, 
                                                       scale_pos_weight=1, gamma=0.0019203206073890719, learning_rate=0.016505459297554252) #
            classifiers["XGB"]["sm"][3] = XGBClassifier(n_estimators=1737, n_jobs=-1, random_state=0, max_depth=67, min_child_weight=4.962147616428278, colsample_bytree=0.3137508006361189, 
                                                       scale_pos_weight=1, gamma=0.0019203206073890719, learning_rate=0.016505459297554252) #
            classifiers["XGB"]["sm"][4] = XGBClassifier(n_estimators=1260, n_jobs=-1, random_state=0, max_depth=54, min_child_weight=2.8455036930729665, colsample_bytree=0.5553874729194997, 
                                                       scale_pos_weight=1, gamma=0.001612504689638591, learning_rate=0.02760582343926545) #
            classifiers["XGB"]["sm"][5] = XGBClassifier(n_estimators=2362, n_jobs=-1, random_state=0, max_depth=69, min_child_weight=2.9596757909422484, colsample_bytree=0.6073579004123729, 
                                                       scale_pos_weight=1, gamma=0.029094018252643763, learning_rate=0.004025324835794404) #
    
    ###RF###
    elif (classifier_method == "RF"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["RF"]["dna"][1] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["dna"][2] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["dna"][3] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["dna"][4] = RandomForestClassifier(n_estimators=518, max_depth=37, min_samples_leaf=31, min_samples_split=31, class_weight="balanced", n_jobs=-1, random_state=0) # 
            classifiers["RF"]["dna"][5] = RandomForestClassifier(n_estimators=886, max_depth=5, min_samples_leaf=32, min_samples_split=11, class_weight="balanced", n_jobs=-1, random_state=0) # 
        #==rna==#
        elif (ligand == "rna"):
            classifiers["RF"]["rna"][1] = RandomForestClassifier(n_estimators=51, max_depth=20, min_samples_leaf=41, min_samples_split=17, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["rna"][2] = RandomForestClassifier(n_estimators=789, max_depth=40, min_samples_leaf=48, min_samples_split=31, class_weight=None, n_jobs=-1, random_state=0) #
            classifiers["RF"]["rna"][3] = RandomForestClassifier(n_estimators=51, max_depth=20, min_samples_leaf=41, min_samples_split=17, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["rna"][4] = RandomForestClassifier(n_estimators=51, max_depth=20, min_samples_leaf=41, min_samples_split=17, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["rna"][5] = RandomForestClassifier(n_estimators=458, max_depth=84, min_samples_leaf=36, min_samples_split=26, class_weight="balanced", n_jobs=-1, random_state=0) #
        #==ion==#
        elif (ligand == "ion"):
            classifiers["RF"]["ion"][1] = RandomForestClassifier(n_estimators=296, max_depth=26, min_samples_leaf=3, min_samples_split=5, class_weight=None, n_jobs=-1, random_state=0) #
            classifiers["RF"]["ion"][2] = RandomForestClassifier(n_estimators=638, max_depth=65, min_samples_leaf=4, min_samples_split=10, class_weight=None, n_jobs=-1, random_state=0) #
            classifiers["RF"]["ion"][3] = RandomForestClassifier(n_estimators=296, max_depth=26, min_samples_leaf=3, min_samples_split=5, class_weight=None, n_jobs=-1, random_state=0) #
            classifiers["RF"]["ion"][4] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["ion"][5] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["RF"]["peptide"][1] = RandomForestClassifier(n_estimators=296, max_depth=26, min_samples_leaf=3, min_samples_split=5, class_weight=None, n_jobs=-1, random_state=0) #
            classifiers["RF"]["peptide"][2] = RandomForestClassifier(n_estimators=518, max_depth=37, min_samples_leaf=31, min_samples_split=31, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["peptide"][3] = RandomForestClassifier(n_estimators=959, max_depth=31, min_samples_leaf=4, min_samples_split=37, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["peptide"][4] = RandomForestClassifier(n_estimators=638, max_depth=65, min_samples_leaf=4, min_samples_split=10, class_weight=None, n_jobs=-1, random_state=0) #
            classifiers["RF"]["peptide"][5] = RandomForestClassifier(n_estimators=101, max_depth=2, min_samples_leaf=15, min_samples_split=37, class_weight=None, n_jobs=-1, random_state=0) #
        #==metabolite==#
    #     classifiers["RF"]["metabolite"][1] = RandomForestClassifier(n_estimators=928, max_depth=19, min_samples_leaf=35, min_samples_split=26, class_weight="balanced",
    #                                                    n_jobs=-1, random_state=0)
    #     classifiers["RF"]["metabolite"][2] = RandomForestClassifier(n_estimators=928, max_depth=19, min_samples_leaf=35, min_samples_split=26, class_weight="balanced",
    #                                                    n_jobs=-1, random_state=0)
    #     classifiers["RF"]["metabolite"][3] = RandomForestClassifier(n_estimators=928, max_depth=19, min_samples_leaf=35, min_samples_split=26, class_weight="balanced",
    #                                                    n_jobs=-1, random_state=0)
    #     classifiers["RF"]["metabolite"][4] = RandomForestClassifier(n_estimators=928, max_depth=19, min_samples_leaf=35, min_samples_split=26, class_weight="balanced",
    #                                                    n_jobs=-1, random_state=0)
    #     classifiers["RF"]["metabolite"][5] = RandomForestClassifier(n_estimators=928, max_depth=19, min_samples_leaf=35, min_samples_split=26, class_weight="balanced",
    #                                                    n_jobs=-1, random_state=0)
        #==sm==#
        elif (ligand == "sm"):
            classifiers["RF"]["sm"][1] = RandomForestClassifier(n_estimators=892, max_depth=57, min_samples_leaf=29, min_samples_split=36, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["sm"][2] = RandomForestClassifier(n_estimators=1214, max_depth=80, min_samples_leaf=16, min_samples_split=22, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["sm"][3] = RandomForestClassifier(n_estimators=1457, max_depth=63, min_samples_leaf=22, min_samples_split=35, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["sm"][4] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["sm"][5] = RandomForestClassifier(n_estimators=814, max_depth=50, min_samples_leaf=26, min_samples_split=5, class_weight="balanced", n_jobs=-1, random_state=0) #
    
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
        #==metabolite==#
    #     classifiers["ADA"]["metabolite"][1] = AdaBoostClassifier(n_estimators=670, learning_rate=0.065686, random_state=0)
    #     classifiers["ADA"]["metabolite"][2] = AdaBoostClassifier(n_estimators=670, learning_rate=0.065686, random_state=0)
    #     classifiers["ADA"]["metabolite"][3] = AdaBoostClassifier(n_estimators=670, learning_rate=0.065686, random_state=0)
    #     classifiers["ADA"]["metabolite"][4] = AdaBoostClassifier(n_estimators=670, learning_rate=0.065686, random_state=0)
    #     classifiers["ADA"]["metabolite"][5] = AdaBoostClassifier(n_estimators=670, learning_rate=0.065686, random_state=0)
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
            classifiers["SVM"]["dna"][1] = SVC(kernel='rbf', class_weight=None, C=85.16144627787163, gamma=2.559642090152666e-05, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["dna"][2] = SVC(kernel='rbf', class_weight=None, C=0.413599739383989, gamma=0.0015119336467640998, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["dna"][3] = SVC(kernel='rbf', class_weight=None, C=0.413599739383989, gamma=0.0015119336467640998, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["dna"][4] = SVC(kernel='rbf', class_weight=None, C= 0.413599739383989, gamma=0.0015119336467640998, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["dna"][5] = SVC(kernel='rbf', class_weight=None, C=0.413599739383989, gamma=0.0015119336467640998, probability=True, random_state=0, cache_size=200) #
        #==rna==#
        elif (ligand == "rna"):
            classifiers["SVM"]["rna"][1] = SVC(kernel='rbf', class_weight=None, C=0.8177482436211175, gamma=0.0005317086697454858, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["rna"][2] = SVC(kernel='rbf', class_weight="balanced", C=0.0023146629912496877, gamma=0.00010409405504965533, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["rna"][3] = SVC(kernel='poly', class_weight="balanced", C=2.246599608767013, gamma=0.001012272707448845, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["rna"][4] = SVC(kernel='rbf', class_weight=None, C=0.413599739383989, gamma=0.0015119336467640998, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["rna"][5] = SVC(kernel='rbf', class_weight="balanced", C=7.634611079196914, gamma=0.0003854987404324532, probability=True, random_state=0, cache_size=200) #
        #==ion==#
        elif (ligand == "ion"):
            classifiers["SVM"]["ion"][1] = SVC(kernel='rbf', class_weight="balanced", C=2.7371889009007244, gamma=0.0009158729542266177, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["ion"][2] = SVC(kernel='rbf', class_weight=None, C=0.8177482436211175, gamma=0.0005317086697454858, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["ion"][3] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=0.008089078990624672, gamma=0.0004539959494059672, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["ion"][4] = SVC(kernel='rbf', class_weight=None, C=19.34724728860398, gamma=0.002113934324402315, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["ion"][5] = SVC(kernel='rbf', class_weight=None, C=0.8177482436211175, gamma=0.0005317086697454858, probability=True, random_state=0, cache_size=200) #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["SVM"]["peptide"][1] = SVC(kernel='sigmoid', class_weight=None, C= 0.0002668187524237052, gamma=2.231090560744303e-05, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["peptide"][2] = SVC(kernel='rbf', class_weight=None, C=72.23651574885359, gamma=0.0007492121411354241, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["peptide"][3] = SVC(kernel='rbf', class_weight=None, C=2.4028805225920222, gamma=0.00017692124299043633, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["peptide"][4] = SVC(kernel='rbf', class_weight=None, C=0.00012963993440545012, gamma=0.0029548945587266834, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["peptide"][5] = SVC(kernel='poly', class_weight=None, C=0.0005124441683726437, gamma=0.003628140404024379, probability=True, random_state=0, cache_size=200) #
        #==metabolite==#
    #     classifiers["SVM"]["metabolite"][1] = SVC(kernel='rbf', class_weight=None, C=1.182, gamma=0.025, probability=True, random_state=0, cache_size=200)
    #     classifiers["SVM"]["metabolite"][2] = SVC(kernel='rbf', class_weight=None, C=1.182, gamma=0.025, probability=True, random_state=0, cache_size=200)
    #     classifiers["SVM"]["metabolite"][3] = SVC(kernel='rbf', class_weight=None, C=1.182, gamma=0.025, probability=True, random_state=0, cache_size=200)
    #     classifiers["SVM"]["metabolite"][4] = SVC(kernel='rbf', class_weight=None, C=1.182, gamma=0.025, probability=True, random_state=0, cache_size=200)
    #     classifiers["SVM"]["metabolite"][5] = SVC(kernel='rbf', class_weight=None, C=1.182, gamma=0.025, probability=True, random_state=0, cache_size=200)
        #==sm==#
        elif (ligand == "sm"):
            classifiers["SVM"]["sm"][1] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=0.05456347144827908, gamma=0.0018781738757161907, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["sm"][2] = SVC(kernel='rbf', class_weight="balanced", C=0.8291821660947628, gamma=0.00010307810128023636, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["sm"][3] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=0.0545634714482790, gamma=0.0018781738757161907, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["sm"][4] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=0.05456347144827908, gamma=0.0018781738757161907, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["sm"][5] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=0.05456347144827908, gamma=0.0018781738757161907, probability=True, random_state=0, cache_size=200) #

    ###Logistic###
    elif (classifier_method == "Logistic"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["Logistic"]["dna"][1] = LogisticRegression(C=0.015295398277813725, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["dna"][2] = LogisticRegression(C=0.0017414134181586204, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["dna"][3] = LogisticRegression(C=0.002972334644335654, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["dna"][4] = LogisticRegression(C=0.0032787264983352754, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["dna"][5] = LogisticRegression(C=0.0019237305096546488, random_state=0, n_jobs=-1, class_weight=None) #
        #==rna==#
        elif (ligand == "rna"):
            classifiers["Logistic"]["rna"][1] = LogisticRegression(C=0.0068471055766840426, random_state=0, n_jobs=-1, class_weight={0: 10, 1: 1}) #
            classifiers["Logistic"]["rna"][2] = LogisticRegression(C=0.001044195710380447, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["rna"][3] = LogisticRegression(C=0.13049073550362392, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["rna"][4] = LogisticRegression(C=0.0011889379831773006, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["rna"][5] = LogisticRegression(C=0.001044195710380447, random_state=0, n_jobs=-1, class_weight="balanced") #
        #==ion==#
        elif (ligand == "ion"):
            classifiers["Logistic"]["ion"][1] = LogisticRegression(C=0.02849988343697157, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["ion"][2] = LogisticRegression(C=0.02849988343697157, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["ion"][3] = LogisticRegression(C=0.01437554676473176, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["ion"][4] = LogisticRegression(C=0.056279320474151635, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["ion"][5] = LogisticRegression(C=0.0014346671987806314, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["Logistic"]["peptide"][1] = LogisticRegression(C=0.001204685241203032, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["peptide"][2] = LogisticRegression(C=0.055992236540633455, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["peptide"][3] = LogisticRegression(C=0.0011889379831773006, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["peptide"][4] = LogisticRegression(C=0.0341795291206101, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["peptide"][5] = LogisticRegression(C=0.001204685241203032, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
        #==metabolite==#
    #     classifiers["Logistic"]["metabolite"][1] = LogisticRegression(C=0.004859, random_state=0, n_jobs=-1, class_weight=None)
    #     classifiers["Logistic"]["metabolite"][2] = LogisticRegression(C=0.004859, random_state=0, n_jobs=-1, class_weight=None)
    #     classifiers["Logistic"]["metabolite"][3] = LogisticRegression(C=0.004859, random_state=0, n_jobs=-1, class_weight=None)
    #     classifiers["Logistic"]["metabolite"][4] = LogisticRegression(C=0.004859, random_state=0, n_jobs=-1, class_weight=None)
    #     classifiers["Logistic"]["metabolite"][5] = LogisticRegression(C=0.004859, random_state=0, n_jobs=-1, class_weight=None)
        #==sm==#
        elif (ligand == "sm"):
            classifiers["Logistic"]["sm"][1] = LogisticRegression(C=0.0044181257379025465, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["sm"][2] = LogisticRegression(C=0.0044181257379025465, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["sm"][3] = LogisticRegression(C=0.0054046235343238, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["sm"][4] = LogisticRegression(C=0.0044181257379025465, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["sm"][5] = LogisticRegression(C=0.009499535455183795, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
    
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
        #==metabolite==#
    #     classifiers["KNN"]["metabolite"][1] = KNeighborsClassifier(n_neighbors=994, n_jobs=-1, weights="distance") ##
    #     classifiers["KNN"]["metabolite"][2] = KNeighborsClassifier(n_neighbors=994, n_jobs=-1, weights="distance") ##
    #     classifiers["KNN"]["metabolite"][3] = KNeighborsClassifier(n_neighbors=461, n_jobs=-1, weights="distance") ##
    #     classifiers["KNN"]["metabolite"][4] = KNeighborsClassifier(n_neighbors=993, n_jobs=-1, weights="uniform") ##
    #     classifiers["KNN"]["metabolite"][5] = KNeighborsClassifier(n_neighbors=473, n_jobs=-1, weights="distance") ##
        #==druglike==#
    #     classifiers["KNN"]["druglike"][1] = KNeighborsClassifier(n_neighbors=211, n_jobs=-1, weights="distance") ##
    #     classifiers["KNN"]["druglike"][2] = KNeighborsClassifier(n_neighbors=356, n_jobs=-1, weights="distance") ##
    #     classifiers["KNN"]["druglike"][3] = KNeighborsClassifier(n_neighbors=367, n_jobs=-1, weights="distance") ##
    #     classifiers["KNN"]["druglike"][4] = KNeighborsClassifier(n_neighbors=213, n_jobs=-1, weights="distance") ##
    #     classifiers["KNN"]["druglike"][5] = KNeighborsClassifier(n_neighbors=115, n_jobs=-1, weights="distance") ##
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
            classifiers["NN"]["dna"][1] = Net(dropout_parameter = 0.5, hidden_units_1=184, hidden_units_2=353, batch_size=54, learning_rate=4.757861870028907e-05, beta=0.8920539466680099,
                                              weight_decay=8.078750523203191e-21, epoch_count=11, weight=0.1, input_size=no_features) #
            classifiers["NN"]["dna"][2] = Net(dropout_parameter = 0.5, hidden_units_1=53, hidden_units_2=444, batch_size=107, learning_rate=1.0473878850506294e-05, beta=0.9177816536796228,
                                              weight_decay=1.241393662311564e-25, epoch_count=167, weight="balanced", input_size=no_features) #
            classifiers["NN"]["dna"][3] = Net(dropout_parameter = 0.5, hidden_units_1=196, hidden_units_2=522, batch_size=69, learning_rate=7.727583480617052e-05, beta=0.9394863192178088,
                                              weight_decay=3.5931344867124515e-20, epoch_count=15, weight=0.1, input_size=no_features) #
            classifiers["NN"]["dna"][4] = Net(dropout_parameter = 0.5, hidden_units_1=252, hidden_units_2=594, batch_size=295, learning_rate=1.313029325614714e-05, beta=0.9258615624322356,
                                              weight_decay=5.5906794105295636e-15, epoch_count=97, weight=None, input_size=no_features) #
            classifiers["NN"]["dna"][5] = Net(dropout_parameter = 0.5, hidden_units_1=223, hidden_units_2=579, batch_size=280, learning_rate=1.3046207963856448e-05, beta=0.887432319663722,
                                              weight_decay=7.816787130242044e-21, epoch_count=129, weight=0.1, input_size=no_features) #     
        #==rna==#
        elif (ligand == "rna"):
            classifiers["NN"]["rna"][1] = Net(dropout_parameter = 0.5, hidden_units_1=199, hidden_units_2=551, batch_size=113, learning_rate=9.471186457972804e-05, beta=0.8524678728391331, 
                                              weight_decay=3.9090709338326653e-22, epoch_count=18, weight=0.1, input_size=no_features) #
            classifiers["NN"]["rna"][2] = Net(dropout_parameter = 0.5, hidden_units_1=283, hidden_units_2=416, batch_size=123, learning_rate=8.644631275506229e-05, beta=0.934903830842851, 
                                              weight_decay=1.4916994781206673e-20, epoch_count=51, weight=None, input_size=no_features) #  
            classifiers["NN"]["rna"][3] = Net(dropout_parameter = 0.5, hidden_units_1=196, hidden_units_2=522, batch_size=69, learning_rate=7.727583480617052e-05, beta=0.9394863192178088, 
                                              weight_decay=3.5931344867124515e-20, epoch_count=40, weight=0.1, input_size=no_features) #
            classifiers["NN"]["rna"][4] = Net(dropout_parameter = 0.5, hidden_units_1=283, hidden_units_2=416, batch_size=123, learning_rate=8.644631275506229e-05, beta=0.934903830842851, 
                                              weight_decay=1.4916994781206673e-20, epoch_count=37, weight=None, input_size=no_features) #
            classifiers["NN"]["rna"][5] = Net(dropout_parameter = 0.5, hidden_units_1=196, hidden_units_2=522, batch_size=69, learning_rate=7.727583480617052e-05, beta=0.9394863192178088, 
                                              weight_decay=3.5931344867124515e-20, epoch_count=28, weight=0.1, input_size=no_features) #    

        #==ion==#
        elif (ligand == "ion"):
            classifiers["NN"]["ion"][1] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
            classifiers["NN"]["ion"][2] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)    
            classifiers["NN"]["ion"][3] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
            classifiers["NN"]["ion"][4] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
            classifiers["NN"]["ion"][5] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)    

        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["NN"]["peptide"][1] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
            classifiers["NN"]["peptide"][2] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)    
            classifiers["NN"]["peptide"][3] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
            classifiers["NN"]["peptide"][4] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
            classifiers["NN"]["peptide"][5] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)  
        #==metabolite==#
    #     classifiers["NN"]["metabolite"][1] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    #     classifiers["NN"]["metabolite"][2] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)    
    #     classifiers["NN"]["metabolite"][3] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    #     classifiers["NN"]["metabolite"][4] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
    #     classifiers["NN"]["metabolite"][5] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)  

        #==sm==#
        elif (ligand == "sm"):
            classifiers["NN"]["sm"][1] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
            classifiers["NN"]["sm"][2] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)    
            classifiers["NN"]["sm"][3] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
            classifiers["NN"]["sm"][4] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)
            classifiers["NN"]["sm"][5] = Net(dropout_parameter = 0.5, hidden_units_1=204, hidden_units_2=986, batch_size=83, learning_rate=5e-5, beta=0.923169375646, weight_decay=9.66247406555e-11,input_size=no_features)  

    with open("hyperparams_dict.pik", 'wb') as handle:
        pickle.dump(classifiers, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return classifiers