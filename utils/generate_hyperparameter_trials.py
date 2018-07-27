import numpy as np
import random

def generate_trials_NN(no_trials, lr_list, batch_size_list,weight_decay_list, beta_list, 
                                               hidden_units_1_list, hidden_units_2_list):
    
    np.random.seed(0)
    random.seed(0)

    hyperparameter_trials = []
    trial_count = 0
    while trial_count != no_trials:
        lr_interval = random.choice(lr_list)
        batch_size_interval = random.choice(batch_size_list)
        weight_decay_interval = random.choice(weight_decay_list)
        beta_interval = random.choice(beta_list)
        hidden_units_1_interval = random.choice(hidden_units_1_list)
        hidden_units_2_interval = random.choice(hidden_units_2_list)

        trial_dict = {
                # learning_rate: started with 10^-6 to 10^-2
                # range for learning_rate: 10^-3 to 10^-2
                "learning_rate": 10 ** (np.random.uniform(lr_interval[0],lr_interval[1])),
                # batch_size: started with 50 to 300
                "batch_size": np.random.randint(batch_size_interval[0], batch_size_interval[1]),
                # weight_decay: started with 10^-5 to 10^-25
                # range for weight_decay: 10^-17 to 10^-7
                "weight_decay": 10 ** (np.random.uniform(weight_decay_interval[0],weight_decay_interval[1])),
                # range for beta: 0.85-0.95                        
                "beta": np.random.uniform(beta_interval[0],beta_interval[1]),
                # hidden units: started with (50, 300), (350,1000)
                "hidden_units_1": np.random.randint(hidden_units_1_interval[0],hidden_units_1_interval[1]),
                "hidden_units_2": np.random.randint(hidden_units_2_interval[0],hidden_units_2_interval[1]),
                "weight": random.choice(["balanced", "0.1", "None"])
               }
        if trial_dict in hyperparameter_trials: continue
        hyperparameter_trials.append(trial_dict)
        trial_count = trial_count + 1

    return hyperparameter_trials


#====================================================================================================================#
def generate_trials_XGB(no_trials, max_depth_list, min_child_weight_list, 
                                                colsample_bytree_list, gamma_list, lr_list):
    
    np.random.seed(0)
    random.seed(0)
                   
    hyperparameter_trials = []
    trial_count = 0
    while trial_count != no_trials:
        
        max_depth_interval = random.choice(max_depth_list)
        min_child_weight_interval = random.choice(min_child_weight_list)
        colsample_bytree_interval = random.choice(colsample_bytree_list)
        gamma_interval = random.choice(gamma_list)
        lr_interval = random.choice(lr_list)

        trial_dict = {
                "max_depth": np.random.randint(max_depth_interval[0],max_depth_interval[1]),
                "min_child_weight": np.random.uniform(min_child_weight_interval[0],min_child_weight_interval[1]), 
                "colsample_bytree": np.random.uniform(colsample_bytree_interval[0],colsample_bytree_interval[1]), 
                "gamma": 10 ** (np.random.uniform(gamma_interval[0], gamma_interval[1])),
                "learning_rate": 10 ** (np.random.uniform(lr_interval[0],lr_interval[1])),
                "scale_pos_weight": random.choice([1, "balanced", 0.1])
               }
        if trial_dict in hyperparameter_trials: continue
        hyperparameter_trials.append(trial_dict)
        trial_count = trial_count + 1

    return hyperparameter_trials

#====================================================================================================================#

def generate_trials_RF(no_trials, n_estimators_list, max_depth_list, min_samples_leaf_list, min_samples_split_list):
    
    np.random.seed(0)
    random.seed(0)

    hyperparameter_trials = []
    trial_count = 0
    while trial_count != no_trials:
        
        n_estimators_interval = random.choice(n_estimators_list)
        max_depth_interval = random.choice(max_depth_list)
        min_samples_leaf_interval= random.choice(min_samples_leaf_list)
        min_samples_split_interval = random.choice(min_samples_split_list)

        trial_dict = {
                "n_estimators": np.random.randint(n_estimators_interval[0], n_estimators_interval[1]),
                "max_depth": np.random.randint(max_depth_interval[0], max_depth_interval[1]),
                "min_samples_leaf": np.random.randint(min_samples_leaf_interval[0], min_samples_leaf_interval[1]),
                "min_samples_split": np.random.randint(min_samples_split_interval[0], min_samples_split_interval[1]),
                "class_weight": random.choice([None, "balanced", {0:10, 1:1}])
                 }
        if trial_dict in hyperparameter_trials: continue
        hyperparameter_trials.append(trial_dict)
        trial_count = trial_count + 1

    return hyperparameter_trials
#====================================================================================================================#
def generate_trials_Log(no_trials, C_list):
    
    np.random.seed(0)
    random.seed(0)

    hyperparameter_trials = []
    trial_count = 0
    while trial_count != no_trials:
        
        C_interval = random.choice(C_list)

        trial_dict = {
                # logarithimic scale
                "C": 10 ** (np.random.uniform(C_interval[0],C_interval[1])),
                "class_weight": random.choice([None, "balanced", {0:10, 1:1}])
               }
        if trial_dict in hyperparameter_trials: continue
        hyperparameter_trials.append(trial_dict)
        trial_count = trial_count + 1

    return hyperparameter_trials
#====================================================================================================================#
def generate_trials_KNN(no_trials, n_neighbors_list):
    
    np.random.seed(0)
    random.seed(0)
    
    hyperparameter_trials = []
    trial_count = 0
    while trial_count != no_trials:
        
        n_neighbors_interval = random.choice(n_neighbors_list)
       
        trial_dict = {
                "n_neighbors": np.random.randint(n_neighbors_interval[0], n_neighbors_interval[1]),
                "weights": random.choice(["uniform", "distance"])
               }
        
        if trial_dict in hyperparameter_trials: continue
        hyperparameter_trials.append(trial_dict)
        trial_count = trial_count + 1
        
    return hyperparameter_trials

#====================================================================================================================#
def generate_trials_ADA(no_trials, n_estimators_list, lr_list):
    
    np.random.seed(0)
                
    hyperparameter_trials = []
    trial_count = 0
    while trial_count != no_trials:
        
        n_estimators_interval = random.choice(n_estimators_list)
        lr_interval = random.choice(lr_list)

        trial_dict = {
                "n_estimators": np.random.randint(n_estimators_interval[0],n_estimators_interval[1]),
                "learning_rate": 10 ** (np.random.uniform(lr_interval[0],lr_interval[1]))
               }
        if trial_dict in hyperparameter_trials: continue
        hyperparameter_trials.append(trial_dict)
        trial_count = trial_count + 1

    return hyperparameter_trials
#====================================================================================================================#
def generate_trials_SVM(no_trials, C_list, gamma_list):
    
    np.random.seed(0)
    random.seed(0)
                
    hyperparameter_trials = []
    trial_count = 0
    while trial_count != no_trials:
        
        C_interval = random.choice(C_list)
        gamma_interval = random.choice(gamma_list)

        trial_dict = {
                "C": 10 ** (np.random.uniform(C_interval[0], C_interval[1])), 
            "gamma": 10 ** (np.random.uniform(gamma_interval[0], gamma_interval[1])),
            "class_weight": random.choice([None, "balanced", {0:10, 1:1}]),
            "kernel": random.choice(["linear", "poly", "rbf", "sigmoid"])
        }
        if trial_dict in hyperparameter_trials: continue
        hyperparameter_trials.append(trial_dict)
        trial_count = trial_count + 1

    return hyperparameter_trials
