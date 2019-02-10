import numpy as np
import random


ligands = ["dna", "rna", "ion", "peptide", "sm"]
models = ["SVM", "RF", "Logistic", "ADA", "KNN"]

#Taken from: https://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python
def weighted_choice_sub(weights):
    """
    A weighted random.choice function.
    
    @weights: a list with the weight for each index.
    return: an index from the list witb probability proportional to the weight"""
    random.seed(0)
    
    rnd = random.random() * sum(weights)
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i

#====================================================================================================================#
def generate_trials_XGB(no_trials, max_depth_list, max_depth_list_weights, min_child_weight_list, min_child_weight_list_weights, colsample_bytree_list, colsample_bytree_list_weights, gamma_list, gamma_list_weights, lr_list, lr_list_weights):
    
    np.random.seed(0)
    random.seed(0)
    
    max_depth_interval_index = weighted_choice_sub(max_depth_list_weights)
    max_depth_lb = max_depth_list[max_depth_interval_index][0]
    max_depth_ub = max_depth_list[max_depth_interval_index][1]
    
    min_child_weight_index = weighted_choice_sub(min_child_weight_list_weights)
    min_child_weight_lb = min_child_weight_list[min_child_weight_index][0]
    min_child_weight_ub = min_child_weight_list[min_child_weight_index][1]
    
    colsample_bytree_index = weighted_choice_sub(colsample_bytree_list_weights)
    colsample_bytree_lb = colsample_bytree_list[colsample_bytree_index][0]
    colsample_bytree_ub = colsample_bytree_list[colsample_bytree_index][1]
    
    gamma_index = weighted_choice_sub(gamma_list_weights)
    gamma_lb = gamma_list[gamma_index][0]
    gamma_ub = gamma_list[gamma_index][1]
    
    lr_index = weighted_choice_sub(lr_list_weights)
    learning_rate_lb = lr_list[lr_index][0]
    learning_rate_ub = lr_list[lr_index][1]    
    
    hyperparameter_trials = []
    trial_count = 0
    while trial_count != no_trials:

        trial_dict = {
                "max_depth": np.random.randint(max_depth_lb, max_depth_ub),
                "min_child_weight": np.random.uniform(min_child_weight_lb,min_child_weight_ub), 
                "colsample_bytree": np.random.uniform(colsample_bytree_lb,colsample_bytree_ub), 
                "gamma": 10 ** (np.random.uniform(gamma_lb, gamma_ub)),
                "learning_rate": 10 ** (np.random.uniform(learning_rate_lb,learning_rate_ub)),
                "scale_pos_weight": random.choice([1, "balanced", 0.1]),
                "ligands": random.sample(ligands, np.random.randint(1,5)),
                "models": random.sample(models, np.random.randint(1,5))
               }
        if trial_dict in hyperparameter_trials: continue
        hyperparameter_trials.append(trial_dict)
        trial_count = trial_count + 1

    return hyperparameter_trials

#====================================================================================================================#
def generate_trials_Log(no_trials, C_list, C_list_weights):
    
    np.random.seed(0)
    random.seed(0)
    
    C_index = weighted_choice_sub(C_list_weights)
    C_lb = C_list[C_index][0]
    C_ub = C_list[C_index][1]
    
    
    hyperparameter_trials = []
    trial_count = 0
    while trial_count != no_trials:

        trial_dict = {
                # logarithimic scale
                "C": 10 ** (np.random.uniform(C_lb,C_ub)),
                "class_weight": random.choice([None, "balanced", {0:10, 1:1}]),
                "ligands": random.sample(ligands, np.random.randint(1,5)),
                "models": random.sample(models, np.random.randint(1,5))
               }
        if trial_dict in hyperparameter_trials: continue
        hyperparameter_trials.append(trial_dict)
        trial_count = trial_count + 1

    return hyperparameter_trials