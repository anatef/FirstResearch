import numpy as np
import random

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


def generate_trials_NN(no_trials, lr_list, lr_list_weights, batch_size_list, batch_size_list_weights, weight_decay_list, weight_decay_list_weights, beta_list, beta_list_weights, hidden_units_1_list, hidden_units_1_list_weights, hidden_units_2_list, hidden_units_2_list_weights):
    
    np.random.seed(0)
    random.seed(0)
    
    lr_interval_index = weighted_choice_sub(lr_list_weights)
    learning_rate_lb = lr_list[lr_interval_index][0]
    learning_rate_ub = lr_list[lr_interval_index][1]
    
    batch_size_interval_index = weighted_choice_sub(batch_size_list_weights)
    batch_size_lb = batch_size_list[batch_size_interval_index][0]
    batch_size_ub = batch_size_list[batch_size_interval_index][1]
    
    weight_decay_interval_index = weighted_choice_sub(weight_decay_list_weights)
    weight_decay_lb = weight_decay_list[weight_decay_interval_index][0]
    weight_decay_ub = weight_decay_list[weight_decay_interval_index][1]
    
    beta_interval_index = weighted_choice_sub(beta_list_weights)
    beta_lb = beta_list[beta_interval_index][0]
    beta_ub = beta_list[beta_interval_index][1]
    
    hidden_units_1_index = weighted_choice_sub(hidden_units_1_list_weights)
    hidden_units_1_lb = hidden_units_1_list[hidden_units_1_index][0]
    hidden_units_1_ub = hidden_units_1_list[hidden_units_1_index][1]
    
    hidden_units_2_index = weighted_choice_sub(hidden_units_2_list_weights)
    hidden_units_2_lb = hidden_units_2_list[hidden_units_2_index][0]
    hidden_units_2_ub = hidden_units_2_list[hidden_units_2_index][1]
    
    hyperparameter_trials = []
    trial_count = 0
    while trial_count != no_trials:

        trial_dict = {
                # learning_rate: started with 10^-6 to 10^-2
                # range for learning_rate: 10^-3 to 10^-2
                "learning_rate": 10 ** (np.random.uniform(learning_rate_lb,learning_rate_ub)),
                # batch_size: started with 50 to 300
                "batch_size": np.random.randint(batch_size_lb, batch_size_ub),
                # weight_decay: started with 10^-5 to 10^-25
                # range for weight_decay: 10^-17 to 10^-7
                "weight_decay": 10 ** (np.random.uniform(weight_decay_lb,weight_decay_ub)),
                # range for beta: 0.85-0.95                        
                "beta": np.random.uniform(beta_lb,beta_ub),
                # hidden units: started with (50, 300), (350,1000)
                "hidden_units_1": np.random.randint(hidden_units_1_lb,hidden_units_1_ub),
                "hidden_units_2": np.random.randint(hidden_units_2_lb,hidden_units_2_ub),
                "weight": random.choice(["balanced", "0.1", "None"])
               }
        if trial_dict in hyperparameter_trials: continue
        hyperparameter_trials.append(trial_dict)
        trial_count = trial_count + 1

    return hyperparameter_trials


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
                "scale_pos_weight": random.choice([1, "balanced", 0.1])
               }
        if trial_dict in hyperparameter_trials: continue
        hyperparameter_trials.append(trial_dict)
        trial_count = trial_count + 1

    return hyperparameter_trials

#====================================================================================================================#

def generate_trials_RF(no_trials, n_estimators_list, n_estimators_list_weights, max_depth_list, max_depth_list_weights, min_samples_leaf_list, min_samples_leaf_list_weights, min_samples_split_list, min_samples_split_list_weights):
    
    np.random.seed(0)
    random.seed(0)
    
    n_estimators_index = weighted_choice_sub(n_estimators_list_weights)
    n_estimators_lb = n_estimators_list[n_estimators_index][0]
    n_estimators_ub = n_estimators_list[n_estimators_index][1]
    
    max_depth_index =  weighted_choice_sub(max_depth_list_weights)
    max_depth_lb = max_depth_list[max_depth_index][0]
    max_depth_ub = max_depth_list[max_depth_index][1]
    
    min_samples_leaf_index = weighted_choice_sub(min_samples_leaf_list_weights)
    min_samples_leaf_lb = min_samples_leaf_list[min_samples_leaf_index][0]
    min_samples_leaf_ub = min_samples_leaf_list[min_samples_leaf_index][1]
    
    min_samples_split_index = weighted_choice_sub(min_samples_split_list_weights)
    min_samples_split_lb = min_samples_split_list[min_samples_split_index][0]
    min_samples_split_ub = min_samples_split_list[min_samples_split_index][1]
       
    
    hyperparameter_trials = []
    trial_count = 0
    while trial_count != no_trials:
    
        trial_dict = {
                "n_estimators": np.random.randint(n_estimators_lb,n_estimators_ub),
                "max_depth": np.random.randint(max_depth_lb, max_depth_ub),
                "min_samples_leaf": np.random.randint(min_samples_leaf_lb, min_samples_leaf_ub),
                "min_samples_split": np.random.randint(min_samples_split_lb, min_samples_split_ub),
                "class_weight": random.choice([None, "balanced", {0:10, 1:1}])
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
                "class_weight": random.choice([None, "balanced", {0:10, 1:1}])
               }
        if trial_dict in hyperparameter_trials: continue
        hyperparameter_trials.append(trial_dict)
        trial_count = trial_count + 1

    return hyperparameter_trials
#====================================================================================================================#
def generate_trials_KNN(no_trials, n_neighbors_list, n_neighbors_list_weights):
       
    np.random.seed(0)
    random.seed(0)
    
    n_neighbors_index = weighted_choice_sub(n_neighbors_list_weights)
    n_neighbors_lb = n_neighbors_list[n_neighbors_index][0]
    n_neighbors_ub = n_neighbors_list[n_neighbors_index][1]    
    
    hyperparameter_trials = []
    trial_count = 0
    while trial_count != no_trials:
       
        trial_dict = {
                "n_neighbors": np.random.randint(n_neighbors_lb,n_neighbors_ub),
                "weights": random.choice(["uniform", "distance"])
               }
        
        if trial_dict in hyperparameter_trials: continue
        hyperparameter_trials.append(trial_dict)
        trial_count = trial_count + 1
        
    return hyperparameter_trials

#====================================================================================================================#
def generate_trials_ADA(no_trials, n_estimators_list, n_estimators_list_weights, lr_list, lr_list_weights):
    
    np.random.seed(0)
    
    n_estimators_index = weighted_choice_sub(n_estimators_list_weights)
    n_estimators_lb = n_estimators_list[n_estimators_index][0]
    n_estimators_ub = n_estimators_list[n_estimators_index][1]
    
    lr_index = weighted_choice_sub(lr_list_weights)
    learning_rate_lb = lr_list[lr_index][0]
    learning_rate_ub = lr_list[lr_index][1]    
    
    
    hyperparameter_trials = []
    trial_count = 0
    while trial_count != no_trials:

        trial_dict = {
                "n_estimators": np.random.randint(n_estimators_lb,n_estimators_ub),
                "learning_rate": 10 ** (np.random.uniform(learning_rate_lb,learning_rate_ub))
               }
        if trial_dict in hyperparameter_trials: continue
        hyperparameter_trials.append(trial_dict)
        trial_count = trial_count + 1

    return hyperparameter_trials
#====================================================================================================================#
def generate_trials_SVM(no_trials, C_list, C_list_weights, gamma_list, gamma_list_weights):
    
    np.random.seed(0)
    random.seed(0)
    
    C_index = weighted_choice_sub(C_list_weights)
    C_lb = C_list[C_index][0]
    C_ub = C_list[C_index][1]
    
    gamma_index = weighted_choice_sub(gamma_list_weights)
    gamma_lb = gamma_list[gamma_index][0]
    gamma_ub = gamma_list[gamma_index][1]

    hyperparameter_trials = []
    trial_count = 0
    while trial_count != no_trials:

        trial_dict = {
                "C": 10 ** (np.random.uniform(C_lb,C_ub)), 
            "gamma": 10 ** (np.random.uniform(gamma_lb, gamma_ub)),
            "class_weight": random.choice([None, "balanced", {0:10, 1:1}]),
            "kernel": random.choice(["linear", "poly", "rbf", "sigmoid"])
        }
        if trial_dict in hyperparameter_trials: continue
        hyperparameter_trials.append(trial_dict)
        trial_count = trial_count + 1

    return hyperparameter_trials
