import numpy as np

def generate_trials_NN(no_trials, learning_rate_ub, learning_rate_lb, batch_size_ub, 
                       batch_size_lb,weight_decay_ub, weight_decay_lb, beta_ub, beta_lb, 
                       hidden_units_1_ub, hidden_units_1_lb, hidden_units_2_ub, hidden_units_2_lb):
    np.random.seed(0)
    hyperparameter_trials = []
    for trial in range(no_trials):

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
                "hidden_units_2": np.random.randint(hidden_units_2_lb,hidden_units_2_ub)            
               }
        hyperparameter_trials.append(trial_dict)

    return hyperparameter_trials


#====================================================================================================================#
def generate_trials_XGB(no_trials, n_estimators_ub, n_estimators_lb,
                       max_depth_ub, max_depth_lb, min_child_weight_ub,
                       min_child_weight_lb, colsample_bytree_ub, colsample_bytree_lb,
                       gamma_ub, gamma_lb, learning_rate_lb, learning_rate_ub):
    np.random.seed(0)
    hyperparameter_trials = []
    for trial in range(no_trials):

        trial_dict = {
                "n_estimators": np.random.randint(n_estimators_lb,n_estimators_ub),
                "max_depth": np.random.randint(max_depth_lb, max_depth_ub),
                "min_child_weight": np.random.uniform(min_child_weight_lb,min_child_weight_ub), 
                "colsample_bytree": np.random.uniform(colsample_bytree_lb,colsample_bytree_ub), 
                "gamma": 10 ** (np.random.uniform(gamma_lb, gamma_ub)),
                "learning_rate": 10 ** (np.random.uniform(learning_rate_lb,learning_rate_ub))
               }
        hyperparameter_trials.append(trial_dict)

    return hyperparameter_trials

#====================================================================================================================#

def generate_trials_RF(no_trials, n_estimators_ub, n_estimators_lb,
                      max_depth_ub, max_depth_lb, min_samples_leaf_ub,
                      min_samples_leaf_lb, min_samples_split_ub, min_samples_split_lb):
    np.random.seed(0)
    hyperparameter_trials = []
    for trial in range(no_trials):

        trial_dict = {
                "n_estimators": np.random.randint(n_estimators_lb,n_estimators_ub),
                "max_depth": np.random.randint(max_depth_lb, max_depth_ub),
                "min_samples_leaf": np.random.randint(min_samples_leaf_lb, min_samples_leaf_ub),
                "min_samples_split": np.random.randint(min_samples_split_lb, min_samples_split_ub)
               }
        hyperparameter_trials.append(trial_dict)

    return hyperparameter_trials
#====================================================================================================================#
def generate_trials_Log(no_trials, C_ub, C_lb):
    np.random.seed(0)
    hyperparameter_trials = []
    for trial in range(no_trials):

        trial_dict = {
                # logarithimic scale
                "C": 10 ** (np.random.uniform(C_lb,C_ub)),  
               }
        hyperparameter_trials.append(trial_dict)

    return hyperparameter_trials
#====================================================================================================================#
def generate_trials_KNN(no_trials, n_neighbors_ub, n_neighbors_lb):
    np.random.seed(0)
    hyperparameter_trials = []
    for trial in range(no_trials):

        trial_dict = {
                "n_neighbors": np.random.randint(n_neighbors_lb,n_neighbors_ub),  
               }
        hyperparameter_trials.append(trial_dict)

    return hyperparameter_trials

#====================================================================================================================#
def generate_trials_ADA(no_trials, n_estimators_ub, n_estimators_lb,
                       learning_rate_lb,learning_rate_ub):
    np.random.seed(0)
    hyperparameter_trials = []
    for trial in range(no_trials):

        trial_dict = {
                "n_estimators": np.random.randint(n_estimators_lb,n_estimators_ub),
                "learning_rate": 10 ** (np.random.uniform(learning_rate_lb,learning_rate_ub))
               }
        hyperparameter_trials.append(trial_dict)

    return hyperparameter_trials
#====================================================================================================================#
def generate_trials_SVM(no_trials, C_ub, C_lb, gamma_ub, gamma_lb):
    np.random.seed(0)
    hyperparameter_trials = []
    for trial in range(no_trials):

        trial_dict = {
                "C": 10 ** (np.random.uniform(C_lb,C_ub)), 
            "gamma": 10 ** (np.random.uniform(gamma_lb, gamma_ub))
        }
        hyperparameter_trials.append(trial_dict)

    return hyperparameter_trials
