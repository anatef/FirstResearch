import numpy as np

#Reject everything 2 std from the median
def reject_outliers2(data, m = 2.):
    np_data = np.array(data)
    d = np.abs(np_data - np.median(np_data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return np_data[s<m]
#-------------------------------------------------------------------------------------------#

#Reject everything 2 std from the mean
def reject_outliers(data, m=2):
    np_data = np.array(data)
    return np_data[abs(np_data - np.mean(np_data)) < m * np.std(np_data)]
#-------------------------------------------------------------------------------------------#

def remove_outliers(states_dict):
    no_outliers_states_dict = states_dict.copy()
    for state in states_dict.keys():
        no_outliers_states_dict[state] = reject_outliers(states_dict[state])
    
    return no_outliers_states_dict