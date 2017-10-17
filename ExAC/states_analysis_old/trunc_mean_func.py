import numpy as np

def reject_trunc_mean(data, low_p, high_p):
    np_data = np.array(data)
    return np_data[low_p <= np_data][np_data <= high_p]
#-------------------------------------------------------------------------------------------#

def truncated_mean(states_dict, percentile):
    truncated_states_dict = states_dict.copy()
    for state in states_dict.keys():
        state_low_percentile = np.percentile(states_dict[state], percentile)
        state_high_percentile = np.percentile(states_dict[state], (100-percentile))
        truncated_states_dict[state] = reject_trunc_mean(states_dict[state], state_low_percentile, state_high_percentile)
    
    return truncated_states_dict
