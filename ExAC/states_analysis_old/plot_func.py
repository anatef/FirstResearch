import matplotlib.pyplot as plt
import numpy as np

#Plot the domain's states in their natural order
def plot_emd(emd_states_dict, plot_type_text, curr_dir, outfile, rev_flag, save_flag):
    
    states_num = len(emd_states_dict.keys())
    
    plt.figure(figsize=(15,8))
    zinc_colors = ['dimgrey', 'dimgrey', 'deepskyblue', 'dimgrey', 'dimgrey', 'deepskyblue', 'dimgrey', 'dimgrey', 'dimgrey', 'dimgrey', 'dimgrey', 'coral', 
              'dimgrey', 'coral', 'coral', 'dimgrey', 'dimgrey', 'coral', 'deepskyblue', 'dimgrey', 'dimgrey', 'dimgrey', 'deepskyblue']
    
    #Reversing to show low EMD is high bars
    if (rev_flag):
        hlim = max(emd_states_dict.values())
        plot_vals = []
        for val in emd_states_dict.values():
            plot_vals.append(hlim - val)
    else: 
        plot_vals = emd_states_dict.values()
        
    #Creating the plot
    plt.bar(emd_states_dict.keys(), plot_vals, color=zinc_colors)
    
    plt.xticks(np.arange(1,states_num + 1), emd_states_dict.keys(), ha='left')
    plt.xlabel("HMM-States", fontsize=16)
    plt.ylabel("EMD", fontsize=16)
    plt.title("EMD to reach to all 0s distribution - "+plot_type_text, fontsize=20)
    if (save_flag): plt.savefig(curr_dir[0]+"/plots/"+outfile+".pdf", bbox_inches="tight")
    plt.show()
#-------------------------------------------------------------------------------------------#

#Plot the domain's states sorted by value
def plot_sorted(vals_dict, plot_title, plot_type_text, curr_dir, outfile, rev_flag, save_flag):
    
    states_num = len(vals_dict.keys())
    
    plt.figure(figsize=(15,8))
    zinc_colors = ['dimgrey', 'dimgrey', 'deepskyblue', 'dimgrey', 'dimgrey', 'deepskyblue', 'dimgrey', 'dimgrey', 'dimgrey', 'dimgrey', 'dimgrey', 'coral', 
              'dimgrey', 'coral', 'coral', 'dimgrey', 'dimgrey', 'coral', 'deepskyblue', 'dimgrey', 'dimgrey', 'dimgrey', 'deepskyblue']
    
    #Reversing to show low EMD is high bars
    if (rev_flag):
        hlim = max(vals_dict.values())
        plot_vals = []
        for val in vals_dict.values():
            plot_vals.append(hlim - val)
    else:
        plot_vals = vals_dict.values()
    
    #Sorting according to the values
    idx = np.array(plot_vals).argsort()
    vals_sorted = np.array(plot_vals)[idx]
    colors_sorted = np.array(zinc_colors)[idx]
    states_sorted = np.array(vals_dict.keys())[idx]
    
    #Creating the plot
    plt.bar(vals_dict.keys(), vals_sorted, color=colors_sorted)
    
    plt.xticks(np.arange(1,states_num + 1), states_sorted, ha='left')
    plt.xlabel("HMM-States", fontsize=16)
    plt.ylabel("Non-zero MAFs", fontsize=16)
    plt.title(plot_title+" - "+plot_type_text, fontsize=20)
    if (save_flag): plt.savefig(curr_dir[0]+"/plots/"+outfile+".pdf", bbox_inches="tight")
    plt.show()
#-------------------------------------------------------------------------------------------#