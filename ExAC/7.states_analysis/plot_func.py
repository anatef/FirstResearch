import matplotlib.pyplot as plt
import numpy as np

#Plot the domain's states in their natural order
def plot_emd(emd_states_dict, plot_type_text, outfile, rev_flag, save_flag):
    
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
    
    csfont = {'fontname':'Cambria'}
    hfont = {'fontname':'Cambria'}
    
    plt.xticks(np.arange(1,states_num + 1), emd_states_dict.keys(), ha='left')
    plt.xlabel("HMM-States", fontsize=16, **csfont)
    plt.ylabel("EMD", fontsize=16, **csfont)
    plt.title("EMD to reach to all 0s distribution - "+plot_type_text, fontsize=20, **hfont)
    if (save_flag): plt.savefig(curr_dir[0]+"/plots/"+outfile+".pdf", bbox_inches="tight")
    plt.show()
#-------------------------------------------------------------------------------------------#

#Plot the domain's states sorted by value
def plot_sorted(vals_list, plot_title, plot_type_text, outfile, rev_flag, save_flag):
    
    states_num = len(emd_states_dict.keys())
    
    plt.figure(figsize=(15,8))
    zinc_colors = ['dimgrey', 'dimgrey', 'deepskyblue', 'dimgrey', 'dimgrey', 'deepskyblue', 'dimgrey', 'dimgrey', 'dimgrey', 'dimgrey', 'dimgrey', 'coral', 
              'dimgrey', 'coral', 'coral', 'dimgrey', 'dimgrey', 'coral', 'deepskyblue', 'dimgrey', 'dimgrey', 'dimgrey', 'deepskyblue']
    
    #Reversing to show low EMD is high bars
    if (rev_flag):
        hlim = max(vals_list)
        plot_vals = []
        for val in vals_list:
            plot_vals.append(hlim - val)
    else:
        plot_vals = vals_list
    
    #Sorting according to the values
    idx = np.array(plot_vals).argsort()
    vals_sorted = np.array(plot_vals)[idx]
    colors_sorted = np.array(zinc_colors)[idx]
    states_sorted = np.array(states_dict.keys())[idx]
    
    #Creating the plot
    plt.bar(states_dict.keys(), vals_sorted, color=colors_sorted)
    
    csfont = {'fontname':'Cambria'}
    hfont = {'fontname':'Cambria'}
    
    plt.xticks(np.arange(1,states_num + 1), states_sorted, ha='left')
    plt.xlabel("HMM-States", fontsize=16, **csfont)
    plt.ylabel("Non-zero MAFs", fontsize=16, **csfont)
    plt.title(plot_title+" - "+plot_type_text, fontsize=20, **hfont)
    if (save_flag): plt.savefig(curr_dir[0]+"/plots/"+outfile+".pdf", bbox_inches="tight")
    plt.show()
#-------------------------------------------------------------------------------------------#