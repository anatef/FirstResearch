
def create_sns_table_from_big_dict(states_dict):
    af_list = []
    af_adj_list = []
    state_list = []
    cov_mean = []
    
    for state in states_dict.keys():
        for i in range(len(states_dict[state])):
            d = states_dict[state][i]
            af_list.append(d["af"])
            af_adj_list.append(d["af_adj"])
            cov_mean.append(d["coverage_mean"])
            state_list.append(int(state))
    
    sns_table = pd.DataFrame([af_list, af_adj_list, cov_mean, state_list])
    sns_table = sns_table.transpose()
    sns_table.columns = ["af", "af_adj", "coverage_mean", "state"]
    return sns_table
#-------------------------------------------------------------------------------------------#

def create_sns_table_from_flat_dict(states_flat_dict, val_name):
    vals_list = []
    state_list = []
    
    for state in states_flat_dict.keys():
        for val in states_flat_dict[state]:
            vals_list.append(val)
            state_list.append(int(state))
        
    
    sns_table = pd.DataFrame([vals_list, state_list])
    sns_table = sns_table.transpose()
    sns_table.columns = [val_name, "state"]
    return sns_table
#-------------------------------------------------------------------------------------------#

def plot_box_plot(sns_table, val_name, title, outfile, save_flag):
    plt.figure(figsize=(20,8))
    sns.set_style("whitegrid")
    colors = ['dimgrey', 'dimgrey', 'deepskyblue', 'dimgrey', 'dimgrey', 'deepskyblue', 'dimgrey', 'dimgrey', 'dimgrey', 'dimgrey', 'dimgrey', 'coral', 
              'dimgrey', 'coral', 'coral', 'dimgrey', 'dimgrey', 'coral', 'deepskyblue', 'dimgrey', 'dimgrey', 'dimgrey', 'deepskyblue']

    ax = sns.boxplot(x="state", y=val_name, data=sns_table, width=1, palette=colors, showfliers=False)
    ax.set_xticklabels(states_dict.keys())

    plt.xlabel("HMM State", fontsize=16)
    plt.ylabel(val_name, fontsize=16)
    plt.title(title, fontsize=20)

    if (save_flag): plt.savefig(curr_dir[0]+"/plots/"+outfile+".pdf", bbox_inches="tight")

    plt.show()