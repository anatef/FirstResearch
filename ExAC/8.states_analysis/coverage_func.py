import matplotlib.pyplot as plt
import numpy as np

#A function that creates a scatter subplot
def subplot_scatter(x_list, y_list, title, title_color):

    plt.scatter(x_list, y_list, c=y_list, cmap='plasma_r')

    # Add correlation line
    axes = plt.gca()
    m, b = np.polyfit(x_list, y_list, 1)
    X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
    plt.plot(X_plot, m*X_plot + b, '-', color="black")

    plt.xlim(0, 0.5)
    plt.ylim(0, max(y_list))

    #plt.xlabel("Adjusted AF", fontsize=10)
    #plt.ylabel("Mean Coverage", fontsize=10)
    plt.title(title, fontsize=14, color=title_color)
#-------------------------------------------------------------------------------------------#

#A function that plots one state coverage scatter
def plot_scatter(x_list, y_list, title, outfile, save_flag):

    plt.figure(figsize=(11,8))
    plt.scatter(x_list, y_list, c=y_list, cmap='plasma_r')

    # Add correlation line
    axes = plt.gca()
    m, b = np.polyfit(x_list, y_list, 1)
    X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
    plt.plot(X_plot, m*X_plot + b, '-', color="black")

    plt.xlim(0, max(x_list))
    plt.ylim(0, max(y_list))

    plt.xlabel("Adjusted AF", fontsize=16)
    plt.ylabel("Mean Coverage", fontsize=16)
    plt.title(title, fontsize=20)

    if (save_flag): plt.savefig(curr_dir[0]+"/plots/"+outfile+".pdf", bbox_inches="tight")
    plt.show()