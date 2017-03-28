import numpy as np
import matplotlib.pyplot as plt


def plot_line_graph():
    arrays = np.array([[401.6489758491516, 210.33531498908997, 144.43129706382751, 112.32910704612732, 133.5269730091095, 129.54033708572388, 138.04477715492249, 141.1620671749115],
                       [287.59411001205444, 150.31080603599548, 102.04712796211243, 80.09337592124939 , 90.166424036026, 92.31054401397705, 103.09541082382202,  100.63411903381348]])
                       #[ 282.3306028842926, 145.40629410743713, 101.71173214912415, 90.8415961265564, 85.4243369102478, 98.98426604270935, 98.31648898124695, 98.8639919757843]])
    labels = ['no caching', 'caching', 'caching and error']
    colors = ['ro-', 'bo-', 'go-']
    title_img =  'Comparing Cores with Time Optimized'
    #label = label_dict[title_img]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ticks = range(1,9)
    for i in range(len(arrays)):
        array = arrays[i]
        index = range(len(array))
        values = array
        ax.plot(ticks, values , colors[i])
        #ax.set_xlabel(tuning_parameter)
        ax.set_ylabel('Seconds')
        ax.set_xlabel('Cores')
    plt.title(title_img)
    plt.legend(labels, loc = 'upper right')

    plt.savefig(title_img+".pdf")

plot_line_graph()


#[[1, 282.3306028842926], [2, 145.40629410743713], [3, 101.71173214912415], [4, 90.8415961265564], [5, 85.4243369102478], [6, 98.98426604270935], [7, 98.31648898124695], [8, 98.8639919757843]]
