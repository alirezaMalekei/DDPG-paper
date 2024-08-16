import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_curve(data, fig_path, colors, names, xlabel, ylabel, ys=None):
    plt.figure(figsize=(20, 10), dpi=300)
    i_s = np.arange(len(names)).tolist()
    for (history, color, name, i) in zip(data, colors, names, i_s):
        y = np.array(history)
        if ys is None:
           episodes = np.arange(len(history))
        else:
            episodes = np.array(ys)
        plt.plot(episodes, y, color, label=name, linewidth=2.5)
        plt.legend(loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.rc('axes', titlesize=18)    
    plt.rc('axes', labelsize=18)
    plt.rc('legend', fontsize=16)
    plt.grid()
    plt.savefig(fig_path)
    plt.show()
    
def plot_clustered_column(cases, methods, values, fig_path, xlabel, ylabel, bar_width=0.1, colors=None):
    num_cases   = len(cases)
    num_methods = len(methods)
    
    data = {
        'case': np.tile(cases, num_methods),
        'method': np.repeat(methods, num_cases),
        'value': values
        }

    df = pd.DataFrame(data)

    width = bar_width # width of each bar  
    
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    
    if colors is None:
        colors = plt.cm.get_cmap('tab10', num_methods).colors 
    
    for i, method in enumerate(methods):
        values = df[df['method'] == method]['value']
        positions = np.arange(len(cases)) + i * width
        ax.bar(positions, values, width=width, label=str(method), color=colors[i % len(colors)])

    ax.set_xticks(np.arange(len(cases)) + width * (num_methods - 1) / 2)
    ax.set_xticklabels(cases)
    ax.legend(loc='upper right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(fig_path)
    plt.show()
     
def plot_clustered_column2(methods, data, fig_path, xlabel, ylabel, bar_width=0.1, colors=None):
        
    values_list = []
    vertical_arr = np.array(data)

    for i in range(vertical_arr.shape[0]):
          arr = vertical_arr[i, :]
          values_list = values_list + arr.tolist()

    cases = np.arange(len(values_list) / len(methods)).tolist()
    
    plot_clustered_column(cases, methods, values_list, fig_path, xlabel, ylabel, bar_width=bar_width, colors=colors)
