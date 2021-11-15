import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import loompy
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import LineCollection

## Import from cytograph
from cytograph.plotting.colors import colorize

def sample_distribution_plot(ds: loompy.LoomConnection, out_file: str) -> None:
    '''
    Generates a bubble plot to inspect distribution of cell counts across tissues.
    
    Args:
        ds                    Connection to the .loom file to use
        out_file              Name and location of the output file
        
    Remarks:
    
    '''

    tissue = np.unique(ds.ca.Tissue) 
    ages = np.unique(ds.ca.Age)
    df = pd.DataFrame([])

    for i, t in enumerate(tissue): 
        age, cells = np.unique(ds.ca.Age[ds.ca.Tissue == t], return_counts=True)

        data = pd.DataFrame({'Regions': t, 'Age': age, 'Cells': cells},
        columns=['Regions', 'Age', 'Cells'])
        df = df.append(data)
    
    factor = np.round(np.log10(np.max(df['Cells']))) - 4
    df['Bubble_size'] = df['Cells'] / 10**factor
    order = sorted(np.unique(df['Age']))
    df['Age'] = [order.index(x) for x in df['Age']]
    df = df.set_index(np.arange(0,df.shape[0])) 
    color = colorize(df.Age) 
    colors = colorize(ages)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    h = lambda c: plt.Line2D([], [], color=c, ls="", marker="o")

    scatter = ax.scatter('Regions', 'Age', c=color, s='Bubble_size', data=df)
    legend1 = ax.legend(handles=[h(colors[i]) for i in range(len(ages))], labels=list(order), 
                        bbox_to_anchor=(0.73, 0., 0.55, 1.0), title='Age', title_fontsize=10, frameon=False, fontsize=15)
    ax.add_artist(legend1)
    handles, labels = scatter.legend_elements(prop="sizes", num=5, color='lightgrey') 
    legend2 = ax.legend(handles, labels, bbox_to_anchor=(0.79, 0., 0.75, 1.0), 
                        labelspacing=1.8, title=f"Cells (x {int(10**factor)})", title_fontsize=10, frameon=False, fontsize=15)
#     plt.xticks(rotation=45, fontsize=15) 
    plt.yticks(range(len(order)), order, fontsize=10) 
    plt.yticks(fontsize=15) 
    ax.yaxis.grid(b=None, which='major', linewidth=0.5) 
    ax.set_axisbelow(True) 
    plt.title('Cells per age and region', fontsize=20, pad=20)

    plt.savefig(out_file, bbox_inches='tight')