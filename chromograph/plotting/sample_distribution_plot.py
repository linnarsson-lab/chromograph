import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import loompy
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import LineCollection

## Import from cytograph
from cytograph.plotting.colors import colorize

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import loompy
from sklearn.neighbors import NearestNeighbors
from matplotlib.collections import LineCollection

## Import from cytograph
from cytograph.visualization.colors import *

def sample_distribution_plot(ds: loompy.LoomConnection, out_file: str) -> None:
    '''
    Generates a bubble plot to inspect distribution of cell counts across tissues.
    
    Args:
        ds                    Connection to the .loom file to use
        out_file              Name and location of the output file
        
    Remarks:
    
    '''
    
    tissue = np.unique(ds.ca.Tissue) 
    all_ages = np.unique(ds.ca.Age)
    df = pd.DataFrame([])

    for i, t in enumerate(tissue):
        age = ds.ca.Age[np.where(ds.ca.Tissue==t)[0]].astype(int).flatten()
        sample = ds.ca.Shortname[np.where(ds.ca.Tissue==t)[0]].flatten()
        
        ages = []
        samples = []
        cells = []
        
        for s in np.unique(sample):
            a, c = np.unique(age[np.where(sample==s)[0]], return_counts=True)
            
            samples.append(s)
            ages.append(a[0])
            cells.append(c[0])

        data = pd.DataFrame({'Regions': np.repeat(t, len(samples)), 'Age': ages, 'Cells': cells, 'Samples': samples}, columns=['Regions', 'Age', 'Cells', 'Samples'])
        df = df.append(data)
    df = df.sort_values(['Age', 'Regions'], ascending=True)
    df = df.reset_index(drop=True)
    y_pos = {k:v for v,k in enumerate(np.unique(df['Age']))}
    df['Y'] = np.array([y_pos[x] for x in df['Age']])

    X = []
    Y = []
    age_row = np.unique(df['Age'])
    row = 0
    for i in range(df.shape[0]):
        if y_pos[df['Age'][i]] != row:
            row += 1
        n = np.sum(np.array(Y)==row)
        X.append(n)
        Y.append(row)
    df['X'] = np.array(X)
    
    factor = np.round(np.log10(np.max(df['Cells']))) - 2
    df['Bubble_size'] = df['Cells'] / 10**factor

    cls = Colorizer('regions').fit(df['Regions']).dict()
    fig, ax = plt.subplots(figsize=(5,5))
    scatter = ax.scatter(df['X'], df['Y'], s=df['Bubble_size'], c=[cls[i] for i in df['Regions']]);
    ax.get_xaxis().set_visible(False)
    labels = np.unique(df['Age'])
    labels = np.array([f'{x} pcw' for x in labels])
    plt.yticks(range(len(labels)), labels, fontsize=12);
    for pos in ['right','top', 'bottom']:
        plt.gca().spines[pos].set_visible(False)

    h = lambda c: plt.Line2D([], [], color=c, ls="", marker="o")

    handles, labels = scatter.legend_elements(prop="sizes", num=4, color='lightgrey') 
    legend2 = ax.legend(handles, labels, bbox_to_anchor=(1.35,1), labelspacing=1.5, 
                        title=f"Cells (x {int(10**factor)})", title_fontsize=12, frameon=False, fontsize=12)
    plt.title('Samples per age and region', fontsize=20, pad=20)
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    return

def cell_distribution_plot(ds: loompy.LoomConnection, out_file: str) -> None:
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

        data = pd.DataFrame({'Regions': t, 'Age': age, 'Cells': cells}, columns=['Regions', 'Age', 'Cells'])
        df = df.append(data)
    
    factor = np.round(np.log10(np.max(df['Cells']))) - 4
    df['Bubble_size'] = df['Cells'] / 10**factor
    order = sorted(np.unique(df['Age']))
    df['Age'] = [order.index(x) for x in df['Age']]
    df = df.set_index(np.arange(0,df.shape[0])) 
    color = colorize(df.Age) 
    colors = colorize(ages)

    fig, ax = plt.subplots(figsize=(15, 15), dpi=200)
    h = lambda c: plt.Line2D([], [], color=c, ls="", marker="o")

    scatter = ax.scatter('Regions', 'Age', c=color, s='Bubble_size', data=df)
    # legend1 = ax.legend(handles=[h(colors[i]) for i in range(len(ages))], labels=list(order), 
    #                     bbox_to_anchor=(0.63, 0., 0.55, 1.0), title='Age', title_fontsize=10, frameon=False, fontsize=15)
    legend1 = ax.legend(handles=[h(colors[i]) for i in range(len(ages))], labels=list(order), 
                        bbox_to_anchor=(1.05,1), title='Age', title_fontsize=16, frameon=False, fontsize=15)
    ax.add_artist(legend1)
    handles, labels = scatter.legend_elements(prop="sizes", num=5, color='lightgrey') 
    # legend2 = ax.legend(handles, labels, bbox_to_anchor=(0.69, 0., 0.75, 1.0), 
    #                     labelspacing=2.7, title=f"Cells (x {int(10**factor)})", title_fontsize=10, frameon=False, fontsize=15)
    legend2 = ax.legend(handles, labels, bbox_to_anchor=(1.35,1), labelspacing=2.7, 
                        title=f"Cells (x {int(10**factor)})", title_fontsize=16, frameon=False, fontsize=15)
    plt.yticks(range(len(order)), order, fontsize=18) 
    plt.xticks(fontsize=15) 
    ax.yaxis.grid(b=None, which='major', linewidth=0.5) 
    ax.set_axisbelow(True) 
    plt.title('Cells per age and region', fontsize=20, pad=20)

    plt.savefig(out_file, bbox_inches='tight', dpi=300)