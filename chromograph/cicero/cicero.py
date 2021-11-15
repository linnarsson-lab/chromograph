## Imports
import numpy as np
import os
import sys
import collections
import matplotlib.pyplot as plt
import loompy
from tqdm import tqdm

import warnings

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.stats as stats
import scipy
from scipy.spatial.distance import pdist, cdist

from chromograph.preprocessing.utils import *
from pybedtools import BedTool

from inverse_covariance import QuicGraphLasso
import pandas as pd
import igraph as ig
import pickle as pkl


def find_distance_parameter(ds: loompy.LoomConnection,
                            window_range,
                            maxit,
                            null_rho,
                            s,
                            distance_constraint,
                            distance_parameter_convergence,
                            verbose=False):

    pos = ds.ra.pos[window_range]
    dist_mat = cdist(pos.reshape(pos.shape[0],1), pos.reshape(pos.shape[0],1))
    
    if np.sum(dist_mat > distance_constraint)/2 < 1:
        if verbose:
            logging.info(f'No long edges')
        return

    ### Get the peak matrix and scale
    mat = ds['CPM'][window_range,:]
    mat -= mat.mean(axis=1).reshape(mat.shape[0],1)
    mat /= mat.std(axis=1).reshape(mat.shape[0],1)
    
    found = False
    starting_max = 2
    dist_param = 2
    dist_param_max = 2
    dist_param_min = 0
    it = 0
        
    while (found == False) & (it < maxit):  
        ## Get the penalty matrix
        rho = rho_matrix(dist_mat, dist_param, s)
        
        ## compute the regularized covariance matrix
        model = QuicGraphLasso(lam=rho, init_method='cov')
        GLasso = model.fit(mat.T);
                
        ## Calculate the number of long distance interactions
        big_entries = np.sum(dist_mat > distance_constraint)
        
        ## Check how many long distance interactions are nonzero
        if ((np.sum(GLasso.precision_[dist_mat > distance_constraint] != 0)/big_entries) > .05) or ((np.sum(GLasso.precision_ == 0)/big_entries) < .2):
            longs_zero = False
        else:
            longs_zero = True
            
        ## Update the distance parameter
        if (longs_zero != True) or (dist_param == 0):
            dist_param_min = dist_param
        else:
            dist_param_max = dist_param
           
        new_dist_param = (dist_param_min+dist_param_max)/2
        
        ## Scale up if too few interactions
        if new_dist_param == starting_max:
            new_dist_param = 2* starting_max
            starting_max = new_dist_param
            
        ## Check if we are done
        if distance_parameter_convergence > abs(dist_param-new_dist_param):
            found = True
        else:
            dist_param = new_dist_param
            
        it += 1
        
    if maxit == it:
        if verbose:
            logging.info(f'WARNING: HIT MAXIMUM ITERATIONS')
    return dist_param



def estimate_distance_parameter(ds: loompy.LoomConnection,
                                reference:str='GRCh38',
                                window = 5e5,
                                s=0.75,
                                sample_num = 100,
                                maxit=100,
                                distance_constraint = 2.5e5,
                                distance_parameter_convergence = 1e-22,
                                max_elements = 200,
                                max_sample_windows = 500,
                                verbose=False):

    ## Suppress depracation warning
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    chromosomes = get_chrom_sizes(reference)
    chrom_bins = generate_bins(chromosomes, window, 0.5)
    bins = [(k[0], str(k[1]), str(k[2])) for k in chrom_bins.keys()]
    filtered = BedTool(bins).subtract(BedTool(get_blacklist(reference)), A=True)

    positions = [(row['chrom'], int(row['start']), int(row['end'])) for row in filtered.sort()] 
    logging.info(f'Bins after cleaning: {len(positions)}')
    
    dist_param = []
    dist_params_calced = 0
    too_many = 0
    too_few = 0
    it = 0
    
    tbar = tqdm(total=sample_num)
    while (sample_num > dist_params_calced) & (it < max_sample_windows):
        it += 1
        win = positions[np.random.choice(len(positions), 1)[0]]
        win_range = (ds.ra.Chr == win[0]) & (ds.ra.pos > win[1]) & (ds.ra.pos < win[2])
        
        if np.sum(win_range)<=1:
            too_few += 1
            continue
        if np.sum(win_range)>max_elements:
            too_many += 1
            continue
        
        distance_parameter = find_distance_parameter(ds,
                                                     win_range, 
                                                     maxit=maxit, 
                                                     null_rho=0, 
                                                     s=s, 
                                                     distance_constraint = distance_constraint, 
                                                     distance_parameter_convergence = distance_parameter_convergence,
                                                     verbose = verbose) 
        dist_param.append(distance_parameter)
        dist_params_calced = dist_params_calced + 1
        tbar.update(1)
        
    ## Close tqdm
    tbar.close()
    if len(dist_param) < sample_num:
        logging.info(f'Could not calculate {sample_num} samples, actually calculated: {dist_params_calced}')

    if len(dist_param) == 0:
        logging.info(f'No Distances calculated')
            
    if verbose:
        logging.info(f'Skipped due to too many peaks: {too_many}, too few peaks: {too_few}')
    
    return [i for i in dist_param if i]

def Calculate_Grahpical_Lasso(ds: loompy.LoomConnection,
                             window,
                             dist_param,
                             s=0.75,
                             max_elements = 200,
                             verbose=False):
    
    ## Check the number of elements
    win_range = (ds.ra.Chr == window[0]) & (ds.ra.pos > window[1]) & (ds.ra.pos < window[2])
    if np.sum(win_range) <= 1:
        return
    if np.sum(win_range) > max_elements:
        return
    
    ## Get the distance matrix
    pos = ds.ra.pos[win_range]
    dist_mat = cdist(pos.reshape(pos.shape[0],1), pos.reshape(pos.shape[0],1))
    
    ### Get the peak matrix and scale
    mat = ds['CPM'][win_range,:]
    mat -= mat.mean(axis=1).reshape(mat.shape[0],1)
    mat /= mat.std(axis=1).reshape(mat.shape[0],1)
    
    ## Compute the rho matrix
    rho = rho_matrix(dist_mat, dist_param, s)
        
    ## compute the regularized covariance matrix
    model = QuicGraphLasso(lam=rho, init_method='cov')
    GLasso = model.fit(mat.T);
    
    return [np.where(win_range)[0], GLasso]
    
def Compute_Coacces(ds: loompy.LoomConnection,
                    reference:str='GRCh38',
                    window = 5e5,
                    alpha=.25,
                    s=0.75,
                    max_elements = 200,
                    verbose=False):
    
    ## Suppress depracation warning
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    chromosomes = get_chrom_sizes(reference)
    chrom_bins = generate_bins(chromosomes, window, 0.5)
    bins = [(k[0], str(k[1]), str(k[2])) for k in chrom_bins.keys()]
    filtered = BedTool(bins).subtract(BedTool(get_blacklist(reference)), A=True)

    positions = [(row['chrom'], int(row['start']), int(row['end'])) for row in filtered.sort()] 
    logging.info(f'Bins after cleaning: {len(positions)}')
    
    tbar = tqdm(total=len(positions))
    tbar.set_description("Running Graphical Lasso")
    X = []
    for win in positions:
        X.append(Calculate_Grahpical_Lasso(ds, win, alpha))
        tbar.update(1)
    tbar.close()

    ## Unify the matrix
    count_dict = {}
    inconsistent = 0

    tbar = tqdm(total = len(X))
    tbar.set_description("Unifying the matrix")
    for submat in X:
        if not submat == None:
            cov = submat[1].covariance_
            for i in range(len(submat[0])):
                for j in range(i+1, len(submat[0])):
                    row = submat[0][i]
                    col = submat[0][j]
                    if (row,col) not in count_dict:
                        if cov[i,j] != 0:
                            count_dict[(row,col)] = cov[i,j]
                    else:
                        if (cov[i,j] != 0) & (((cov[i,j]> 0) & (count_dict[(row,col)] > 0)) | 
                                                    ((cov[i,j] < 0) & (count_dict[(row,col)] < 0))):
                            count_dict[(row,col)] = np.mean([cov[i,j], count_dict[(row,col)]])
                        else:
                            del count_dict[(row,col)]
                            inconsistent += 1
        tbar.update(1)
    tbar.close()
    
    logging.info(f'Fraction inconsistent: {inconsistent/len(count_dict.keys())}')
    logging.info(f'Generating the matrix')
    
    ## Generating the matrix
    col = []
    row = []
    v = []
    for k in count_dict:
        col.append(k[0])
        row.append(k[1])
        v.append(count_dict[k])
    matrix = sparse.csr_matrix((v, (row,col)), shape=(ds.shape[0], ds.shape[0]), dtype='float')
    logging.info(f'Finished generating matrix')
    
    return matrix

def number_of_ccans(matrix, cut_off):
    sources, targets = matrix.nonzero()
    weights = matrix.data.flatten()  
    x = weights > cut_off
    G = ig.Graph(list(zip(sources[x],targets[x])), directed=False, edge_attrs={'weight': weights[x]})
    comp_membership = G.community_multilevel()
    
    return sum(np.array(comp_membership.sizes())>2)

def find_ccan_cutoff(matrix, tolerance_digits):
    logging.info(f'Finding ccan cutoff value')
    
    ## Make matrix non-negative
    sources, targets = matrix.nonzero()
    weights = matrix.data.flatten()
    x = weights > 0
    matrix = sparse.csr_matrix((weights[x], (sources[x],targets[x])), shape=matrix.shape, dtype='float')
    
    ## Find correct threshold
    tolerance = 10**-tolerance_digits
    bottom = 0
    top = 1
    while (top-bottom) > tolerance:
        test_val = bottom + np.round((top-bottom)/2, tolerance_digits+1)
        ccan_num_test = number_of_ccans(matrix, test_val)
        next_step = test_val
        
        ccan_num_test2 = ccan_num_test
        while ccan_num_test2 == ccan_num_test:
            next_step = next_step + (top-bottom)/10
            ccan_num_test2 = number_of_ccans(matrix, next_step)
        
        if ccan_num_test > ccan_num_test2:
            top = test_val
        else:
            bottom = test_val
        logging.info(f"Test val: {test_val}")
        
    return np.round((top+bottom)/2, tolerance_digits)

def generate_ccans(matrix,
                   peaks:np.array,
                   coaccess_cutoff_override: int = None,
                   tolerance_digits: int = 2):
    
    if coaccess_cutoff_override != None:
        assert (coaccess_cutoff_override <= 1) & (coaccess_cutoff_override >= 0), "Cutoff value must be between 0 and 1"
        
    if coaccess_cutoff_override != None:
        coaccess_cutoff = coaccess_cutoff_override
        logging.info(f'Override cutoff value: {coaccess_cutoff_override}')
    else:
        coaccess_cutoff = find_ccan_cutoff(matrix, tolerance_digits)
        logging.info(f'Coaccessibility cutoff set empirically at: {coaccess_cutoff}')
    
    ## Make the ccan graph
    sources, targets = matrix.nonzero()
    weights = matrix.data.flatten()
    x = weights > coaccess_cutoff
    sources, targets, weights = sources[x], targets[x], weights[x]
    
    ## Cluster
    G = ig.Graph(list(zip(sources,targets)), directed=False, edge_attrs={'weight': weights})
    comp_membership = G.community_multilevel()
    sizes = np.array(comp_membership.sizes())>2
    comp_list = comp_membership.membership
    valids = np.where(sizes)[0]
    
    peaks1 = ds.ra.ID[sources]
    peaks2 = ds.ra.ID[targets]
    
    df = pd.DataFrame({'peak1':peaks1, 'peak2':peaks2, 'CCAN': np.array(comp_list)[sources], 'coaccess': weights})
    sub = df['CCAN'].isin(valids)
    df = df[sub]
    reorder_dict = {c: i for i, c in enumerate(np.unique(df.CCAN))}
    
    df['CCAN'] = [reorder_dict[i] for i in df['CCAN']]
    
    filtered_matrix = sparse.csr_matrix((weights[sub], (sources[sub],targets[sub])), shape=matrix.shape, dtype='float')
    
    logging.info(f"Total Networks: {len(np.unique(df['CCAN']))}, Positive connections: {len(df['peak1'])}")
    return df, filtered_matrix

def save_connections(ds, df, outdir):
    ## Save pd dataframe
    f_CCANs = os.path.join(outdir, 'CCANs.pkl')
    all_arcs = os.path.join(outdir, 'all.arcs')
    prom_arcs = os.path.join(outdir, 'proms.arcs')
    
    ## Save connections
    df.to_pickle(f_CCANs)
    
    with open(all_arcs, 'a') as file:
        for index, row in df.iterrows():
            new_line = []
            [new_line.extend(row[p].replace(':', '-').split('-')) for p in ['peak1', 'peak2']]
            new_line.append(str(round(row['coaccess'], 2)))
            file.write('\t'.join(new_line))
            file.write('\n')
            
    ## Save promoters
    X = np.array([x.split(' ')[0] for x in ds.ra.Annotation])
    TSS_pos = np.where(X=='promoter-TSS')[0]
    proms = ds.ra.ID[TSS_pos]
    df_prom = df[df['peak1'].isin(proms) | df['peak2'].isin(proms)]
    
    with open(prom_arcs, 'a') as file:
        for index, row in df_prom.iterrows():
            new_line = []
            [new_line.extend(row[p].replace(':', '-').split('-')) for p in ['peak1', 'peak2']]
            new_line.append(str(round(row['coaccess'], 2)))
            file.write('\t'.join(new_line))
            file.write('\n')

def generate_Gene_Activity(ds, matrix, dist_thresh:int=2.5e5):
    logging.info(f'Starting Gene activity calculation')
    out_file = '/' + os.path.join(*ds.filename.split("/")[:-1], f'{ds.filename.split("/")[-2]}_GA.loom')
    
    ## Check if position coords already exist
    if not 'pos' in ds.ra:
        ds.ra.pos = np.ceil((ds.ra.Start.astype(int) + ds.ra.End.astype(int))/2).astype(int)
    
    ## Extract the TSS peaks
    TSS_pos = np.where(['TSS' in x for x in ds.ra['Detailed Annotation']])[0]

    ## Calculate the distances between associated peaks
    sources, targets = matrix.nonzero()
    weights = matrix.data.flatten()  
    pos1 = ds.ra.pos[sources].reshape((sources.shape[0],1))
    pos2 = ds.ra.pos[targets].reshape((targets.shape[0], 1))
    dists = abs(pos1-pos2).flatten()

    ## Filter out peaks that are not associated with a TSS or too far away
    TSS_pos_s = set(TSS_pos)
    v1 = [x in TSS_pos_s for x in sources]
    v2 = [x in TSS_pos_s for x in targets]
    valids = [x ^ y for x,y in zip(v1, v2)] & (dists < dist_thresh)

    ## Generate the promoter connectivity matrix
    logging.info(f'Getting matrices')
    promoter_conn_matrix = sparse.csr_matrix((weights[valids], (sources[valids],targets[valids])), shape=matrix.shape, dtype='float')
    promoter_conn_matrix[targets[valids], sources[valids]] = weights[valids]
    promoter_conn_matrix.setdiag(1)
    promoter_conn_matrix = promoter_conn_matrix[TSS_pos,:]

    ## Distal peaks
    peaks = np.unique(np.concatenate([promoter_conn_matrix.nonzero()]))
    distal_peaks = peaks
     
    ## Weigh peaks by their connectivity to the TSS
    scaled_site_weights = np.ones(distal_peaks.shape[0])
    total_linked_site_weights = promoter_conn_matrix.tocsr()[:,distal_peaks]
    total_linked_site_weights = np.asarray(div0(1, np.sum(total_linked_site_weights, axis=1))).reshape(-1)
    total_linked_site_weights = np.diag(total_linked_site_weights)
    scaled_site_weights = total_linked_site_weights @ promoter_conn_matrix[:,distal_peaks]
    scaled_site_weights[scaled_site_weights>1] = 1

    ## Check if file already exists
    if os.path.isfile(out_file):
        os.remove(out_file)
    
    logging.info(f'Generating file')   
    
    rows = {k: ds.ra[k][TSS_pos] for k in ds.ra}
    M = len(TSS_pos)
    
    empty_mat = sparse.csr_matrix((M,ds.shape[1]), dtype=np.float32)
    logging.info(f'Create file')
    loompy.create(out_file, empty_mat, rows, ds.ca)
    with loompy.connect(out_file) as dsout:

        ## Transfer column_graphs
        for k in ds.col_graphs:
            dsout.col_graphs[k] = ds.col_graphs[k]
        
        ## Generate Gene Accessibility Scores
        logging.info(f'Generating gene accessibility scores')
        progress = tqdm(total = ds.shape[1])
        for (ix, selection, view) in ds.scan(axis=1):
            X = view[''][:,:][distal_peaks,:].T @ scaled_site_weights.T
            dsout[:,selection] = X.T
            progress.update(512)
        progress.close()
        
        knn = dsout.col_graphs['KNN'].astype("bool")
        
        ## Start pooling over the network
        logging.info(f'Start pooling over network')
        dsout["pooled"] = 'float32'
        progress = tqdm(total = dsout.shape[0])
        for (_, indexes, view) in dsout.scan(axis=0, layers=[""], what=["layers"]):
            dsout["pooled"][indexes.min(): indexes.max() + 1, :] = view[:, :] @ knn.T 
            progress.update(512)
        progress.close()