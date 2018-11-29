import numpy as np
import pandas as pd
import time
import pickle
from multiprocessing import Pool
from functools import partial
from nilearn.signal import clean
from problem import get_train_data

# ref: https://github.com/MRegina/DTW_for_fMRI

def _load_fmri_motion_correction(fmri_filenames, fmri_motions):
    """
    load frmi files with correction to confounding factor 'fmri_motions'
    
    PARAMETERS:
        fmri_filenames - eg. relative addresses from data_train['fmri_msdl'], one column per input
        fmri_motions - the txt version confounding factor data_train['fmri_motions']
    RETURNS:
        fmri - list of standardized + confounding factor removed fmri data
    """
    fmri = []
    for (i, j) in zip(fmri_filenames, fmri_motions):
        x = pd.read_csv(i, header=None).values
        y = np.loadtxt(j)
        fmri.append(clean(x, detrend=False, standardize=True, confounds=y))
    return fmri
    
def calc_dist(s1, s2, method, w=None):
    """
    calculate correlation/distance between time-series
    *TODO: should change dissimilarity to similarity (standard measure)
    
    PARAMETERS:
        s1, s2 - two time-series, should have same length
        method - choose from 'corr', 'dtw_dist' and 'dtw_path_len'
        w - warping window for DTW method, if method == 'corr', need not input
    RETURNS:
        The distance/correlation between s1 and s2
    """
    ### Traditional Correlation coefficient
    if method == 'corr':
        # np.correlate --> cross-correlation
        # np.corrcoef --> pearson correlation
        return np.corrcoef(s1, s2)[0,1]
    
    ### Dynamic Time Warping distance & Warping path length
    # A so called edit distance, which means that it measures the “cost” of transforming one time-series to the other one.
    if method.startswith('dtw'):
        
        n = len(s1)
        m = len(s2)
        assert n == m
        # allocate DTW matrix and fill it with large values
        DTW = np.ones([n + 1, m + 1]) * float("inf")
    
        # set warping window size
        if w == None:
            raise ValueError('No warping window length!')
        w = max(w, abs(n - m))
    
        DTW[0, 0] = 0
    
        # precalculate the squared difference between each time-series element pair
        cost = np.square(np.transpose(np.tile(s1, [n, 1])) - np.tile(s2, [m, 1]))
    
        # fill the DTW matrix
        for i in range(n):
            for j in range(max(0, i - w), min(m, i + w)):
                DTW[i + 1, j + 1] = cost[i, j] + np.min([DTW[i, j + 1], 
                                                         DTW[i + 1, j], 
                                                         DTW[i, j]])
    
        if method == 'dtw_dist' or method == 'dtw_all':
            # obtain Euclidean distance
            dtw_dist = np.sqrt(DTW[-1, -1])
        
        if method == 'dtw_path_len' or method == 'dtw_all':
            # obtain the optimal path
            path = [[n-1, m-1]]
            i = n-1
            j = m-1
            while i > 0 or j > 0:
                if i == 0: 
                    j -= 1
                elif j == 0: 
                    i -= 1
                else:
                    last = min(DTW[i, j], DTW[i, j+1], DTW[i+1, j])
                    if DTW[i, j] == last:
                        i -= 1
                        j -= 1
                    elif DTW[i+1, j] == last:
                        j -= 1
                    else:
                        i -= 1
                path.append([i, j])
            path_len = abs(len(path) - n)

        if method == 'dtw_dist':
            return dtw_dist
        if method == 'dtw_path_len':
            return path_len
        if method == 'dtw_all':
            return dtw_dist, path_len



def dist_connectome(time_series, method='corr', w=None):
    """
    calculate atlas distance list for the time-series matrix
    
    PARAMETERS:
        time_series - time-series matrix of each atlas, eg. 200*39 for fmri_msdl
        method - choose from 'corr', 'dtw_dist' and 'dtw_path_len'
        w - warping window for DTW method, if method == 'corr', need not input
    RETURNS:
        distances - distance list (length: eg. 39*38/2)
    """
    # reset NaNs and infs
    time_series = np.nan_to_num(time_series)
    distances = []
    path_lengths = []

    if method != 'dtw_all':
        # calculate the lower triangle of the connectivity matrix
        for i in range(1, time_series.shape[1]):
            for j in range(0, i):
                distances.append(calc_dist(time_series[:, i], time_series[:, j], method, w))
        return distances
    else:
        for i in range(1, time_series.shape[1]):
            for j in range(0, i):
                dist, path_len = calc_dist(time_series[:, i], time_series[:, j], method, w)
                distances.append(dist)
                path_lengths.append(path_len)
        return distances, path_lengths
    

def dist_connectomes_thread(time_series_list, method='corr', w=None):
    """
    auxiliary function for multithreading
    """
    if method != 'dtw_all':
        return [dist_connectome(ts, method, w) for ts in time_series_list]
    else:
        distances = []
        path_lengths = []
        count = 0
        for ts in time_series_list:
            count += 1
            print('ts_list:',count,'/',len(time_series_list))
            dist, path_len = dist_connectome(ts, method, w)
            distances.append(dist)
            path_lengths.append(path_len)
        print('out_dist',len(distances))
        print('out_path_len',len(path_lengths))
        return distances, path_lengths


def calculate_connectomes_thread(num_threads, time_series_list, method, w=None):
    """
    calculate atlas distance list for each individual via multithreading
    
    PARAMETERS:
        num_threads - number of threads
        time_series_list - list of time_series matrix, could with different time-series lengths, eg. 200*39,156*39,120*39,etc
        method - choose from 'corr', 'dtw_dist' and 'dtw_path_len'
        w - warping window for DTW method, if method == 'corr', need not input
    RETURNS:
        distance list calculated from multithreading (for 1127 individuals)
    """

    # calculate the index boundaries: which time_series_list elements should be processed by which threads
    boundaries = (np.ceil(len(time_series_list) / num_threads) * np.arange(0, num_threads + 1)).astype(np.int)
    boundaries[-1] = min(boundaries[-1], len(time_series_list))

    # create list of lists of time-series, so every thread can operate on its own list of time-series
    time_series_list_for_pool = [time_series_list[boundaries[i]:boundaries[i + 1]] for i in range(num_threads)]
    
    # create pool for parallel processing
    pool = Pool(processes=num_threads)
    
    # run DTW distance calculations parallel for the time-series lists of each thread
    if method != 'dtw_all':
        start = time.time()
        func = partial(dist_connectomes_thread, method = method, w = w)
        distances = (pool.map(func, time_series_list_for_pool))
        print("--- %s seconds ---" % (time.time() - start))
        return [item for sublist in distances for item in sublist]
    else:
        start = time.time()
        print('start time')
        func = partial(dist_connectomes_thread, method = method, w = w)
       	output = pool.map(func, time_series_list_for_pool)
        print("--- %s seconds ---" % (time.time() - start))
        
        # each thread
        distances = []
        path_lengths = []
        for sublist in output:
            # distance and path length
            assert len(sublist[0]) == len(sublist[1])
            for i in range(len(sublist[0])):
                distances.append(sublist[0][i])
                path_lengths.append(sublist[1][i])

        return distances, path_lengths


if __name__ == '__main__':
    data_train, labels_train = get_train_data()
    #fmri_features = [col for col in data_train.columns 
    #                 if col.startswith('fmri') 
    #                 and not col.endswith('select') 
    #                 and not col.endswith('motions')]
    fmri_features = ['fmri_basc122','fmri_basc197']
    for feature in fmri_features:
        print(feature)
        ts_extracted = _load_fmri_motion_correction(data_train[feature], data_train['fmri_motions'])
        #path_length_list = calculate_connectomes_thread(8, ts_extracted, 'dtw_path_len', w=10)
        distance_list, path_length_list = calculate_connectomes_thread(8, ts_extracted, 'dtw_all', w=10)
        with open('train_dtw_Dist_'+ feature +'.pkl', 'wb') as f1:
        	pickle.dump(np.array(distance_list), f1)
        with open('train_dtw_PathLen_'+ feature +'.pkl', 'wb') as f2:
        	pickle.dump(np.array(path_lengths_list), f2)

