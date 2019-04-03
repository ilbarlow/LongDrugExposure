#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:26:27 2019

@author: ibarlow
"""

""" Script to analyse the Long exposure worms that have been split into 5 
minute windows:
    1. Is there a change in response time over the 4 hour period?
    
    2. When is the optimum?
    
    3. Is it consistent across the two days of experiments?
    
    nb. these experiments were done before metadata files were created and so
    the drug info in the filename itself.
    
    """
    
import pandas as pd
import sys
import os
import glob
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy import stats
import itertools
from sklearn.decomposition import PCA

#add path of custom functions
sys.path.insert(0, '/Users/ibarlow/Documents/GitHub/pythonScripts/Functions')

import statsFeats as statsFeat

def find_window(filename):
    """ Little function to find the window number 
    Input - filename
    
    Output - the window as int
    """
    try:
        window = int(re.search(r"\d+", re.search(r"_window_\d+", filename).group()).group())
        return window+1 #add one as window indexing starts at 0
    except Exception as error:
        print ('{} has error:{}'.format(filename, error))
        return

if __name__ == '__main__':
    FoldIn = '/Volumes/behavgenom_archive$/Ida/MultiWormTracker/LongExposure'
    control = 'DMSO'
    threshold = 0.5
    
    save_dir = os.path.join(FoldIn, 'Figures5Mins')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    #import features, filenames and metadata
    feat_files = glob.glob(os.path.join(FoldIn, 'features_summary_tierpsy*'))
    filename_files = glob.glob(os.path.join(FoldIn, 'filenames_summary_tierpsy*'))
#    meta_files = glob.glob(os.path.join(FoldIn, '**/*metadata.csv'), recursive=True)
    
    FeatMat={'chunked': pd.DataFrame(), 'unchunked':pd.DataFrame()}
    FilenameMat={'chunked': pd.DataFrame(), 'unchunked':pd.DataFrame()}
    DrugInfo = {'chunked': pd.DataFrame(), 'unchunked':pd.DataFrame()}
    for f in feat_files:
        _window = find_window(f)
        if _window:
            _FeatFrame = pd.read_csv(f, index_col =False)
            _FeatFrame['window'] = _window
            FeatMat['chunked']= FeatMat['chunked'].append(_FeatFrame,\
                   sort=True).reset_index(drop=True)
        else:
            FeatMat['unchunked']= pd.read_csv(f, index_col=False)
            FeatMat['unchunked']['window'] =-1 #set as negative for grouping purposes
    
    metadata = pd.DataFrame()
    for fn in filename_files:
        _fileinfo = pd.read_csv(fn, index_col=False)
        _window = find_window(fn)
        for i,r in _fileinfo.iterrows():
            r = r.append(pd.Series({'drug': r.file_name.split('/')[-1].split('_')[0],\
                                    'concentration': r.file_name.split('/')[-1].split('_')[1],\
                                    'window': _window,\
                                    'date': r.file_name.split('_')[-3]}))
            metadata = metadata.append(r, ignore_index=True)
        
    #match up fileid and window
    _metaIndex = [metadata['file_id'], metadata['window']]
    _metaApply = pd.MultiIndex.from_arrays(_metaIndex, names = ('file_id',\
                                                                'window'))
    metadata.index = _metaApply

    _featIndex = [FeatMat['chunked']['file_id'], FeatMat['chunked']['window']]
    _featApply = pd.MultiIndex.from_arrays(_featIndex, names = ('file_id',\
                                                                 'window'))
    FeatMat['chunked'].index = _featApply
    
    FeatMatFinal = pd.concat([FeatMat['chunked'], metadata], join= 'inner', axis=1, sort=True)
    FeatMatFinal = FeatMatFinal.drop(columns = ['window', 'file_id'])
    
    #filter out features with too many nans and bad files
    BadFiles = np.where(FeatMatFinal.isna().sum(axis=1)>FeatMatFinal.shape[1]*threshold)[0]
    BadFiles = np.unique(np.append(BadFiles, np.where(FeatMatFinal['is_good']==0)[0]))
    FeatMatFinal = FeatMatFinal.drop(columns=FeatMatFinal.columns[FeatMatFinal.isna().sum(axis=0)>FeatMatFinal.shape[0]*threshold])
    FeatMatFinal = FeatMatFinal.drop(columns = FeatMatFinal.select_dtypes(include='float').columns[FeatMatFinal.select_dtypes(include='float').std(axis=0)==0])
    FeatMatFinal = FeatMatFinal.reset_index(drop=False).drop(index=BadFiles)

    FeatMatFinal = FeatMatFinal.drop(columns = ['file_id', 'file_name'])
    
    metadata_dict = {}
    metadata_dict['drug'] = np.unique(FeatMatFinal['drug'])
    metadata_dict['date'] = np.unique(FeatMatFinal['date'])
    metadata_dict['window'] = np.unique(FeatMatFinal['window'])
    
    #plot euclidean distance as a function of time
    FeatMatFinal = FeatMatFinal.fillna(FeatMatFinal.mean(axis=0))
    FeatMatZ = pd.DataFrame(data = stats.zscore(FeatMatFinal.select_dtypes(include='float'), \
                            ddof=1,\
                            axis=0), columns = FeatMatFinal.select_dtypes(include='float').columns)
    FeatMatZ = FeatMatZ.fillna(FeatMatZ.mean(axis=0))
    FeatMatZ[['drug', 'window' , 'date']] = FeatMatFinal[['drug', 'window', 'date']]
    
    groupedFeatMatZ = FeatMatZ.groupby(['drug', 'date', 'window']) 
    FeatMatZ_iterator = list(itertools.product(*metadata_dict.values()))
    EucDist= pd.DataFrame()
    for i in FeatMatZ_iterator:
        control_group = tuple(s if type(s)!=str or s.startswith('18') else control for s in i)
        try:
            EucDist = EucDist.append(pd.Series({'eDist': euclidean_distances(np.array([groupedFeatMatZ.get_group(i).select_dtypes(include='float').mean(axis=0).values,\
                                groupedFeatMatZ.get_group(control_group).select_dtypes(include= 'float').mean(axis=0).values]))[0,1],\
                                                'metadata': i}),\
                                                ignore_index=True)
        except KeyError: 
            continue
            #print('no data for {} worms on {} drug on {}'.format(i[0], i[1], i[2]))
            
    EucDist[['drug', 'date', 'window']] = pd.DataFrame(EucDist.metadata.tolist(), index=EucDist.index)
    
    #make a figure
    sns.lineplot(x = 'window', y='eDist', hue = 'drug', data = EucDist)
    
    #PCA of all the drugs at 15 minute interval snapshots -
    pca = PCA()
    X = pca.fit_transform(FeatMatZ.select_dtypes(include='float').values)  
    cut_off = sum(np.cumsum(pca.explained_variance_ratio_)<0.95)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.plot(cut_off*np.ones([2,]), [0,1])
    plt.savefig(os.path.join(save_dir, 'cumvar.tif'))
    
    #components that explain the variance
    #make a dataframe ranking the features for each PC and also include the explained variance
        #in separate dataframe called PC_sum
    PC_feat = [] #features
    PC_sum =[] #explained variance
    for PC in range(0, cut_off):
        _sortPCs = np.flip(np.argsort(pca.components_[PC]**2), axis=0)
        PC_feat.append(list(FeatMatZ.select_dtypes(include='float').columns[_sortPCs]))
        _weights = (pca.components_[PC]**2)/np.sum(pca.components_[PC]**2)
        PC_sum.append(list(_weights))
    
    #dataframe containing standards scores of contibution of each feature
        #rows correspond to PCs
    PC_vals = pd.DataFrame(data= PC_sum, columns = FeatMatZ.select_dtypes(include='float').columns)
    
    #okay so now can plot as biplot
    plt.figure()
    plt.arrow(0,
              0,
              PC_vals[PC_feat[0][0]][0]*100, \
              PC_vals[PC_feat[0][0]][1]*100,\
              color= 'b')
    plt.arrow(0,\
              0, \
              PC_vals[PC_feat[1][0]][0]*100,\
              PC_vals[PC_feat[1][0]][1]*100, \
              color='r')
    plt.text(PC_vals[PC_feat[0][0]][0] + 0.7,\
             PC_vals[PC_feat[0][0]][1] - 0.3,\
             PC_feat[0][0],\
             ha='center', \
             va='center')
    plt.text(PC_vals[PC_feat[1][0]][0]+0.5, \
             PC_vals[PC_feat[1][0]][1]+1,\
             PC_feat[1][0],\
             ha='center',\
             va='center')

    plt.xlim (-1, 1)
    plt.ylim (-1, 1)
    plt.xlabel('%' + 'PC_1 (%.2f)' % (pca.explained_variance_ratio_[0]*100), fontsize = 16)
    plt.ylabel('%' + 'PC_2 (%.2f)' % (pca.explained_variance_ratio_[1]*100), fontsize = 16)
    plt.show()
    
    #add on the metadata
    PC_df = pd.DataFrame(X[:,:cut_off], columns = ['PC_{}'.format(i) for i in range (1,cut_off+1)])
    PC_df = pd.concat([PC_df, FeatMatFinal[['drug','date', 'window']]], axis=1)
    
#    PC_df[PC_df['drug']==metadata_dict['drug'][0]]['PC_1']
    import PC_traj as PCJ

    PC_df_grouped = PC_df.groupby(['drug', 'date', 'window'])
    PC_av = pd.DataFrame()
    PC_std = pd.DataFrame()
    for i in FeatMatZ_iterator:
        try:
            PC_av = PC_av.append(PC_df_grouped.get_group((i)).mean(), 
                                 ignore_index=True
                                 )
            PC_std = PC_std.append(PC_df_grouped.get_group(i).std(), 
                                   ignore_index=True
                                   )
        except Exception as error:
#            print(error)
            continue
