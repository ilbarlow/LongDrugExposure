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
    the drug info is in the filename itself.
    
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

#import statsFeats as statsFeat
from PCAexplainedVariance import explainVariance

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
    FoldIn = '/Volumes/behavgenom_archive$/Ida/MultiWormTracker/LongExposure'#/5minWindows'
    control = 'DMSO'
    threshold = 0.5 #for bad features
    
    save_dir = os.path.join(FoldIn, 'Figures5Mins')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    #import features, filenames and metadata
    feat_files = glob.glob(os.path.join(FoldIn, '**/features_summary_tierpsy*'), recursive=True)
    filename_files = glob.glob(os.path.join(FoldIn, '**/filenames_summary_tierpsy*'), recursive=True)
#    meta_files = glob.glob(os.path.join(FoldIn, '**/*metadata.csv'), recursive=True)
    
    #import the data
    FeatMat={'chunked': pd.DataFrame(), 'unchunked':pd.DataFrame()}
    FilenameMat={'chunked': pd.DataFrame(), 'unchunked':pd.DataFrame()}
    DrugInfo = {'chunked': pd.DataFrame(), 'unchunked':pd.DataFrame()}
    for f in feat_files:
        _window = find_window(f)
        if _window:
            _FeatFrame = pd.read_csv(f, index_col =False)
            _FeatFrame['window'] = _window
            FeatMat['chunked']= FeatMat['chunked'].append(_FeatFrame, 
                   sort=True).reset_index(drop=True)
        else:
            FeatMat['unchunked']= pd.read_csv(f, index_col=False)
            FeatMat['unchunked']['window'] =-1 #set as negative for grouping purposes
    
    #extract metadata from filenames
    metadata = {'chunked':pd.DataFrame(), 'unchunked':pd.DataFrame()}
    for fn in filename_files:
        _window = find_window(fn)
        _fileinfo = pd.read_csv(fn, index_col=False) #read in the filenames
        if _window:
            for i,r in _fileinfo.iterrows():
                metadata['chunked'] = metadata['chunked'].append(pd.Series({
                                        'drug': r.file_name.split('/')[-1].split('_')[0],
                                        'concentration': r.file_name.split('/')[-1].split('_')[1],
                                        'window': _window,
                                        'date': r.file_name.split('_')[-3],
                                        'file_id': r.file_id,
                                        'is_good':r.is_good
                                        }),
                                        ignore_index=True)
        else:
            files = list(_fileinfo.file_name)
            metadata['unchunked'] = pd.DataFrame({
                                        'drug': [i.split('/')[-1].split('_')[0] for i in files],
                                        'concentration': [i.split('/')[-1].split('_')[1] for i in files],
                                        'window': -1,
                                        'date': [i.split('_')[-3] for i in files],
                                        'file_id': list(_fileinfo.file_id),
                                        'is_good': list(_fileinfo.is_good)
                                        })
        
    #match up fileid and window to be able to concatenate the FeatMat and metaMat
    for c in FeatMat:
        metadata[c]['indexMatch'] = list(zip(metadata[c]['window'].astype(int),metadata[c]['file_id'].astype(int)))
        FeatMat[c]['indexMatch'] = list(zip(FeatMat[c]['window'], FeatMat[c]['file_id']))
        
        metadata[c].drop(columns = ['window', 'file_id'], inplace=True)
    
    FeatMatConcat = pd.concat(FeatMat, sort=False).reset_index(drop=True)
    metadataConcat = pd.concat(metadata, sort=False).reset_index(drop=True)
    
    FeatMatFinal = pd.concat([FeatMatConcat.set_index('indexMatch'),
                             metadataConcat.set_index('indexMatch')],
                             axis=1,
                             join = 'inner',
                             )
    FeatMatFinal.reset_index(drop = True, inplace=True)
    
    #filter out features with too many nans, bad files, and features with standard deviation=0
    BadFiles = np.where(FeatMatFinal.isna().sum(axis=1)>FeatMatFinal.shape[1]*threshold)[0]
    BadFiles = np.unique(np.append(BadFiles, np.where(FeatMatFinal['is_good']==0)[0]))
    FeatMatFinal = FeatMatFinal.reset_index(drop=True).drop(index=BadFiles)
    FeatMatFinal = FeatMatFinal.drop(columns=FeatMatFinal.columns[FeatMatFinal.isna().sum(axis=0)>FeatMatFinal.shape[0]*threshold])
    FeatMatFinal = FeatMatFinal.drop(columns = FeatMatFinal.select_dtypes(include='float').columns[FeatMatFinal.select_dtypes(include='float').std(axis=0)==0])
    FeatMatFinal.drop(columns = 'file_id', inplace=True)
    
    #extract out list of the feature names
    feat_names = list(FeatMatFinal.select_dtypes(include='float').columns)
    
    #make a dictionary for sorting the data by unique drugs, times, and windows
    metadata_dict = {}
    metadata_dict['drug'] = np.unique(FeatMatFinal['drug'])
    metadata_dict['date'] = np.unique(FeatMatFinal['date'])
    metadata_dict['window'] = np.unique(FeatMatFinal['window'])
    
    #normalize
    FeatMatFinal = FeatMatFinal.fillna(FeatMatFinal.mean(axis=0))
    FeatMatZ = pd.DataFrame(data = stats.zscore(FeatMatFinal.select_dtypes(include='float'), 
                            ddof=1, #n-1 degrees of freedom for calculating stdev
                            axis=0),
                            columns = feat_names)
    FeatMatZ[['drug', 'window' , 'date']] = FeatMatFinal[['drug', 'window', 'date']]
    
    #plot euclidean distance as a function of time
    groupedFeatMatZ = FeatMatZ.groupby(['drug', 'date', 'window']) 
    FeatMatZ_iterator = list(itertools.product(*metadata_dict.values()))
    EucDist= pd.DataFrame()
    for i in FeatMatZ_iterator:
        control_group = tuple(s if type(s)!=str or s.startswith('18') else control for s in i)
        try:
            EucDist = EucDist.append(
                    pd.Series({'eDist': euclidean_distances(np.array([groupedFeatMatZ.get_group(i).select_dtypes(include='float').mean(axis=0).values,
                                groupedFeatMatZ.get_group(control_group).select_dtypes(include= 'float').mean(axis=0).values]))[0,1],
                                                'metadata': i}),
                                                ignore_index=True)
        except KeyError: 
            print('no data from {}, {} window on {} drug'.format(i[0], i[1], i[2]))            
            continue
    EucDist[['drug', 'date', 'window']] = pd.DataFrame(EucDist.metadata.tolist(), index=EucDist.index)
    
    #make a figure
    sns.lineplot(x = 'window', y='eDist', hue = 'drug', data = EucDist)
    plt.savefig(os.path.join(save_dir, 'euclideanDistTS.png'))
    
    #PCA of all the drugs at 5 minute interval snapshots -
    pca = PCA()
    X = pca.fit_transform(FeatMatZ.select_dtypes(include='float').values)  
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    cut_off = sum(cumvar<0.95)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.plot(cut_off*np.ones([2,]), [0,1])
    plt.savefig(os.path.join(save_dir, 'cumvar.tif'))
        
    PC_df, PC_sum, PC_feat = explainVariance(pca, X, cut_off, feat_names, save_dir, (0,1)) 
    PC_df = pd.concat([PC_df, FeatMatZ[['drug','date', 'window']]], axis=1)    
    
    import PC_traj as PCJ
    from matplotlib.colors import LinearSegmentedColormap
    
    PC_df_grouped = PC_df.groupby(['drug', 'date', 'window'])
    PC_av = pd.DataFrame()
    PC_std = pd.DataFrame()
    for i in FeatMatZ_iterator:
        try:
            PC_av = PC_av.append(PC_df_grouped.get_group(i).mean().append(
                                    pd.Series({'drug':i[0],
#                                               'date':i[1],
#                                               'window':i[2]
                                                })).to_frame().transpose(),
                                    ignore_index=True)
    
            PC_std = PC_std.append(PC_df_grouped.get_group(i).std().append(
                                    pd.Series({'drug':i[0],
#                                               'date':i[1],
#                                               'window':i[2]
                                                })).to_frame().transpose(), 
                                   ignore_index=True)
        except Exception as error:
            print(error)
            continue
    
    #make some figures
    cmap1 = sns.color_palette('tab20',len(metadata_dict['drug']))
    cmapGraded = [] #and graded colormaps
    for item in cmap1:
        cmapGraded.append([(1,1,1), (item)])
    
    lutGraded = dict(zip(metadata_dict['drug'], cmapGraded))
    cm={}
    for drug in lutGraded:
        cmap_name = drug
        # Create the colormap
        cm[drug] = LinearSegmentedColormap.from_list(
            cmap_name, lutGraded[drug], N=60)
        plt.register_cmap(cmap = cm[drug])   
    
    PCJ.PC_trajGraded(PC_av,
                      PC_std,
                      ['PC_1', 'PC_2'],
                      metadata_dict['date'][0],
                      save_dir,
                      '.png',
                      scaling = 'window',
                      start_end = False,
                      cum_var = cumvar,
                      legend = 'off')
