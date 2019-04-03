#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 13:22:56 2018

@author: ibarlow
"""

""" Analyse the second of the combined long drug exposure experiments together

    -- eventually will combine the two"""
    
%cd Documents/GitHub/pythonScripts/Functions

import displayfiles as disp
import pandas as pd
import os
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


saveDir, csvFiles = disp.displayfiles('.csv', inputfolder = None, outputfile= 'ChunkedFiles.tsv')

#load each of these csv files
FeatMatAll = pd.DataFrame()
for item in csvFiles:
    temp = pd.read_csv(item)
    #add on the metadata
    temp['drug'] = os.path.basename(item).split('_')[0]
    temp['date'] = os.path.basename(item).split('_')[-3]
    temp['chunkTime']=item.split('/')[-3]
    temp = temp.rename(columns = {'Unnamed: 0': 'chunk'})

    FeatMatAll = FeatMatAll.append(temp, sort=True)

    del temp
 
FeatMatAll = FeatMatAll.reset_index(drop=True)

#get information about all drugs and dates, and make a dictionary for the chunksize and chunk numbers
allDrugs= np.unique(FeatMatAll['drug'])
allDates = np.unique(FeatMatAll['date'])
chunkSize = np.unique(FeatMatAll['chunkTime'])
chunkDeets = {}
for item in chunkSize:
    chunkDeets[item] = list(FeatMatAll[FeatMatAll['chunkTime']==item]['chunk'].unique())

#filter the features with too many nans
FeatMatAll =FeatMatAll.drop(columns = FeatMatAll.columns[FeatMatAll.isna().sum()>FeatMatAll.shape[0]/2])

#filtering of data to remove outliers - for every drug at each time point 
# remove data points that are more than 2 sd away from the average for that drug and time

FeatMatAll2 = pd.DataFrame()
for item in chunkDeets:
    foo = FeatMatAll['chunkTime']==item
    for drug in allDrugs:
        #average for the drug
        bar = FeatMatAll['drug'] == drug
        temp = FeatMatAll[foo&bar]
        deets = temp[['date', 'exp', 'chunk']]
        temp = temp.select_dtypes(include = ['float', 'int'])
        for chunk in chunkDeets[item]:
            botmargin = temp.mean() - 2*temp.std()
            topmargin = temp.mean() + 2*temp.std()
            botmargin = botmargin.drop ('chunk')
            topmargin = topmargin.drop('chunk')
            temp2 = temp[temp['chunk']==chunk].select_dtypes(include = 'float')
            deets2 = deets[deets['chunk']==chunk]
            
            foo2 = temp2>botmargin
            bar2 = temp2<topmargin
            
            #replace with nan and then fill with mean values for PCA
            temp3 = temp2[foo2&bar2].fillna(temp.mean(), axis=0)
            
            temp3['chunkTime'] = item
            temp3['drug'] =drug
            temp3['chunk']= chunk
            temp3['date'] = deets2['date']
            temp3['exp'] = deets2['exp']
            
            FeatMatAll2 = FeatMatAll2.append(temp3)
            
            del botmargin, topmargin, temp2, temp3, foo2, bar2, deets2
            
        del bar, temp, deets
    del foo

FeatMatAll2 = FeatMatAll2.reset_index(drop=True)

#some of the V4 experiments are highly variable so going to drop
foo = FeatMatAll2['exp'] != 'V4_10_0_Set3_Pos4_Ch5_180712_185642'
bar= FeatMatAll2['exp'] != 'V4_10_0_Set2_Pos5_Ch5_180712_141924'

FeatMatAll3 = FeatMatAll2[foo & bar] #or FeatMatAll2[FeatMatAll2['exp']!=V4_2]
FeatMatAll3 = FeatMatAll3.reset_index(drop=True)

FeatMatAll3.to_csv(os.path.join(saveDir, 'FeatMatAllCombined.csv'))

#now Z-score
featZ = FeatMatAll3.select_dtypes(include=['float64'])
featZ = pd.DataFrame(stats.zscore(featZ, axis=0), columns = featZ.columns)

#drop the features with too many nans
featZ = featZ.drop(columns = featZ.columns[featZ.isna().sum()>featZ.shape[0]/2])
featZ = pd.concat([featZ, FeatMatAll3[['drug', 'date', 'chunk', 'chunkTime', 'exp']]], axis=1)
            
FeatZinfo = featZ[['drug', 'chunk', 'date', 'chunkTime', 'exp']]

#plot as clustermap
    #set lut to chunk number
cmap = sns.color_palette('tab20', allDrugs.shape[0])
lut = dict(zip(allDrugs, cmap))
 
rowColors = featZ['drug'].map(lut)#map onto the feature Matrix

cg = sns.clustermap(featZ.select_dtypes(include='float64'),\
                    metric  = 'euclidean', cmap = 'inferno', row_colors = rowColors,vmin= -5, vmax= 5)
plt.setp(cg.ax_heatmap.yaxis.set_ticklabels\
     (featZ['drug'][cg.dendrogram_row.reordered_ind]))
#plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize = 8)
#plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation = 90, fontsize = 10) 
col = cg.ax_col_dendrogram.get_position()
cg.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*1, col.height*1])

#and make lut map
#make a figure of the colorbar
colors = [(v) for k,v in lut.items()]
    #plot separately
plt.figure()
ax = plt.imshow([colors])
ax.axes.set_xticklabels(range(0,13,1))
ax.axes.set_xticklabels(lut.keys(), rotation = 90)
ax.axes.set_xticks(np.arange(0,13,1))
ax.axes.xaxis.set_ticks_position('top')
plt.savefig(os.path.join(os.path.dirname(saveDir), 'Figures', 'drugColors.png'))

#%% move on to PCA
from sklearn.decomposition import PCA
import PCA_analysis as PC_custom 

#make array of z-scored data
X = np.array(featZ.select_dtypes(include='float64'))

#initialise PCA
pca = PCA()
X2= pca.fit_transform(X)
cumvar = np.cumsum(pca.explained_variance_ratio_)
thresh = cumvar <= 0.95 #set 95% variance threshold
cut_off = int(np.argwhere(thresh)[-1])
#make a plot
sns.set_style('whitegrid')
plt.figure()
plt.plot(range(0, len(cumvar)), cumvar*100)
plt.plot([cut_off, cut_off], [0, 100], 'k')
plt.text(cut_off, 100, cut_off)
plt.xlabel('Number of Principal Components', fontsize =16)
plt.ylabel('variance explained', fontsize =16)
plt.show()

#now put the 1:cut_off PCs into a dataframe
PCname = ['PC_%d' %(p+1) for p in range (0,cut_off+1)]
PC_df = pd.DataFrame(data= X2[:,:cut_off+1], columns = PCname)
PC_df['drug'] = featZ['drug']
PC_df['chunk'] = featZ['chunk']
PC_df['chunkTime'] = featZ['chunkTime']
PC_df['exp']= featZ['exp']

#make graded colormap
cmapGraded = []
for item in cmap:
    cmapGraded.append([(1,1,1), (item)])
    
lutGraded = dict(zip(allDrugs, cmapGraded))
cm={}
for drug in lutGraded:
    cmap_name = drug
    # Create the colormap
    cm[drug] = LinearSegmentedColormap.from_list(
        cmap_name, lutGraded[drug], N=60)
    plt.register_cmap(cmap = cm[drug])
    #plt.register_cmap(name=drug, data=LinearSegmentedColormap.from_list())  # optional lut kwarg

#have a look at the colors
import make_colormaps as mkc
mkc.plot_color_gradients(cmap_list=cm, drug_names = lutGraded.keys())
plt.savefig(os.path.join(os.path.dirname(saveDir), 'Figures', 'GradeddrugColors.png'))


#make the PC plots
for chunks in chunkSize:    
    PC_custom.PC12_plots(PC_df[PC_df['chunkTime']==chunks], [], chunks ,  cmap, saveDir,'tif', 'chunk')

PCmean = {}
PCsem = {}
for chunks in chunkSize:
    PCmean[chunks], PCsem[chunks] = PC_custom.PC_av(PC_df[PC_df['chunkTime'] == chunks], [], 'chunk')

#make the plots
for chunks in PCmean:
    plt.figure()
    xscale = 1/(PCmean[chunks].max()['PC_1'] - PCmean[chunks].min()['PC_1'])
    yscale = 1/(PCmean[chunks].max()['PC_1'] - PCmean[chunks].min()['PC_2'])
    cscale = np.arange(1, np.unique(PCmean[chunks]['chunk']).shape[0]+1,1)
    
    for drug in selDrugs:
        plt.errorbar(x= PCmean[chunks][PCmean[chunks]['drug']==drug]['PC_1']*xscale,\
                     y = PCmean[chunks][PCmean[chunks]['drug']==drug]['PC_2']*yscale,\
                     xerr = PCsem[chunks][PCsem[chunks]['drug']==drug]['PC_1']*xscale,\
                     yerr = PCsem[chunks][PCsem[chunks]['drug']==drug]['PC_2']*yscale, \
                     color = [0.9, 0.9, 0.9], zorder = -1, label = None)
        plt.pause(0.1)
        plt.scatter(x = PCmean[chunks][PCmean[chunks]['drug']==drug]['PC_1']*xscale,\
                    y=PCmean[chunks][PCmean[chunks]['drug']==drug]['PC_2']*yscale,\
                    cmap = plt.get_cmap(drug),c=cscale , vmin = 0, label = drug)
    plt.axis('scaled')
    plt.xlim (-1,1)
    plt.ylim (-1,1)
    plt.title(chunks)
    plt.xlabel('PC_1' + str(cumvar[0]*100) + '%')
    plt.ylabel('PC_2' + str(cumvar[1]*100) + '%')
    plt.legend(loc= 'best')
    plt.savefig(os.path.join(saveDir, 'PC12_errorbar.png'),dpi = 200)

#make a density plot
plt.figure()
selDrugs = ['V9', 'Haloperidol','V5', 'DMSO', 'Clozapine']
for drug in selDrugs:
    ax = sns.kdeplot(PC_df[PC_df['chunkTime']=='15minChunkedTrajectories'][PC_df['drug'] == drug]['PC_1'],\
                     PC_df[PC_df['chunkTime']=='15minChunkedTrajectories'][PC_df['drug'] == drug]['PC_2'],
                     shade = True, shade_lowest = False, cmap = plt.get_cmap(drug), alpha = 0.8, label = drug)
#plt.legend()
plt.show()
plt.axis('equal')

#which features contribute to the variance?
#components that explain the variance
    #make a dataframe ranking the features for each PC and also include the explained variance (z-normalised)
        #in separate dataframe called PC_sum
PC_feat = [] #features
PC_sum =[] #explained variance
for PC in range(0, len(PCname)):
    PC_feat.append(list(featZ.select_dtypes(include = 'float64').columns[np.argsort(pca.components_[PC])]))
    PC_sum.append(list((pca.components_[PC])/ np.sum(abs(pca.components_[PC]))))

#dataframe containing standards scores of contibution of each feature
        #rows correspond to PCs
PC_vals = pd.DataFrame(data= PC_sum, columns = featZ.select_dtypes(include = 'float64').columns)

#okay so now can plot as biplot
plt.figure()
for i in range(0,1):
    plt.arrow(0,0, PC_vals.iloc[0,:][PC_feat[0][-1-i]]*100, \
              PC_vals.iloc[1,:][PC_feat[0][-1-i]]*100,color= 'b')
    plt.arrow(0,0, PC_vals.iloc[0,:][PC_feat[1][-1-i]]*100,\
              PC_vals.iloc[1,:][PC_feat[1][-1-i]]*100, color='r')
    plt.text(PC_vals.iloc[0,:][PC_feat[0][-1-i]] + 0.7,\
             PC_vals.iloc[1,:][PC_feat[0][-1-i]] - 0.3, PC_feat[0][-1-i],\
             ha='center', va='center')
    plt.text(PC_vals.iloc[0,:][PC_feat[1][-1-i]]+0.5, PC_vals.iloc[1,:][PC_feat[1][-1-i]]+1,\
         PC_feat[1][-1-i], ha='center', va='center')

plt.xlim (-5, 5)
plt.ylim (-5, 5)
plt.xlabel('%' + 'PC_1 (%.2f)' % (pca.explained_variance_ratio_[0]*100), fontsize = 16)
plt.ylabel('%' + 'PC_2 (%.2f)' % (pca.explained_variance_ratio_[1]*100), fontsize = 16)
plt.show()
#plt.savefig(os.path.join(os.path.dirname(directoryA), 'Figures', 'agar_biplot.png'))
    

#can also just plot the PCs across the time chunks
for chunk in chunkSize:
    plt.figure()
    for PC in range(0,2):
        plt.subplot(1,2,PC+1)
        ax =sns.pointplot(x="chunk", y=PC_df.columns[PC], data=PC_df[PC_df['chunkTime']==chunk], \
                          palette = cmap, hue = 'drug', legend = True)
        ax.legend_.remove()
        ax.axes.set_xticklabels(labels = [], rotation = 45)
        #plt.ylim([-1,1])
    plt.show()
    
    plt.legend(loc=6, bbox_to_anchor=(1, 0.5) ,ncol = 1, frameon= True)
    plt.tight_layout(rect=[0,0,1,1])
    plt.savefig(os.path.join(os.path.dirname(saveDir), 'Figures',chunk+ '_PC1PC2plts.png'))
    

#%% Calculate euclidean distance between 
    #and plot the distance between the DMSO and No_compound controls and each drug over the time chunks

from scipy.spatial import distance
import re

dist1 = pd.DataFrame(distance.squareform(distance.pdist(featZ.select_dtypes(include = 'float'), \
                                                        metric= 'euclidean')))
dist1 = pd.concat([dist1, FeatZinfo], axis=1)

#compare to DMSO
DMSOres = dist1.iloc[:,dist1[dist1['drug']=='DMSO'].index]
#DMSOres.columns = 
DMSOres = pd.concat([DMSOres, FeatZinfo], axis = 1)

#need to remodel so that only comparing drugs at the same timepoint
DMSOmat = {}
DMSOmat0 = {}
for chunk in chunkDeets:
    DMSOmat[chunk] = pd.DataFrame()
    DMSOmat0[chunk] = pd.DataFrame()
    chunkTemp = DMSOres[DMSOres['chunkTime']==chunk]
    allChunks = np.unique(chunkTemp['chunk'])
    for drug in allDrugs:
        foo = chunkTemp['drug'] == drug
        foo2 = chunkTemp['drug'] == 'DMSO'
        for step in allChunks:
            bar = chunkTemp['chunk'] == step
            bar2 = chunkTemp['chunk']==0
            temp = chunkTemp[foo&bar]
            temp2 = temp.loc[:,chunkTemp[foo2&bar].index]
            temp3 = pd.DataFrame(temp2.mean(axis=1))
            temp4 = temp.loc[:,chunkTemp[foo2&bar2].index]
            temp5 = pd.DataFrame(temp4.mean(axis=1))
            temp3['drug'] = drug
            temp3['chunk'] = step
            temp3['rep'] = np.arange(1,temp3.shape[0]+1)
            
            temp5['drug'] = drug
            temp5['chunk'] = step
            temp5['rep'] = np.arange(1,temp3.shape[0]+1)
            DMSOmat[chunk] = DMSOmat[chunk].append(temp3)
            DMSOmat0[chunk] = DMSOmat0[chunk].append(temp5)
            
            del temp, temp2, temp3, temp4, temp5,  bar
    del foo, foo2
del bar2

def errBar (pDistDF, figName, savedir, cmap, csize):
    """Function for plotting the drug response using shaded error bar
    Input:
        pDistDF - dataframe containing the euclidean distances between drug and control
        chunkRange - a dictionary for the x ticks and tick labels
        saveDir - directory to save figures into
        cmap - colormap to use
        csize- chunksize
        
    Output:
        Figure"""
    plt.figure()
    ax = sns.lineplot(data = pDistDF, x= 'chunk', y = 'pdist', \
                hue = 'drug',palette =cmap)
    
    ax.axes.set_xlim([0,pDistDF['chunk'].max()])
    ax.axes.set_xticks(np.arange(1,pDistDF['chunk'].max(),2))
    ax.axes.set_xticklabels(np.arange(5,pDistDF['chunk'].max()*csize,csize*2), rotation = 45)
    plt.ylim ([5, 20])
    plt.ylabel('Euclidean distance')
    plt.xlabel('Chunk')
    plt.legend(loc=0, bbox_to_anchor=(1, 1) ,ncol = 1, frameon= True)
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.savefig(os.path.join(os.path.dirname(savedir), 'Figures', figName))
    plt.show() 
    
#reset index
for chunk in DMSOmat:
    chsize = int(re.findall(r'\d+', chunk)[0])
    DMSOmat[chunk] = DMSOmat[chunk].reset_index(drop=True)
    DMSOmat[chunk] = DMSOmat[chunk].rename(columns = {0:'pdist'})
    
    DMSOmat0[chunk] = DMSOmat0[chunk].reset_index(drop=True)
    DMSOmat0[chunk] = DMSOmat0[chunk].rename(columns = {0:'pdist'})

    errBar(DMSOmat[chunk], chunk + '_pDistAbs.png', saveDir, cmap, chsize)
    errBar(DMSOmat0[chunk], chunk + '_pDistNormto0.png', saveDir, cmap, chsize)

selDrugs = ['V9', 'DMSO', 'Haloperidol', 'V5', 'V7']
plt.figure()
for drug in selDrugs:
    ax = sns.lineplot(data=DMSOmat[chunk][DMSOmat[chunk]['drug']==drug], x= 'chunk', y ='pdist',\
                 color = lut[drug], label=drug)
csize = int(re.findall(r'\d+', chunk)[0])
ax.axes.set_xlim([0,DMSOmat[chunk]['chunk'].max()])
ax.axes.set_xticks(np.arange(1,DMSOmat[chunk]['chunk'].max(),2))
ax.axes.set_xticklabels(np.arange(5,DMSOmat[chunk]['chunk'].max()*csize,csize*2), rotation = 45)
plt.ylim ([5,25])
plt.xlabel('time (mins)')
plt.ylabel('euclidean distance')
plt.legend(loc=0, bbox_to_anchor=(1, 1) ,ncol = 1, frameon= True)
plt.tight_layout(rect=[0,0,0.75,1])
plt.show()
plt.savefig(os.path.join(os.path.dirname(saveDir), 'Figures', '_'.join(selDrugs) + 'pDistDMSO.png'), dpi=200)
