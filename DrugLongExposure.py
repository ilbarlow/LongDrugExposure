#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:22:45 2018

@author: ibarlow
"""

""" Analysis of long-term drug exposure"""

import TierPsyInput as TP
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import the features and trajectories
directoryA, fileDirA, featuresA, trajectoriesA = TP.TierPsyInput('new', 'none')

#list the files in
reps = list(trajectoriesA.keys())
exps = list(trajectoriesA[reps[0]].keys())
conds=[]
drugs =[]
for line in exps:
    conds.append(line.split('_')[0] +'_'+ line.split('_')[-2])
    drugs.append(line.split('_')[0])

allDrugs = np.unique(drugs)

del featuresA

#now cut the trajectories data up into 30 min chunks, fps = 25
chunk = 45000
nChunks = 10
chunkSize = {}
for i in range(0,nChunks+1):
    chunkSize[i] = (chunk*(i+1)) - (chunk-1)
    #chunkSize[i]+=(chunk*i)
del i
    
#make new dataframe containing the data for the half-hour windows
delT = 5*25 #5 second sliding window
trajectories2 = {}
for rep in trajectoriesA:
    for chunkT in range(1,len(chunkSize)-1):
        trajectories2[chunkT]={}
        for i in range(0,len(conds)):
            foo = trajectoriesA[rep][exps[i]]['timestamp']<chunkSize[chunkT]
            bar = trajectoriesA[rep][exps[i]]['timestamp']>=chunkSize[chunkT-1]
            trajectories2[chunkT][conds[i]]=trajectoriesA[rep][exps[i]][foo & bar]
            trajectories2[chunkT][conds[i]]['ttBin']=np.ceil(trajectoriesA[rep][exps[i]]['timestamp']/delT) #create binning window
            del foo,bar

del trajectoriesA
#%%
#now lets plot to see how the speed changes

def tsplotDF(ax, data, time, varToplot):
    """ Custom function for shaded error bars
    ax : axes generated
    data : pandas dataframe input
    time : timebins
    varToplot: variable to plot
    """
    #data.name = data
    x = np.unique(data[time])
    est = np.empty(shape = x.shape[0])
    sem =np.empty(shape = x.shape[0])
    for tbin in range(len(x)):
        est[int(tbin)] = data[varToplot][data[time]==x[tbin]].mean()
        sem[int(tbin)] = data[varToplot][data[time]==x[tbin]].std()
    cis = (est - sem, est + sem)
    ax.fill_between(x,cis[0],cis[1],alpha=0.2)
    ax.plot(x,est)
    ax.margins(x=0)
    ax.set_ylabel (varToplot)
    

for c in trajectories2:
    fig,(ax1,ax2) = plt.subplots(nrows=2, sharex=True)
    for exp in trajectories2[c]:
        tsplotDF(ax1, trajectories2[c][exp], 'ttBin', 'speed')
        ax1.set_ylim((-150, 150))
        tsplotDF(ax2, trajectories2[c][exp], 'ttBin', 'angular_velocity')
        ax2.set_ylim((-0.25, 0.25))
    fig.show()

  
#%% Do PCA on the 30 min time windows
from scipy import stats
from sklearn.decomposition import PCA

#first need to average the tracks, z-score and then standardise
    #for each time chunk
        #take mean for every track on each plate, and then median, std, 10th and 90th quartile for plate
features = {}
featMatMean = pd.DataFrame()
for chunk in trajectories2:
    features[chunk]=pd.DataFrame()
    for rep in trajectories2[chunk]:
        worms = np.unique(trajectories2[chunk][rep]['worm_index'])
        for worm in worms:
            temp = trajectories2[chunk][rep][trajectories2[chunk][rep]['worm_index']==worm].mean().to_frame().transpose()
            temp['rep'] = rep
            temp['chunk'] = chunk
            temp['drug'] = list(temp['rep'])[0].split('_')[0]
            features[chunk] = features[chunk].append(temp)
            del temp
        #fill in the nans for each drug
    for drug in allDrugs:
        temp2 = features[chunk][features[chunk]['drug'] == drug]
        temp2 = temp2.fillna(temp2.mean(axis=0))
        featMatMean = featMatMean.append(temp2) #put into one big dataframe
    
    features[chunk] = features[chunk].reset_index(drop=True)
featMatMean = featMatMean.reset_index(drop=True)

#Save the chunked feature dataframe to a .csv file so that can be combined with the other experiments
writer = pd.ExcelWriter(os.path.join(os.path.dirname(directoryA), 'LongExposureFeatures.xlsx'))
for chunk in features.keys():
    features[chunk].to_excel(writer, sheet_name = str(chunk))
writer.save()    
writer.close()

#Save the featMatMean to a .csv file so that can be combined with the other experiments
writer = pd.ExcelWriter(os.path.join(os.path.dirname(directoryA), 'LongExposureFeatMatMean.xlsx'))
featMatMean.to_excel(writer, sheet_name = 'FeatMatMean')
writer.save() 

#%% Import excel sheet generated on MacPros

%reset_selective -f trajectories2, featMatMean, features

from scipy import stats

#import spreadsheet
featMatMean = pd.read_excel(os.path.join(os.path.dirname(directoryA), 'LongExposureFeatMatMean.xlsx'), na_values = 'nan')

allDrugs = list(np.unique(featMatMean['drug']))
allChunks = list(np.unique(featMatMean['chunk']))
allConds = list(np.unique(featMatMean['rep']))

#zscore
featZ = pd.DataFrame(stats.zscore(featMatMean.iloc[:,:-3], axis=0), columns = featMatMean.iloc[:,:-3].columns)
featZ = pd.concat([featZ, featMatMean[['rep', 'drug', 'chunk']]], axis=1)

#filter out nan features
featZ.drop(featZ.columns[np.sum(featZ.isna())>(featZ.shape[0]/2)], axis=1, inplace=True)
featZ = featZ.drop(['timestamp', 'worm_index', 'ttBin'],axis=1)

#take median and IQR for all these features to make a plate summary
featZm = pd.DataFrame()
featZiqr = pd.DataFrame()
for chunk in allChunks:
    foo = featZ['chunk'] == chunk
    for rep in allConds:
        bar = featZ['rep']==rep
        temp = featZ[foo & bar]
        temp2 = temp.median().to_frame().transpose()
        temp2['rep'] = rep
        temp2['drug'] = rep.split('_')[0]
        featZm = featZm.append(temp2)

        temp3 = temp.quantile(0.75) - temp.quantile(0.25)
        temp3 =pd.concat([temp3[:-1].to_frame().transpose(), temp2[['rep', 'drug', 'chunk']]], axis=1)
        featZiqr = featZiqr.append(temp3)
        del temp, temp2, temp3

#test is any nan values
featZm.isnull().values.any()
featZiqr.isnull().values.any()

#reset index
featZm = featZm.reset_index(drop=True)
featZiqr = featZiqr.reset_index(drop=True)

#combine the mean and iqr feature matrices
featZm = pd.concat([featZm.iloc[:,:-3].rename(columns=lambda x: x +'_med'), featZm.iloc[:,-3:]], axis=1)
featZiqr = pd.concat([featZiqr.iloc[:,:-3].rename(columns = lambda x:x + '_iqr'), featZiqr.iloc[:,-3:]], axis=1)

#concatenate
featZsummary = pd.concat([featZm.iloc[:,:-3], featZiqr.iloc[:,:-3], featZm.iloc[:,-3:]], axis=1)

del featZm, featZiqr,featZ 

#covert to float for clustermap
featZall = featZsummary[featZsummary.iloc[:,:-3].columns].astype(float) 
featZall = pd.concat([featZall, featZsummary[['rep', 'drug', 'chunk']]], axis = 1)

#plot as clustermap
    #set lut to chunk number
cmap = sns.color_palette('tab20', np.unique(featZall['chunk']).shape[0])
lut = dict(zip(np.unique(featZall['chunk']), cmap))

rowColors = featZall['chunk'].map(lut)#map onto the feature Matrix

cg = sns.clustermap(featZall.iloc[:,:-3],  metric  = 'euclidean', cmap = 'inferno', \
           row_colors = rowColors)
plt.setp(cg.ax_heatmap.yaxis.set_ticklabels\
     (featZall['drug'][cg.dendrogram_row.reordered_ind]))
#plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize = 8)
#plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation = 90, fontsize = 10) 
col = cg.ax_col_dendrogram.get_position()
cg.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*1, col.height*1])
plt.title(allDrugs[drug])
#save fig

sns.clustermap(featZall.iloc[:,:-3], row_colors =rowColors)

#and make lut map
#make a figure of the colorbar
colors = [(v) for k,v in lut.items()]
    #plot separately
plt.figure()
ax = plt.imshow([colors])
ax.axes.set_xticklabels(range(0,11,1))
ax.axes.set_xticklabels(lut.keys(), rotation = 90)
ax.axes.set_xticks(np.arange(0,10,1))
ax.axes.xaxis.set_ticks_position('top')

del featZiqr, featZm
#%% PCA analysis
from sklearn.decomposition import PCA
import PCA_analysis as PC_custom 

#make array of z-scored data
X = np.array(featZall.iloc[:,:-3].values)

#initialise PCA
pca = PCA()
X2= pca.fit_transform(X)
cumvar = np.cumsum(pca.explained_variance_ratio_)
thresh = cumvar <= 0.95 #set 95% variance threshold
cut_off = int(np.argwhere(thresh)[-1])
#make a plot
sns.set_style('whitegrid')
plt.plot(range(0, len(cumvar)), cumvar*100)
plt.plot([cut_off, cut_off], [0, 100], 'k')
plt.text(cut_off, 100, cut_off)
plt.xlabel('Number of Principal Components', fontsize =16)
plt.ylabel('variance explained', fontsize =16)

#now put the 1:cut_off PCs into a dataframe
PCname = ['PC_%d' %(p+1) for p in range (0,cut_off+1)]
PC_df = pd.DataFrame(data= X2[:,:cut_off+1], columns = PCname)
PC_df['drug'] = featZall['drug']
PC_df['chunk'] = featZall['chunk']

#make the PC plots
PC_custom.PC12_plots(PC_df, [], 'all' , directoryA, 'tif', 'chunk')
PCmean, PCsem = PC_custom.PC_av(PC_df, ['PC_1', 'PC_2'], 'chunk')
test = ['DMSO', 'V3']
PC_custom.PC_traj(PCmean, PCsem,['PC_1', 'PC_2'], str(test) , directoryA, 'tif', cmap, test)



#which features contribute to the variance?
#components that explain the variance
    #make a dataframe ranking the features for each PC and also include the explained variance (z-normalised)
        #in separate dataframe called PC_sum
PC_feat = [] #features
PC_sum =[] #explained variance
for PC in range(0, len(PCname)):
    PC_feat.append(list(featZall.iloc[:,:-3].columns[np.argsort(pca.components_[PC])]))
    PC_sum.append(list((pca.components_[PC])/ np.sum(abs(pca.components_[PC]))))

#dataframe containing standards scores of contibution of each feature
        #rows correspond to PCs
PC_vals = pd.DataFrame(data= PC_sum, columns =  featZall.iloc[:,:-3].columns)

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
plt.savefig(os.path.join(os.path.dirname(directoryA), 'Figures', 'agar_biplot.png'))
    

#can also just plot the PCs across the time chunks
for PC in range(0,4):
    plt.subplot(1,4,PC+1)
    ax =sns.pointplot(x="chunk", y=PC_df.columns[PC], data=PC_df, hue = 'drug', legend = True)
    ax.legend_.remove()
    plt.ylim([-0.5,4])
plt.legend(loc=2, bbox_to_anchor=(1.05, 1) ,ncol = 1, frameon= True)
plt.show()

#%% Calculate euclidean distance between 
    #and plot the distance between the DMSO and No_compound controls and each drug over the time chunks

from scipy.spatial import distance

#average the featMatMean matrix to calculate distance between control and drug   
featMatMeanMean = pd.DataFrame()
for drug in allDrugs:
    temp1 = featZall[featZall['drug']==drug]
    for chunk in np.unique(featZall['chunk']):
        temp2 = temp1[temp1['chunk']==chunk].mean()
        temp2['drug'] = drug
        temp2['chunk'] = chunk
        featMatMeanMean = featMatMeanMean.append(temp2.to_frame().transpose(), ignore_index=True)
        del temp2
    del temp1
#actually not going to use this

dist1 = pd.DataFrame(distance.squareform(distance.pdist(featZall.iloc[:,:-3])))
dist1 = pd.concat([dist1, featZall[['rep', 'drug', 'chunk']]], axis=1)

#compare to DMSO
DMSOres = dist1.iloc[:,dist1[dist1['drug']=='DMSO'].index]
#DMSOres.columns = 
DMSOres = pd.concat([DMSOres, dist1[['drug', 'chunk']]], axis = 1)

#need to remodel so that only comparing drugs at the same timepoint
eye = pd.DataFrame(np.eye(3, dtype = bool))
DMSOmat = pd.DataFrame()
for drug in allDrugs:
    foo = DMSOres['drug'] == drug
    foo2 = DMSOres['drug'] == 'DMSO'
    for chunk in allChunks:
        bar = DMSOres['chunk'] == chunk        
        temp = DMSOres[foo&bar]
        temp = temp.loc[:,DMSOres[foo2&bar].index]
        eye.columns = temp.columns
        eye.index = temp.index
        temp2 = np.array(temp[eye]).flatten()
        temp2 = pd.DataFrame(temp2[np.where(~np.isnan(temp2))])
        temp2 =temp2.transpose()
        temp2['drug'] = drug
        temp2['chunk'] = chunk
        DMSOmat = DMSOmat.append(temp2)
        del temp, temp2, bar
    del foo, foo2

#reset index
DMSOmat = DMSOmat.reset_index(drop=True)
DMSOmat2 = pd.DataFrame(pd.concat([DMSOmat.loc[:,0], DMSOmat.loc[:,1],\
                      DMSOmat.loc[:,2]], axis=0, ignore_index=True))
DMSOmat2 = DMSOmat2.rename(columns = {0:'pdist'})
DMSOmat2['drug'] = np.matlib.repmat(DMSOmat['drug'], 3,1).flatten()
DMSOmat2['chunk'] = np.matlib.repmat(DMSOmat['chunk'], 3,1).flatten()
DMSOmat2['rep'] = np.matlib.repmat(DMSOmat['rep'], 3,1).flatten()

sns.tsplot (data = DMSOmat2, time= 'chunk', value = 'pdist')#, unit = 'drug')
plt.show()

#define a function to plot this
def tsplotDF(ax, data, time, varToplot, name):
    """ Custom function for shaded error bars
    ax : axes generated
    data : pandas dataframe input
    time : timebins
    varToplot: variable to plot
    name: label for the line
    """
    #data.name = data
    x = np.unique(data[time])
    est = np.empty(shape = x.shape[0])
    sem =np.empty(shape = x.shape[0])
    for tbin in range(len(x)):
        est[int(tbin)] = data[varToplot][data[time]==x[tbin]].mean()
        sem[int(tbin)] = data[varToplot][data[time]==x[tbin]].std()
    cis = (est - sem, est + sem)
    ax.fill_between(x,cis[0],cis[1],alpha=0.2)
    ax.plot(x,est, label = name)
    ax.margins(x=0)
    ax.set_ylabel (varToplot)
    

fig, ax1 = plt.subplots(1,1)
for drug in allDrugs:
    tsplotDF(ax1, DMSOmat2[DMSOmat2['drug']==drug], 'chunk', 'pdist', drug)
ax1.legend()
plt.xlim([0,10])
plt.show()
plt.savefig(os.path.join(os.path.dirname(directoryA), 'Figures', 'pdist.png'))


ax1.set_ylim((-150, 150))
tsplotDF(ax2, trajectories2[c][exp], 'ttBin', 'angular_velocity', drug)
ax2.set_ylim((-0.25, 0.25))
fig.show()

      
plt.plot(DMSOmat.iloc[:,:-1].transpose())
plt.legend = DMSOmat['drug']
    
    
