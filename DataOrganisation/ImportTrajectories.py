#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 16:11:23 2018

@author: ibarlow
"""

""" Import the split trajectories data from the long drug exposure experiments"""

import pandas as pd
import numpy as np
#from tkinter import Tk,filedialog
import os
import concurrent.futures
import time

#
#print ('Select Data Folder')
#root = Tk()
#root.withdraw()
#root.lift()
#root.update()
#root.directory = filedialog.askdirectory(title = "Select Trajectories folder", \
#                                         parent = root)
#
#if root.directory == []:
#    print ('No folder selected')
#else:
#    directory = root.directory
#    #find the folders within
#    reps = os.listdir(directory)
#    print(reps)

directory = '/Volumes/behavgenom_archive$/Ida/MultiWormTracker/LongExposure/120718LongExposure2/Results/30minChunkedTrajectories/HDF5'
reps = os.listdir(directory)
trajFilter = 250 #10 second filter for trajectories

def featMatGenerator(dirName, trajfile, trajFilter):
    """ Creates the feature dataframe for the trajectories and plate summaries
    for each time point
    Input - 
        dirName : directory containing all the trajectory files
    
        trajfile : chunked trajectory files from running parTierpsyIn.py script in format
        
        trajFilter : minimum number of frames for a trajectory to be counted in the summary
        
    
    Output - featMatTraj
        featMatPlate"""
        
    #load the data and extract feature vectors for each trajectory and plate summary for each chunk
    featMatTraj = {}
    featMatPlate = pd.DataFrame()
    try:
        if len(trajfile.split('_'))<10:
            fshort = '_'.join(trajfile.split('_')[0:-2:6])
        else:
            fshort = '_'.join(trajfile.split('_')[0:-1:7])
        featMatPlate = pd.DataFrame()
        with pd.HDFStore(os.path.join(dirName, trajfile), 'r') as fid:
            nChunks = list(fid.keys())
            for chunk in nChunks:
                chunkno = [int(s) for s in chunk.split('_') if s.isdigit()]
                chunkno = chunkno[0]

                featMatTraj[chunkno] = pd.DataFrame()
                nWorms = np.unique(fid[chunk]['worm_index'])
                for w in nWorms:
                    if fid[chunk][fid[chunk]['worm_index']==w].shape[0]>=trajFilter:
                        featMatTraj[chunkno] = featMatTraj[chunkno].append(\
                                   fid[chunk][fid[chunk]['worm_index']==w].mean(),ignore_index=True)
                
                featMatTraj[chunkno].reset_index(drop=True)
                         
                temp = featMatTraj[chunkno].median()
                temp = temp.drop(['worm_index', 'timestamp']).rename(lambda x: x +'_med').to_frame().transpose()
                
                temp2 = featMatTraj[chunkno].quantile(0.75) - featMatTraj[chunkno].quantile(0.25)
                temp2 = temp2.drop(['worm_index', 'timestamp']).rename(lambda x: x + '_iqr').to_frame().transpose()
                
                tempfinal = pd.concat([temp, temp2], axis = 1)
                tempfinal ['exp'] = fshort
                tempfinal['Chunk'] = chunk
                tempfinal ['drug'] = fshort.split('_')[0]
                
                featMatPlate = featMatPlate.append(tempfinal, ignore_index=True)
                del temp, temp2, tempfinal
                del nWorms
            del nChunks
     
        featMatPlate.reset_index(drop=True)      
        featMatPlate.drop(featMatPlate.columns[np.sum(featMatPlate.isna()>featMatPlate.shape[0]/2)], \
                                               axis=1, inplace = True)
    except OSError:
        print (trajfile + 'is invalid file format')        

    #write the featMatPlate to a .csv file
    featMatPlate.to_csv(os.path.join(os.path.dirname(dirName), fshort + '_FeatMatPlate.csv'))

    #save the featMatTraj to an excel file
    writer = pd.ExcelWriter(os.path.join(os.path.dirname(dirName), fshort + '_FatMatTraj.xlsx'))
    for chunk in featMatTraj.keys():
        featMatTraj[chunk].to_excel(writer, sheet_name = str(chunk))
    writer.save()
    
    return featMatTraj, featMatPlate


b =  (time.time())

with concurrent.futures.ThreadPoolExecutor(max_workers = 4) as ex:
    fut = []
    for file in reps:
        fut.append(ex.submit(featMatGenerator, directory, file, 250))        
    e = time.time()
print ('time taken : ' + str(e-b) + 'seconds')


##zscore and fill nans
#featZ =  
#
##may need to parallelerise            
#
##drop timestamp and worm_index before PCA
#
#test = reps[0]
#
#fid = pd.HDFStore(os.path.join(directory,test), 'r')
#
#

# =============================================================================
# Ideas for plotting :
# 1. Plot the number of trajectories extracted as a function of block size - 
#   this will be useful to compare to the NwormsTest data

# 2. Run same analysis as previously with PCA etc.

# 3. 
# =============================================================================
 
#test = reps[1]
#
#featMatTraj1, featMatPlate1 = featMatGenerator(directory, test, 250)
#
#now generate for all files using parallel processing

#with concurrent.futures.ThreadPoolExecutor(max_workers = 4) as ex:
#    fut = []
#    for file in reps:
#        fut.append(ex.submit(featMatGenerator, directory, file, 250))        
#e = time.time()
#print ('time taken : ' + str(e-b) + 'seconds')