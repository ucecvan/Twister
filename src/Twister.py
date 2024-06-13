#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:00:14 2024
#Mary had a little lamb
@author: alex
"""

import numpy as np
from Pipeline import DPA
import random

kb = 8.314462618 * (10**-3)
#beta = 1/(kb*T)


def distanceCircle(x,y):
    '''
    Function for the shortest distance around a unit n-sphere for two points on the surface of said n-sphere
    
    args:
        x(tuple): The coordinates of one of the points
        y(tuple): The coordinates of the second point. Must have the same dimensionality of the first point
    
    returns
        float: The distance between the two points
    '''
    pyt1 = []
    for i in range(len(x)):
        pyt2 = y[i]-x[i]
        if abs(pyt2) > np.pi:
            if pyt2 > 0:
                pyt2 = (pyt2) - (2*np.pi)
            elif pyt2 < 0:
                pyt2 = (2*np.pi) + pyt2
            
        pyt3 = pyt2**2
        pyt1.append(pyt3)
    pyt4 = (sum(pyt1))**(0.5)
    return(pyt4)

    
def clusterPop(labels):
    '''
    Builds a dictionary of the populations of the clusters
    
    args:
        labels(list): DPA.labels_ list from DPA clustering object
    returns:
        dict: population dictionary, keys are cluster indices, values are cluster populations

    '''
    popDict = {}
    for i in labels:
        if i not in popDict.keys():
            popDict[i] = 1
        else:
            popDict[i]+=1
    return popDict
    

def Eaverage(Est, r):
    '''
    Replaces the density of the cluster centers with the average density of all points within a fixed radius of the center
    args:
        Est(DPA object): DPA object fitted on data
    returns:
        list: New, averaged densities of cluster centers
    
    '''
    avdens = []
    for i in Est.centers_:
        inbrs = []
        d = 0 
        while d < r:
            for k in range(len(Est.nn_indices_[i])):
                d = Est.nn_distances_[i][k]
                if d < r:
                    inbrs.append(Est.nn_indices_[i][k])
        idens = []
        for j in inbrs:
            idens.append(Est.densities_[j])
        avden = np.mean(idens)
        avdens.append(avden)
    return(avdens)


def Esmooth(Est, r):
    '''
    Replaces the density of all points with the average density of all points within a fixed radius of itself
    args:
        Est(DPA object): DPA object fitted on data
    returns:
        list: New, averaged densities of data points
    '''
    avdens = []
    for i,l in enumerate(Est.densities_):
        inbrs = []
        d = 0 
        while d < r:
            for k in range(len(Est.nn_indices_[i])):
                d = Est.nn_distances_[i][k]
                if d < r:
                    inbrs.append(Est.nn_indices_[i][k])
        idens = []
        for j in inbrs:
            idens.append(Est.densities_[j])
        avden = np.mean(idens)
        avdens.append(avden)
    return(avdens)


def CompareClusters(Ref, Clusdict, biid):
    '''
    Compares two sets of clusters generated from datasets originating from the same simulation. 
    args:
        Ref (dict): Cluster dictionary for the reference cluster set. Generally the cluster set generated from the largest available dataset
        Clusdict (dict): Cluster dictionary to be compared with Ref. Generally generated from a smaller, downsampled dataset
        biid (list): list indicating the structure of the COLVAR file and cluster dictionary.
    returns:
        meansep (float): Mean separation in Euclidean radians of the cluster sets' cluster centers
        meanedif (float): Mean energy difference in kJ/mol of the cluster sets' cluster centers
    
    '''
    nbias = biid.count(1)
    nangle = len(biid)
    dlist1 = []
    for i in Clusdict:
        dlist2 = []
        for k in Ref:
            d = distanceCircle(Clusdict[i][:nangle],Ref[k][:nangle])
            dlist2.append(d)
        da2 = np.array(dlist2)
        dlist1.append(da2)
    da1 = np.array(dlist1)
    #return(da1)
    NearestRefCluster = []
    RefClusterDist = []
    for i in da1:
        idx = np.where(i == min(i))
        NearestRefCluster.append(idx)
        RefClusterDist.append(min(i))
    edif = []
    for i in range(len(NearestRefCluster)):
        e = abs(Clusdict[i][nangle+nbias+1] - Ref[NearestRefCluster[i][0][0]][nangle+nbias+1])
        edif.append(e)
        
    meansep = np.mean(RefClusterDist)
    meanedif = np.mean(edif)
    return(meansep, meanedif)

class Twister:
    '''
    Main class for handling the DPA clustering, reweighing, free energy calculations on datasets of conformational datasets sampled
    by enhanced sampling molecular dynamics simulations. Also includes methods for performing sanity checks on high-dimensional results.
    
    attributes:
        biid (list): reflects the structure of the Colvar file being interpreted. Length of list corresponds to number of torsions in
        T (Float) : Temperature of simulation in Kelvin, default = 300 K.
        smoothrad (float): Smoothing radius for density smoothing
        Colvar. Items in list are either 0 or 1. Unbiased dihedrals are represented with 0 and biased torsions are represented wtih 1.
        Colvar (np.array): Array representation of the COLVAR file generated by MD. Contains columns representing time, torsion values,
        and bias values.
        coords (np.array): Extracted directly from Colvar, array of the all the torsion values sampled through MD
        biases (np.array): Extracted directly from Colvar, array of the final bias values at the the equivalent place in coords
        metric (callable): Distance metric used to judge separation in CV space (default: distanceCircle, which measures periodic Euclidean distance
        in CV space)
        DPAobj(DPA.DensityPeaksAdvanced): The DPA object associated with the clustering process. generated by the Clustering method
        RefClusters(dict): Properties of cluster centers (free energy minima) presented as a dictionary. For each center, a list is
        stored containing, in order: the coordinates, the value(s) of the biases, the population of the cluster, the free energy of the
        cluster (relative to the lowest energy cluster), the time in ns at which this configuration was sampled by the MD simulation
        FAdata (list): A list of lists containing the key results of the clustering consistency analysis. within are: dlist, a list of the
        mean distances between equivalent centers; elist, a list of the mean energy differences between equivalent centers; nlist, list of 
        the degrees of subsampling carried out by the check; clist, a list of the number of clusters identified during each iteration
        of the consistency analysis; cdlist, a list of the cluster dictionaries for each iteration of the consistency analysis
        energies (np.array): array of energies for each coordinate, calculated from density values
        energies2 (np.array): array of energies for each coordinate, with data points above 100 kJ/mol removed
        coords2 (np.array): same as coords,with data points above 100 kJ/mol removed
        bias2 (np.array): same as biases, with data points above 100 kJ/mol removed
        Colvar2 (np.array): same as Colvar, with data points above 100 kJ/mol removed
        
    '''
    def __init__(self, biid, metric = distanceCircle, T = 300, smoothrad = 0.1, Colvar = None, coords = None, biases = None, DPAobj = None, RefClusters = None, coords2 = None, bias2 = None, Colvar2 = None, energies = None, energies2 = None, FAdata = None):
        self.biid = biid
        self.T = T
        self.smoothrad = smoothrad
        self.Colvar = Colvar
        if self.Colvar == None:
            self.coords = coords
            self.biases = biases
        else:
            self.ColvarSplitter()
        self.metric = metric
        self.DPAobj = DPAobj
        self.RefClusters= RefClusters
        self.FAdata = FAdata
        self.coords2 = coords2
        self.bias2 = bias2
        self.Colvar2 = Colvar2
        self.energies = energies
        self.energies2 = energies2

    
            
    def ColvarLoader(self, ColvarLoc, skip):
        '''
        Loads a Plumed generated COLVAR file and sets the Colvar, coords, bias, attributes
        args:
            ColvarLoc (str): The file location for the Colvar file containing the molecular configurations and associated biases
            skip (int): Number of lines to skip in Colvar file
        actions:
            Defines self.Colvar
            
        
        '''
        Colvar = np.loadtxt(ColvarLoc, skiprows = skip)
        Colvar = Colvar[int(len(Colvar)/3):,:]
        self.Colvar = Colvar
        self.ColvarSplitter()

    def ColvarSetter(self, NewColvar):
        '''
        Sets the Colvar attribute, sets coords and bias as follows from that. 
        args:
            NewColvar(np.array): New Colvar, data item containing the molecular configurations and associated biases
        actions:
            sets self.Colvar, self.coords, self.bias
        '''
        self.Colvar = NewColvar
        self.ColvarSplitter()


    
    
    def DownSample(self, n):
        '''
        args:
            n(int): size of downsampled dataset to be generated
        returns:
            ColvarD (np.array): A downsampled Colvar, randomly sampled from the original, of size n
        '''
        ColvarL = list(self.Colvar)
        ColvarDl = random.sample(ColvarL, n)
        ColvarD = np.array(ColvarDl)
        return(ColvarD)
    
    def ColvarSplitter(self):
        '''
        actions:
            Updates the coord and bias attributes to match the data in the Colvar attribute
        '''
        nangle = len(self.biid)
        coords = self.Colvar[:,1:(nangle+1)]
        bias = self.Colvar[:,(nangle+1):]
        self.coords = coords
        self.bias = bias
    

    
    def presClusters(self, coords, bias, Colvar, clusterPop, est): 
        '''
        
        args:
            clusterPop(dict): The cluster population dictionary generated by the clusterPop function
            data(np.array): The original structure array on which the clustering was performed
            ColvarD(np.array): The complete COLVARD array
            est (DPA.DensityPeaksAdvanced): DPA object
        returns:
            dict: python dict where keys are cluster indices and values are a list where the first members are the
            clusters coordinates in phase space and the final member is it's population
        
        '''
        ClusDict = {}
        count = 0
        avdens = Eaverage(est, self.smoothrad)
        denmax = np.max(avdens)
        emin = -kb*self.T*(denmax)
        for i in range(len(est.centers_)):
            if clusterPop[i] > (len(coords)*0.01):
                ClusDict[count] = list(coords[est.centers_[i]])
                ClusDict[count] = ClusDict[count] + list(bias[est.centers_[i]])
                ClusDict[count].append(clusterPop[i])
                ClusDict[count].append(-kb*self.T*(avdens[i]) - emin)
                ClusDict[count].append(Colvar[est.centers_[i]][0])
                count += 1
        return ClusDict
    


    def Clustering(self):
        '''
        Performs a DPA clustering analysis on the the configurations from COLVAR. Reweighs the density of data points to correct from the
        distorting effect of the bias and reruns DPA on the reweighed densities to obtain the per-point density map. Converts densities to 
        free energies (kJ/mol)
        
        actions:
            DPA object assigned as attribute DPAobj
            Resulting Cluster set assigned as attribute RefClusters
            initial energies assigned as attribute energies
            coordinates, biases, Colvar, and energies from dataset trimmed to remove datapoints of energies > 100 kJ/mol are
            assigned attributes coords2, bias2, Colvar2, energies2
        
        '''
        beta = 1/(kb*self.T)
        nbias = self.biid.count(1)
        nangle = len(self.biid)
        
        estRW = DPA.DensityPeakAdvanced(Z=0.1, metric=distanceCircle, dim=nangle, k_max=int(len(self.coords)/10))
        estRW.fit(self.coords)
      
        den = np.array(estRW.densities_)
        popDictRW = clusterPop(estRW.labels_)
        ClusDictRW = self.presClusters(self.coords, self.bias, self.Colvar, popDictRW, estRW)
        biasum = 0
        for i in range(nbias):
            biasum += (self.bias[:,i] - min(self.bias[:,i]))
        newdens = np.log((np.e**(beta*biasum))*(np.exp(den)))
        newdene = np.full(len(den), ((beta * 2.5)/2))
        params = estRW.get_computed_params()
        estRW.set_params(**params)
        estRW.set_params(Z=0.00, densities = list(newdens), err_densities = list(newdene))
        estRW.fit(self.coords)
        newdens = Esmooth(estRW, self.smoothrad)
        newdens = np.array(newdens)
        newdene = np.array(newdene)
        ens = -1/beta * (newdens)
        ens = ens - min(ens)
        self.energies = ens
    
        newdens1 = newdens[ens < 100]
        newdene1 = newdene[ens < 100]
        coords2 = self.coords[ens < 100]
        bias2 = self.bias[ens < 100]
        Colvar2 = self.Colvar[ens < 100]
        ens2 = ens[ens < 100]
        kval = estRW.k_max
        estRW = DPA.DensityPeakAdvanced(Z=1, metric=distanceCircle, dim=nangle, k_max=kval)
        estRW.set_params(densities = list(newdens1), err_densities = list(newdene1))
        estRW.fit(coords2)
        params = estRW.get_computed_params()
        estRW.set_params(**params)
        estRW.set_params(densities = list(newdens1), err_densities = list(newdene1))
        estRW.fit(coords2)
        
        popDictRW = clusterPop(estRW.labels_)
        ClusDictRW = self.presClusters(coords2, bias2, Colvar2, popDictRW, estRW)
    
        self.DPAobj = estRW
        self.RefClusters = ClusDictRW
        self.coords2 = coords2
        self.bias2 = bias2
        self.Colvar2 = Colvar2
        self.energies2 = ens2


    
    def SCAnalysis(self, samplerange):
        '''
        Carries out a consistency analysis on the results of the clustering to assist in determining the degree to which the 
        finite number of data points represent the simulation. Repeats the main clustering analysis on smaller datasets.
        args:
            samplerange: range of dataset sizes to carry out clustering analysis on
        
        actions:
            assigns FAdata attribute, a list containing the following lists:
                dlist: list of mean cluster separations between smaller-datasets cluster sets and reference cluster set
                elist: list of mean energy differences between smaller-datasets cluster sets and reference cluster set
                nlist: list of the sizes of the smaller datasets
                clist: list of the number of cluster centers in the smaller-datasets cluster sets
                cdlist: list of cluster dictionaries for each of the smaller-datasets cluster sets
        
        
        '''
        dlist = []
        elist = []
        nlist = []
        clist = []
        cdlist = []
        for i in samplerange:
            DS = Twister(self.biid)
            ColvarD = self.DownSample(i)
            DS.Colvar = ColvarD
            DS.ColvarSplitter()
            DS.Clustering()
            compare = CompareClusters(self.RefClusters, DS.RefClusters, self.biid)
            dlist.append(compare[0])
            elist.append(compare[1])
            nlist.append(i)
            clist.append(len(DS.RefClusters))
            cdlist.append(DS.RefClusters)
            del(DS)
        self.FAdata = [dlist, elist, nlist, clist, cdlist]


        
    



