#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 20:50:37 2022

@author: jace
"""
import numpy as np
import pymatgen.core.bonds  as mgBond


def TreeSearch(x, visited, bond_array):
    """
    
    Parameters
    ----------
    x : init
        current start node.
    visited : array
        atom visted.
    bond_array : array
        bond info.

    Returns
    -------
    visited : array
        atom visted..
    current_molecular_index : list 
        current molecular index .

    """
    unexploredNode = [x]
    current_molecular_index = [x]
    
    while unexploredNode:
        current = unexploredNode[0]
        visited.append(current)
        unexploredNode.remove(current)
        bonded = np.where( bond_array[current,:]==True)[0]
        for x in bonded:
            if (x not in visited) and (x not in unexploredNode):
                unexploredNode.append(x)
                current_molecular_index.append(x)
    
    return visited, current_molecular_index        

def CheckConnectivity(linker, linker_or_metal = None, mode = 'norm'):
    
### a dumb way to check the connectivity of the atoms  ###
    coord_dict = []
    len1,len2 = len(linker),len(linker_or_metal)
    bond_array = np.full((len1,len2), False, dtype=bool)
    
    if mode == 'coord':
        for i in range(len1):
            for j in range(len2):
                _distance_ = linker[i].distance(linker_or_metal.sites[j])
                if ((linker[i].specie.value =='O') or (linker[i].specie.value =='N')) and _distance_ < 2.8:
                    bond_array[i,j] = True
                    if j not in coord_dict:
                        coord_dict.append(j)
                else:
                    bond_array[i,j] = False  
                   
    if mode == 'norm':
        for i in range(len1):
            for j in range(len2):
                if i != j:
                    bond_array[i,j] = mgBond.CovalentBond.is_bonded(linker[i],linker_or_metal.sites[j])      
            
    return bond_array, coord_dict
    
    
    return