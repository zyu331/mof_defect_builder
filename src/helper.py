#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 20:50:37 2022

@author: jace
"""
from cmath import isclose
import numpy as np
import os
import copy
from sklearn.cluster import DBSCAN,KMeans
from collections import Counter

import pymatgen.core.bonds  as mgBond
from pymatgen.io.vasp.inputs import Poscar
import pymatgen.core.structure as mgStructure
from pymatgen.core.bonds import CovalentBond, get_bond_length
from pymatgen.util.coord import all_distances, get_angle, lattice_points_in_supercell
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.operations import SymmOp
from scipy.sparse.csgraph import connected_components
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.vis.structure_vtk import StructureVis
from pymatgen.analysis.local_env import CrystalNN, BrunnerNN_real


import src.cappingAgent as cappingAgent 
import warnings

#################################################################################################
# important feature to be included:                                                             #
# 1. charge dict for compensation: either through auto method or handcoded dict file            #
# 2. coord bond info based on different metal center                                            #
#################################################################################################

#################################################################################################
# important Note:                                                             
# Lots of things in this file are ugly and need fix
#################################################################################################


def StructureAnalysis(linker, metal):
    def cluster_assginemnt(structure,length):     
        bond_array = np.full((length,length), 0, dtype=int)          
        for i in range(length):
            for j in range(i+1,length):
                try:
                    isbond = mgBond.CovalentBond.is_bonded(structure[i],structure[j])
                except:
                    isbond = False
                if isbond:
                    bond_array[i,j] = 1
                    bond_array[j,i] = 1          
                else:
                    bond_array[i,j] = 0
        cluster_assignment = connected_components(bond_array)

        return cluster_assignment

    len1,len2 = len(linker),len(metal)
    
    cluster_assignment_linker = cluster_assginemnt(linker,len1)
    # array for linker: linker atom can only coord to one metal
    coord_bond_array = np.full(len1, np.nan)
    # dict for metal: one metal can have multiple coord atom
    coord_bond_list = [[] for _ in metal]

    # generate the coord bond array
    for i in range(len1):
        for j in range(len2):
            _distance_ = linker[i].distance(metal.sites[j])
            # abs_distance = abs(linker[i].frac_coords - metal.sites[j].frac_coords)
            # TODO: no coord bond info in pymatgen, thus use a naive cutoff at 2.8 A, needs to be fixed            
            if ((linker[i].specie.value =='O') or (linker[i].specie.value =='N')) and _distance_ < 2.8 and metal[j].specie.is_metal:
                coord_bond_array[i] = j
                coord_bond_list[j].append(i)

    # group the metal cluster
    cluster_array = np.zeros((len2,len2))
    for i in range(len2):
        for j in range(i+1,len2):
            if metal[i].distance(metal[j]) < 4.0:
                cluster_array[i,j] = 1
                cluster_array[j,i] = 1
            else:
                cluster_array[i,j] = 0
    cluster_assignment_metal = connected_components(cluster_array)

    """ decide whether the linker has pbc problem: i.e. in the boudary.
    use the "max-void algorithm" to define.
    """
    pbc_linker = np.full((len1,3), ['no_pbc','no_pbc','no_pbc'])
    pbc_setting = [(False,True,True),(True,False,True),(True,True,False)]

    for i in range(cluster_assignment_linker[0]):
        linker_coords = []
        indexes = np.where(cluster_assignment_linker[1]==i)[0]
        for index in indexes:
            # append index to the end of the frac_coords
            linker_coords.append( np.append(linker[index].frac_coords,int(index)))
        linker_coords = np.array(linker_coords)

        for axis in range(3):
            working_linker = mgStructure.Structure.from_sites([linker[_index_] for _index_ in indexes])
            working_linker._lattice._pbc = pbc_setting[axis]
            cluster_assignment_pbc = cluster_assginemnt(working_linker,len(working_linker))
            working_linker._lattice._pbc = (True,True,True)
            if cluster_assignment_pbc[0] > 1:
                for iii in range(cluster_assignment_pbc[0]):
                    indexes_cluster = np.where(cluster_assignment_pbc[1]==iii)[0]
                    cluster_min = np.min([ working_linker[_index_cluster_].frac_coords[axis] for _index_cluster_ in indexes_cluster],axis=0)
                    cluster_max = np.max([ working_linker[_index_cluster_].frac_coords[axis] for _index_cluster_ in indexes_cluster],axis=0)
                    if cluster_min > (1-cluster_max):
                        for xx in indexes_cluster:
                            pbc_linker[indexes[xx]][axis] = 'large'
                    else:
                        for xx in indexes_cluster:
                            pbc_linker[indexes[xx]][axis] = 'small'
                        

    pbc_metal = np.full((len2,3), ['no_pbc','no_pbc','no_pbc'])
    
    for i in range(cluster_assignment_metal[0]):
        metal_coords = []
        indexes = np.where(cluster_assignment_metal[1]==i)[0]
        for index in indexes:
            # append index to the end of the frac_coords
            metal_coords.append( np.append(metal[index].frac_coords,int(index)))
        metal_coords = np.array(metal_coords)
        for axis in range(3):
            metal_coords = metal_coords[metal_coords[:,axis].argsort()]
            if len(metal_coords) == 2:
                class temp_class(object):
                    pass
                cluster, clusters_count = temp_class(), {}
                if abs(metal_coords[0,axis] - metal_coords[1,axis]) > 0.5:
                    cluster.labels_ = [0,1]
                    clusters_count[0],clusters_count[1] = 1,1
                else:
                    cluster.labels_ = [0,0]
                    clusters_count[0] = 2
            else:
                cluster = DBSCAN(eps = float(3/linker.lattice.abc[axis]),metric='euclidean').fit(metal_coords[:,axis].reshape(-1,1))
                clusters_count = Counter(cluster.labels_)
            if len(clusters_count) == 1:
                break
            elif len(clusters_count) == 2:
                type1, type2 = list(clusters_count.keys())[0],list(clusters_count.keys())[1]
                cluster1 = np.array([metal_coords[_index_] for _index_,ele in enumerate(cluster.labels_) if ele == type1])
                cluster2 = np.array([metal_coords[_index_] for _index_,ele in enumerate(cluster.labels_) if ele == type2])
                if np.mean(cluster1[:,axis]) > np.mean(cluster2[:,axis]):
                    for ele in cluster1:
                        pbc_metal[int(ele[-1])][axis] = 'large'
                    for ele in cluster2:
                        pbc_metal[int(ele[-1])][axis] = 'small'
                else:
                    for ele in cluster1:
                        pbc_metal[int(ele[-1])][axis] = 'small'
                    for ele in cluster2:
                        pbc_metal[int(ele[-1])][axis] = 'large'                    
            else:
                print("pbc error please check metal pbc")

    return cluster_assignment_linker, pbc_linker, coord_bond_array, cluster_assignment_metal, coord_bond_list, pbc_metal

def NbMAna(mof, deleted_linker, _index_linker_):

    """
    goal of this helper function: 
        input a seperate linker(specified by _index_linker)
        return the metal-coordLinker-Linker1stneighour pair(three components!) 
    
    return varibles:
    linker_coord_list : (type, coord_linker, coord_linker_neighbour, metal)

    """
    fixed_id = [x.fixed_id for x in mof]
    linker_coord_list = []
    coord_atom_linker = []
    for index in _index_linker_:
        if ~np.isnan(deleted_linker[index].NbM):
            coord_atom_linker.append(index)
    
    childs = {}
    for i in range(len(coord_atom_linker)):
        try:
            loc_env = CrystalNN()
            child = loc_env.get_nn_info(deleted_linker,coord_atom_linker[i])
            childs[i] = child
        except:
            loc_env = BrunnerNN_real()
            child = loc_env.get_nn_info(deleted_linker,coord_atom_linker[i])
            childs[i] = child
    
    visited = [0]*len(coord_atom_linker)
    for i in range(len(coord_atom_linker)):
        if visited [i] == 1:
            continue
        common_list = []
        child_1 = [x['site_index'] for x in childs[i]]
        for j in range(i+1,len(coord_atom_linker)):
            if visited[j] == 1 :
                continue
            
            child_2 = [x['site_index'] for x in childs[j]]
            if len(list( set(child_1).intersection(child_2))) == 0 :
                continue
            else:
                common_list = list( set(child_1).intersection(child_2))
                pairs = [deleted_linker[coord_atom_linker[i]],deleted_linker[coord_atom_linker[j]]]
                metal_1 = mof[ [ii for ii,x in enumerate(fixed_id) if x==int(deleted_linker[coord_atom_linker[i]].NbM)][0]]
                metal_2 = mof[ [ii for ii,x in enumerate(fixed_id) if x==int(deleted_linker[coord_atom_linker[j]].NbM)][0]]
                metal = [metal_1, metal_2]
                visited[j] = 1
        
        if len(common_list) == 1 and deleted_linker[coord_atom_linker[i]].species_string == 'O' and deleted_linker[coord_atom_linker[j]].species_string == 'O':
            linker_coord_list.append(('OO',pairs, deleted_linker[common_list[0]], metal ))
                # raise("local coordination env error")
        else:
            metal = mof[ [ii for ii,x in enumerate(fixed_id) if x==int(deleted_linker[coord_atom_linker[i]].NbM)][0]]
            if deleted_linker[coord_atom_linker[i]].species_string == 'N':
                linker_coord_list.append(('N',deleted_linker[coord_atom_linker[i]], [ deleted_linker[_index_] for _index_ in child_1], metal ))
            elif deleted_linker[coord_atom_linker[i]].species_string == 'O':
                linker_coord_list.append(('O',deleted_linker[coord_atom_linker[i]], [ deleted_linker[_index_] for _index_ in child_1], metal ))
            else:
                raise("local coordination env error")
        visited[i] = 1
     
    return linker_coord_list

def WarrenCowleyParameter(neighbor_list, center_atom, noncenter_atom):
    '''
    neighbor_list: 2-D list. [[1, 'Yb', ['Nd', 'Yb']]...]
    center_atom: reference atom. 'Yb' or 'Nd'
    '''
    
    # n_neighbor = len(neighbor_list[0][-1])
    # find center atom list
    center_atoms =[]
    for neighbor in neighbor_list:
        if neighbor[1] == center_atom:
            center_atoms.append(neighbor)
    
    # neighbor_list_names = ['neighbor_'+str(i+1) for i in range(n_neighbor)]
    probs = []
    for _index_, _type_, _neighbor_list_ in neighbor_list:
        # list_name = [x[-1][i] for x in center_atoms]
        n_center_atom = [ neighbor_list[x][1] for x in _neighbor_list_ ].count(center_atom)
        n_noncenter_atom = [ neighbor_list[x][1] for x in _neighbor_list_ ].count(noncenter_atom)
        if (n_center_atom + n_noncenter_atom) != len(_neighbor_list_):
            raise "neighbor error"
        # if n_center_atom == 0:
        #     prob = 0
        else:
            prob = n_center_atom /( n_noncenter_atom + n_center_atom )
        probs.append(prob)
    prob_BgivenA = np.average(probs)
    n_center_atom = len(center_atoms)
    n_metal = len(neighbor_list)
    prob_A = n_center_atom/n_metal
    prob_B = 1-prob_A
#    alpha = 1-prob_A*prob_BgivenA/prob_B
    alpha = 1-prob_BgivenA/prob_A
    return alpha

def SwapNeighborList(neighbor_list, ID_1, ID_2):
    '''
    neighbor_list: 2-D list. [[1, 'Yb', ['Nd', 'Yb']]...]
    ID_1, ID_2: the two metals to be swapped
    '''
    new_neighbor_list = copy.deepcopy(neighbor_list)
    #swap center metals
    new_neighbor_list[ID_1][1], new_neighbor_list[ID_2][1] = neighbor_list[ID_2][1], neighbor_list[ID_1][1] 
    #update neighbors
    # for nebr in neighbor_list[ID_1][-1]:
    #     new_neighbor_list[nebr][-1][neighbor_list[nebr-1][-2].index(ID_1)] = neighbor_list[ID_2-1][1] 
    # for nebr in neighbor_list[ID_2-1][-2]:
    #     new_neighbor_list[nebr-1][-1][neighbor_list[nebr-1][-2].index(ID_2)] = neighbor_list[ID_1-1][1]
    return new_neighbor_list
    
def WriteStructure(output_dir, structure, name = 'POSCAR', sort = True):
    
    if sort == True:
        structure.sort()
    out_POSCAR = Poscar(structure=structure)
    # out_POSCAR.selective_dynamics=DynamicsM
    out_POSCAR.write_file(os.path.join(output_dir,name))                
    
    return

def addOH(coord_linker, coord_linker_neighbour, metal):

    if len(coord_linker_neighbour)==0:
        print('neighbour error, Cl added')
        O_site = copy.deepcopy(coord_linker)
        O_site.species = 'Cl'
        return [O_site]

    M_O = coord_linker.frac_coords - metal.frac_coords
    N_O,N_O_dist = [],[]
    for site in coord_linker_neighbour:
        _pos_diff_ = site.frac_coords - coord_linker.frac_coords

        N_O.append(PosDiffAdjust(_pos_diff_))
        N_O_dist.append(site.distance(coord_linker))

   
    O_coords = metal.frac_coords + M_O
    O_site = copy.deepcopy(coord_linker)
    O_site.species, O_site.frac_coords = 'O', O_coords

    O_H_bond_length =  0.97856
    H_coords1 = coord_linker.frac_coords + O_H_bond_length/N_O_dist[0]*N_O[0]
    H_sites1 = copy.deepcopy(coord_linker_neighbour[0])
    H_sites1.species, H_sites1.frac_coords = 'H', H_coords1

    return [O_site, H_sites1]


def addH2O(coord_linker, coord_linker_neighbour, metal):

    M_O = coord_linker.frac_coords - metal.frac_coords
    N_O,N_O_dist = [],[]
    for site in coord_linker_neighbour:
        _pos_diff_ = site.frac_coords - coord_linker.frac_coords

        N_O.append(PosDiffAdjust(_pos_diff_))
        N_O_dist.append(site.distance(coord_linker))


    if len(N_O_dist) == 1:
        N_O.append(N_O[0]-2*(N_O[0] - np.dot(M_O,N_O[0])/np.dot(M_O,M_O)*M_O ))
        N_O_dist.append(N_O_dist[0])


    O_coords = metal.frac_coords + M_O
    O_site = copy.deepcopy(metal)
    O_site.species, O_site.frac_coords = 'O', O_coords

    O_H_bond_length =  0.97856
    H_coords1 = coord_linker.frac_coords + O_H_bond_length/N_O_dist[0]*N_O[0]
    H_sites1 = copy.deepcopy(metal)
    H_sites1.species, H_sites1.frac_coords = 'H', H_coords1

    H_coords2 = coord_linker.frac_coords + O_H_bond_length/N_O_dist[1]*N_O[1]
    H_sites2 = copy.deepcopy(metal)
    H_sites2.species, H_sites2.frac_coords = 'H', H_coords2

    return [O_site, H_sites1, H_sites2]
    # rescale the bond length : currently based on Zn-O


def addHOHOH(coord_linker, coord_linker_neighbour, metals):

    site_O1 = coord_linker[0]
    site_O2 = coord_linker[1]

    O_M1 = PosDiffAdjust(coord_linker[0].frac_coords - metals[0].frac_coords)
    O_M2 = PosDiffAdjust(coord_linker[1].frac_coords - metals[1].frac_coords)
    site_O1.frac_coords = metals[0].frac_coords + O_M1
    site_O2.frac_coords = metals[1].frac_coords + O_M2
    _dumb_site_= copy.deepcopy(site_O1)
    _dumb_site_.frac_coords = O_M1 + O_M2
    
    M_mid_v = PosDiffAdjust(metals[0].frac_coords - metals[1].frac_coords)/2

    length_M_H_mid = 2.638178855934525
    H_mid = copy.deepcopy(site_O1)
    H_mid.species, H_mid.frac_coords = 'H', metals[1].frac_coords+ M_mid_v + length_M_H_mid/np.sqrt(sum(_dumb_site_.coords**2))*(O_M1 + O_M2)

    OH_length = 0.97
    H_left = copy.deepcopy(site_O1)
    vector = PosDiffAdjust(H_mid.frac_coords - site_O1.frac_coords)
    norm_vector = site_O1.distance(H_mid)
    H_left.species, H_left.frac_coords = 'H', site_O2.frac_coords  + OH_length/norm_vector*vector

    H_right = copy.deepcopy(site_O2)
    vector = PosDiffAdjust(H_mid.frac_coords - site_O2.frac_coords)
    norm_vector = site_O2.distance(H_mid)
    H_right.species, H_right.frac_coords = 'H', site_O1.frac_coords  + OH_length/norm_vector*vector

    return [site_O1, site_O2, H_mid, H_left, H_right]
    # rescale the bond length : currently based on Zn-O

def addX():

    return

def DebugVisualization(vis_structure):
        vis = StructureVis()
        vis.set_structure(vis_structure)
        vis.show()


def nodes_expansion(structure):
    
    newsites = []

    for site in structure.sites:
        permutationList = [[],[],[]]
        for i,pos in enumerate(site.frac_coords):
            if pos > 0.5:
                permutationList[i].extend([pos])
                permutationList[i].extend([pos-1])
            else:
                permutationList[i].extend([pos])
                permutationList[i].extend([pos+1])
        for x in permutationList[0]:
            for y in permutationList[1]:
                for z in permutationList[2]:
                    frac_coords = [x,y,z]
                    new_site = copy.deepcopy(structure[0])
                    new_site.frac_coords = frac_coords
                    newsites.append(new_site)

    new_structure = mgStructure.Structure.from_sites(newsites)

    return new_structure

def PosDiffAdjust(pos):
    for i,x in enumerate(pos):
        if x > 0.5:
            pos[i] = x-1
        if x < -0.5:
            pos[i] = x+1

    return pos

def merge_sites(structure, tol: float = 0.01, mode = 'delete') -> None:
    """
    Merges sites (adding occupancies) within tol of each other.
    Removes site properties.
    Args:
        tol (float): Tolerance for distance to merge sites.
        mode ('sum' | 'delete' | 'average'): "delete" means duplicate sites are
            deleted. "sum" means the occupancies are summed for the sites.
            "average" means that the site is deleted but the properties are averaged
            Only first letter is considered.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    d = structure.distance_matrix
    np.fill_diagonal(d, 0)
    clusters = fcluster(linkage(squareform((d + d.T) / 2)), tol, "distance")
    sites = []
    for c in np.unique(clusters):
        inds = np.where(clusters == c)[0]
        species = structure[inds[0]].species
        coords = structure[inds[0]].frac_coords
        props = structure[inds[0]].properties
        for n, i in enumerate(inds[1:]):
            if n > 0:
                break
            sp = structure[i].species
            if mode.lower()[0] == "s":
                species += sp
            offset = structure[i].frac_coords - coords
            coords = coords + ((offset - np.round(offset)) / (n + 2)).astype(coords.dtype)
            for key in props:
                if key =='pbc_custom':
                    continue
                if props[key] is not None and structure[i].properties[key] != props[key]:
                    if mode.lower()[0] == "a" and isinstance(props[key], float):
                        # update a running total
                        props[key] = props[key] * (n + 1) / (n + 2) + structure[i].properties[key] / (n + 2)
                    else:
                        props[key] = None
                        warnings.warn(
                            f"Sites with different site property {key} are merged. So property is set to none"
                        )
        sites.append(PeriodicSite(species, coords, structure.lattice, properties=props))

    structure._sites = sites


    return structure
