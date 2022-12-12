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
from sklearn.cluster import DBSCAN

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

    len1,len2 = len(linker),len(metal)
    bond_array = np.full((len1,len1), 0, dtype=int)
                   
    for i in range(len1):
        for j in range(i+1,len1):
            try:
                isbond = mgBond.CovalentBond.is_bonded(linker[i],linker[j])
            except:
                isbond = False
            if isbond:
                bond_array[i,j] = 1
                bond_array[j,i] = 1          
            else:
                bond_array[i,j] = 0
    cluster_assignment_linker = connected_components(bond_array)   

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
    
    for i in range(cluster_assignment_linker[0]):
        linker_coords = []
        indexes = np.where(cluster_assignment_linker[1]==i)[0]
        for index in indexes:
            # append index to the end of the frac_coords
            linker_coords.append( np.append(linker[index].frac_coords,int(index)))
        linker_coords = np.array(linker_coords)
        for axis in range(3):
            linker_coords = linker_coords[linker_coords[:,axis].argsort()]
            cluster = DBSCAN(eps =0.1).fit(linker_coords[:,axis].reshape(-1,1))
            if max(cluster.labels_) == 0:
                break
            elif max(cluster.labels_) == 1:
                cluster1 = np.array([linker_coords[_index_] for _index_,ele in enumerate(cluster.labels_) if ele == 0])
                cluster2 = np.array([linker_coords[_index_] for _index_,ele in enumerate(cluster.labels_) if ele == 1])
                if np.mean(cluster1[:,axis]) > np.mean(cluster2[:,axis]):
                    for ele in cluster1:
                        pbc_linker[int(ele[-1])] = 'large'
                    for ele in cluster2:
                        pbc_linker[int(ele[-1])] = 'small'
                else:
                    for ele in cluster1:
                        pbc_linker[int(ele[-1])] = 'small'
                    for ele in cluster2:
                        pbc_linker[int(ele[-1])] = 'large'                    
            else:
                print("pbc error please check linker pbc")

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
            cluster = DBSCAN(eps =0.1,min_samples=1).fit(metal_coords[:,axis].reshape(-1,1))
            if max(cluster.labels_) == 0:
                break
            elif max(cluster.labels_) == 1:
                cluster1 = np.array([metal_coords[_index_] for _index_,ele in enumerate(cluster.labels_) if ele == 0])
                cluster2 = np.array([metal_coords[_index_] for _index_,ele in enumerate(cluster.labels_) if ele == 1])
                if np.mean(cluster1[:,axis]) > np.mean(cluster2[:,axis]):
                    for ele in cluster1:
                        pbc_metal[int(ele[-1])] = 'large'
                    for ele in cluster2:
                        pbc_metal[int(ele[-1])] = 'small'
                else:
                    for ele in cluster1:
                        pbc_metal[int(ele[-1])] = 'small'
                    for ele in cluster2:
                        pbc_metal[int(ele[-1])] = 'large'                    
            else:
                print("pbc error please check metal pbc")

    return cluster_assignment_linker, pbc_linker, coord_bond_array, cluster_assignment_metal, coord_bond_list, pbc_metal

def NbMAna(mof, deleted_linker, _index_linker_):

    """
    goal of this helper function: 
        input a seperate linker(specified by _index_linker)
        return the metal-coordLinker-Linker1stneighour pair(three components!) 
    
    return varibles:
        metal_dict_sitebased: fixed_id:site
        MCA_dict_notuniqueid: metal_cluster_num: site
        metal_coord_dict_sitebased: fixed_id:[ (fixed_id, coord sites)]
        metal_coord_neighbour_dict_sitebased: fixed_id: [fixed_id, bonded_sites]
        dist_metal_cluster: dist array between metal clusters
        index_fixed_id_dict : dict from fixed_id to index

    """

    # initiate the return varibles
    metal_dict_sitebased = {}
    metal_coord_dict_sitebased = {}
    metal_coord_neighbour_dict_sitebased = {}
    # fetch the the current index (Compared to the fixed id )
    index_fixed_id_dict = {}
    for i, s in enumerate(mof):
        index_fixed_id_dict[s.fixed_id] = i

    # construct metal - coord dict first
    for index in _index_linker_:
        if ~np.isnan(deleted_linker[index].NbM):
            metal_fixed_id = deleted_linker[index].NbM
            if metal_fixed_id not in metal_dict_sitebased.keys():
                metal_dict_sitebased[metal_fixed_id] = mof[index_fixed_id_dict[metal_fixed_id]]
                metal_coord_dict_sitebased[metal_fixed_id] = [] 
                metal_coord_dict_sitebased[metal_fixed_id].append( (deleted_linker[index].fixed_id, deleted_linker[index]))
        else:
            continue
    
    # based on metal - coord dict, construct coord-neighbour dict 
    for item in metal_coord_dict_sitebased.items():
        coord_index, coord_atoms = item[0], item[1]
        for atom in coord_atoms:
            for _i in _index_linker_:
                try: 
                    isbond = mgBond.CovalentBond.is_bonded(atom[1], deleted_linker[_i])
                except:
                    isbond = False
                if isbond and deleted_linker[_i].fixed_id!=atom[0]:
                    if atom[0] not in metal_coord_neighbour_dict_sitebased.keys():
                        metal_coord_neighbour_dict_sitebased[atom[0]] = []
                        metal_coord_neighbour_dict_sitebased[atom[0]].append((_i,atom[1].distance(deleted_linker[_i])))
                    else:
                        metal_coord_neighbour_dict_sitebased[atom[0]].append((_i,atom[1].distance(deleted_linker[_i])))

    # a very naive way to calculate the min dist between two metal cluster:
    # TODO: could combined with code above 
    associated_NbM = [ deleted_linker[s].NbM for s in _index_linker_]
    coord_metal_index = np.array([i for i,s in enumerate(mof) if s.fixed_id in associated_NbM])
    MCA_dict_notuniqueid = {}
    for index in coord_metal_index:
        if mof[index].MCA not in MCA_dict_notuniqueid.keys():
            MCA_dict_notuniqueid[ mof[index].MCA] = []
            MCA_dict_notuniqueid[ mof[index].MCA].append( (mof[index].fixed_id, mof[index]))
        else:
            MCA_dict_notuniqueid[ mof[index].MCA].append( (mof[index].fixed_id, mof[index]))
    num_metal_cluster = len(MCA_dict_notuniqueid)
    dist_metal_cluster = np.ones((num_metal_cluster,num_metal_cluster))*100
    for ii,key1 in enumerate(MCA_dict_notuniqueid.keys()):
        for jj,key2 in enumerate(MCA_dict_notuniqueid.keys()):
            x, y = MCA_dict_notuniqueid[key1], MCA_dict_notuniqueid[key2]
            for _x in x:
                for _y in y:
                    _dist_ = _x[1].distance(_y[1])
                    if _dist_ < dist_metal_cluster [ii,jj] and ii!=jj:
                        dist_metal_cluster [ii,jj] = _dist_
    
    return metal_dict_sitebased, MCA_dict_notuniqueid, metal_coord_dict_sitebased, metal_coord_neighbour_dict_sitebased, dist_metal_cluster, index_fixed_id_dict

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

def addOH(metal_dict_sitebased, metal_coord_dict_sitebased,metal_coord_neighbour_dict_sitebased):

    M_O = metal_coord_dict_sitebased[1].frac_coords - metal_dict_sitebased.frac_coords
    N_O,N_O_dist = [],[]
    for site in metal_coord_neighbour_dict_sitebased:
        _pos_diff_ = site.frac_coords - metal_coord_dict_sitebased[1].frac_coords

        N_O.append(PosDiffAdjust(_pos_diff_))
        N_O_dist.append(site.distance(metal_coord_dict_sitebased[1]))

   
    O_coords = metal_dict_sitebased.frac_coords + M_O
    O_site = copy.deepcopy(metal_coord_dict_sitebased[1])
    O_site.species, O_site.frac_coords = 'O', O_coords

    O_H_bond_length =  0.97856
    H_coords1 = metal_coord_dict_sitebased[1].frac_coords + O_H_bond_length/N_O_dist[0]*N_O[0]
    H_sites1 = copy.deepcopy(metal_coord_neighbour_dict_sitebased[0])
    H_sites1.species, H_sites1.frac_coords = 'H', H_coords1

    return [O_site, H_sites1]


def addH2O(metal_dict_sitebased, metal_coord_dict_sitebased,metal_coord_neighbour_dict_sitebased):

    M_O = metal_coord_dict_sitebased[1].frac_coords - metal_dict_sitebased.frac_coords
    N_O,N_O_dist = [],[]
    for site in metal_coord_neighbour_dict_sitebased:
        _pos_diff_ = site.frac_coords - metal_coord_dict_sitebased[1].frac_coords

        N_O.append(PosDiffAdjust(_pos_diff_))
        N_O_dist.append(site.distance(metal_coord_dict_sitebased[1]))

    O_M_bond_length = 2.1365
    O_coords = metal_dict_sitebased.frac_coords + M_O
    O_site = copy.deepcopy(metal_coord_dict_sitebased[1])
    O_site.species, O_site.frac_coords = 'O', O_coords

    O_H_bond_length =  0.97856
    H_coords1 = metal_coord_dict_sitebased[1].frac_coords + O_H_bond_length/N_O_dist[0]*N_O[0]
    H_sites1 = copy.deepcopy(metal_coord_neighbour_dict_sitebased[0])
    H_sites1.species, H_sites1.frac_coords = 'H', H_coords1

    H_coords2 = metal_coord_dict_sitebased[1].frac_coords + O_H_bond_length/N_O_dist[1]*N_O[1]
    H_sites2 = copy.deepcopy(metal_coord_neighbour_dict_sitebased[1])
    H_sites2.species, H_sites2.frac_coords = 'H', H_coords2

    return [O_site, H_sites1, H_sites2]
    # rescale the bond length : currently based on Zn-O


def addHOHOH(metals, coord_atom):

    if len(coord_atom[0])!=1 or len(coord_atom[1])!=1:
        print("ERROR: for HOHOH, not unique coord O, nothing added")
        return []

    site_O1 = coord_atom[0][0]
    site_O2 = coord_atom[1][0]

    O_M1 = PosDiffAdjust(coord_atom[0][0][1].frac_coords - metals[0].frac_coords)
    O_M2 = PosDiffAdjust(coord_atom[1][0][1].frac_coords - metals[1].frac_coords)
    site_O1[1].frac_coords = metals[0].frac_coords + O_M1
    site_O2[1].frac_coords = metals[1].frac_coords + O_M2
    _dumb_site_= copy.deepcopy(site_O1)[1]
    _dumb_site_.frac_coords = O_M1 + O_M2
    
    M_mid_v = PosDiffAdjust(metals[0].frac_coords - metals[1].frac_coords)/2

    length_M_H_mid = 2.638178855934525
    H_mid = copy.deepcopy(site_O1)[1]
    H_mid.species, H_mid.frac_coords = 'H', metals[1].frac_coords+ M_mid_v + length_M_H_mid/np.sqrt(sum(_dumb_site_.coords**2))*(O_M1 + O_M2)

    OH_length = 0.97
    H_left = copy.deepcopy(site_O1)[1]
    vector = PosDiffAdjust(H_mid.frac_coords - site_O1[1].frac_coords)
    norm_vector = site_O1[1].distance(H_mid)
    H_left.species, H_left.frac_coords = 'H', site_O2[1].frac_coords  + 0.97/norm_vector*vector

    H_right = copy.deepcopy(site_O2)[1]
    vector = PosDiffAdjust(H_mid.frac_coords - site_O2[1].frac_coords)
    norm_vector = site_O2[1].distance(H_mid)
    H_right.species, H_right.frac_coords = 'H', site_O1[1].frac_coords  + 0.97/norm_vector*vector



    return [site_O1[1], site_O2[1], H_mid, H_left, H_right]
    # rescale the bond length : currently based on Zn-O


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