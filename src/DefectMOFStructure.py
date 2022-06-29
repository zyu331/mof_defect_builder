#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 21:39:33 2022

@author: jace
"""

import numpy as np
import time 
import copy 
import random

from pymatgen.core.lattice import Lattice
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.core.sites import PeriodicSite, Site
import pymatgen.core.structure as mgStructure
from pymatgen.util.coord import all_distances

from helper import CheckConnectivity, TreeSearch, WarrenCowleyParameter, SwapNeighborList, WriteStructure,substitute_funcGroup
from cappingAgent import water, water2, dummy, h2ooh, oh
class DefectMOFStructure():
    
    def __init__(self, linkers, nodes, superCell, defectConc):
        
        self.original_linkers = linkers
        self.linkers = []
        self.original_nodes = nodes
        self.nodes = []
        
        self.coord_indexes_linker = {}
        self.coord_indexes_MetalCluster = {}
        self.superCell = superCell
        self.defectConc = defectConc

    def __mul__(self, structure, scaling_matrix, structure_type):
        """
        Makes a supercell. Allowing to have sites outside the unit cell

        Args:
            scaling_matrix: A scaling matrix for transforming the lattice
                vectors. Has to be all integers. Several options are possible:

                a. A full 3x3 scaling matrix defining the linear combination
                   the old lattice vectors. E.g., [[2,1,0],[0,3,0],[0,0,
                   1]] generates a new structure with lattice vectors a' =
                   2a + b, b' = 3b, c' = c where a, b, and c are the lattice
                   vectors of the original structure.
                b. An sequence of three scaling factors. E.g., [2, 1, 1]
                   specifies that the supercell should have dimensions 2a x b x
                   c.
                c. A number, which simply scales all lattice vectors by the
                   same factor.

        Returns:
            Supercell structure. Note that a Structure is always returned,
            even if the input structure is a subclass of Structure. This is
            to avoid different arguments signatures from causing problems. If
            you prefer a subclass to return its own type, you need to override
            this method in the subclass.
        """
        scale_matrix = np.array(scaling_matrix, np.int16)
        if scale_matrix.shape != (3, 3):
            scale_matrix = np.array(scale_matrix * np.eye(3), np.int16)
        new_lattice = Lattice(np.dot(scale_matrix, structure._lattice.matrix))

        f_lat = lattice_points_in_supercell(scale_matrix)
        c_lat = new_lattice.get_cartesian_coords(f_lat)

        new_sep_moleculars = []
        for v in c_lat:
            _new_sites_seperate_molecular = []
            for site in structure:
                s = PeriodicSite(
                    site.species,
                    site.coords + v,
                    new_lattice,
                    properties=site.properties,
                    coords_are_cartesian=True,
                    to_unit_cell=False,
                    skip_checks=True,
                )
                _new_sites_seperate_molecular.append(s)
            new_sep_moleculars.append( mgStructure.Structure.from_sites(_new_sites_seperate_molecular) )
        #new_charge = structure._charge * np.linalg.det(scale_matrix) if structure._charge else None
        
        
        return new_sep_moleculars




    def make_supercell(self, molecular, scaling_matrix, molecular_type, to_unit_cell: bool = True):
        """
        Create a supercell.

        Args:
            scaling_matrix: A scaling matrix for transforming the lattice
                vectors. Has to be all integers. Several options are possible:

                a. A full 3x3 scaling matrix defining the linear combination
                   the old lattice vectors. E.g., [[2,1,0],[0,3,0],[0,0,
                   1]] generates a new structure with lattice vectors a' =
                   2a + b, b' = 3b, c' = c where a, b, and c are the lattice
                   vectors of the original structure.
                b. An sequence of three scaling factors. E.g., [2, 1, 1]
                   specifies that the supercell should have dimensions 2a x b x
                   c.
                c. A number, which simply scales all lattice vectors by the
                   same factor.
            to_unit_cell: Whether or not to fall back sites into the unit cell
        """
        s_sep = self.__mul__(molecular , scaling_matrix, molecular_type)
        
        # if to_unit_cell:
        #     for site in s:
        #         site.to_unit_cell(in_place=True)
        # structure._sites, s_sep._sites = s.sites, s_sep.sites
        # structure._lattice, s_sep._lattice = s.lattice, s_sep.lattice
        
        return s_sep

    def Coord_bond_Analysis(self):
        t0 = time.time()
        for i,linker in enumerate(self.linkers):
            _coord_dict_ = []
            _coord_index_ = []
            for j, node_cluster in enumerate(self.node_clusters):
            # for metal, linker, coord mode, put metal in the second 
                linker_coord_array, _metal_dict_ = CheckConnectivity(linker,node_cluster, mode='coord') 
                if len(_metal_dict_) > 0:
                    _coord_index_.append(j)
                    _coord_dict_.extend(list(np.where((linker_coord_array==True).any(axis=1)==True)[0]))
            self.coord_indexes_linker[i] = _coord_dict_
            self.coord_indexes_MetalCluster[i] = _coord_index_
        
        self.linker_indexes_MetalCluster = {}
        key_list = list(self.coord_indexes_MetalCluster.keys())
        val_list = list(self.coord_indexes_MetalCluster.values())
        
        for i in range(0,len(self.node_clusters)):
            pos = [index for index,x in enumerate(val_list) if i in x ]
            self.linker_indexes_MetalCluster[i] = pos
         
        print("Finish Coordination env analysis, which took %f Seconds" % (time.time()-t0))

        
    def Sub_with_Capping_agent(self,neighbor_list):
        
        def substitue(structure, coord_indexes_linker):
            """
            cluster coord_atoms
            """
            visited, pair, delList, addList = [], [], [], []
            newMolecuar = None
            for _index1_ in coord_indexes_linker:
                if _index1_ not in visited:
                    _pair_=[_index1_]
                    visited.append(_index1_)
                    for _index2_ in coord_indexes_linker:
                        if _index1_ != _index2_:
                            dist = structure.sites[_index1_].distance(structure.sites[_index2_])
                            if dist < 3 and structure.sites[_index1_].species_string == 'O' : 
                                _pair_.append(_index2_)
                                visited.append(_index2_)
                                pair.append(_pair_)   
                    if len(_pair_) ==1 and structure.sites[_index1_].species_string == 'N':
                        pair.append(_pair_) 
            if not pair:
                raise "error"
            
            for _pair_ in pair:
                _structure_ = copy.deepcopy(structure)
                if len(_pair_) == 1:
                    site = structure.sites[_pair_[0]]
                    if site.species_string == 'N':
                        if newMolecuar:
                            newMolecuar = substitute_funcGroup( 'N', _structure_, _pair_, oldMolecular = newMolecuar )
                        else:
                            newMolecuar = substitute_funcGroup( 'N', _structure_, _pair_, oldMolecular = newMolecuar )
                    if site.species_string == 'O':
                        delList, addList = substitute_funcGroup( 'O', structure, _pair_)
                        
                if len(_pair_) == 2: 
                    if newMolecuar:
                        newMolecuar = substitute_funcGroup( 'OO', _structure_, _pair_, oldMolecular = newMolecuar )
                    else:
                        newMolecuar = substitute_funcGroup( 'OO', _structure_, _pair_, oldMolecular = newMolecuar) 
                
            return newMolecuar
        
        all_components = [self.nodes]
        for _index_, _type_, _neighbor_list_ in neighbor_list:
            if _type_ == 'defect':
                _current_structure_T = copy.deepcopy(self.linkers[_index_])
                _current_structure_ = substitue( _current_structure_T, self.coord_indexes_linker[_index_])
            else:
                _current_structure_ = copy.deepcopy(self.linkers[_index_])
            all_components.append(_current_structure_)
            
        for i,component in enumerate(all_components):
            if i == 0:
                output = copy.deepcopy(component)
            else:
                if component == None :
                    print("capping error")
                else:
                    for site in component.sites:
                        # _nonpbc_site_ = Site(site.species,site.coords)
                        output.sites.append(site)
                    
        WriteStructure('.',output)
        
        return
        
    def Build_supercell_component(self):
        
        t0 = time.time()
        
        """
        linker
        """
        _linkers_ = self.make_supercell(copy.deepcopy(self.original_linkers), self.superCell, self.original_linkers.formula)
        for i,x in enumerate(_linkers_):
            if i ==0:
                linkers_all = x
            else:
                for site in x:
                    linkers_all.sites.append(site)
            
        # cluster metal nodes 
            
        visited = []
        self.linkers, self.moleculars_indexes = [],[]
        
        linker_bond_array, _ = CheckConnectivity(linkers_all,linkers_all)        
        while len(visited) < len(linkers_all):
            for x in range(len(linkers_all)):
                if x not in visited:
                    visited, _molecular_index_ = TreeSearch(x, visited, linker_bond_array)
                    self.moleculars_indexes.append(_molecular_index_)

        for i,molecular_index in enumerate(self.moleculars_indexes):
            _sites_ = [linkers_all[index] for index in molecular_index]
            if len(molecular_index) <=2:
                continue
            else:       
                molecular = mgStructure.Structure.from_sites(_sites_)
                self.linkers.append(molecular)
               
        """
        metal cluster
        """                                   
        _node_clusters_ = self.make_supercell(copy.deepcopy(self.original_nodes), self.superCell, self.original_nodes.formula)
        for i,x in enumerate(_node_clusters_):
            if i ==0:
                node_clusters_all = x
            else:
                for site in x:
                    node_clusters_all.sites.append(site)
        self.nodes = node_clusters_all 
        # cluster metal nodes 
        len_nodes = len(node_clusters_all)
        node_dist = np.zeros([len_nodes,len_nodes])
        for i in range(len_nodes):
            for j in range(len_nodes):
                node_dist[i,j] = node_clusters_all.sites[i].distance(node_clusters_all.sites[j])
                
        visited = []
        self.node_clusters = []
        
        for i,x in enumerate(node_clusters_all):
            if i not in visited:
                _sites_ = [x]
                _working_ = [i]
                visited.append(i)
                while _working_:
                    for j, y in enumerate(node_clusters_all):
                        if ( node_dist[ _working_[0],j] < 3.5) and (j not in visited):
                            _sites_.append(y)
                            visited.append(j)
                            _working_.append(j)
                    _working_.remove(_working_[0])
                        
            _node_cluster_ = mgStructure.Structure.from_sites(_sites_)
            _dup_ = False
            for x in self.node_clusters:
                if x == _node_cluster_:
                    _dup_ = True
            if _dup_ == False:
                self.node_clusters.append(_node_cluster_)
               
                
        print("Finish super cell build, which took %f Seconds" % (time.time()-t0))
        
    def DefectGen(self):
        superCell_count = self.superCell[0]*self.superCell[1]*self.superCell[2]
        num_of_delete_linkers = round(self.defectConc*superCell_count)
        if num_of_delete_linkers==0:
            num_of_delete_linkers=1
        rand_linker = random.sample(range(0,len(self.linkers)),num_of_delete_linkers)
        self.defectConc = len(rand_linker)/len(self.linkers)
        
        self.neighbor_list = []
        for i,x in enumerate(self.linkers):
            _neighbor_linkers_list_ = []
            _neighbor_nodes_list_ = self.coord_indexes_MetalCluster[i]
            for _node_index_ in _neighbor_nodes_list_:
                _neighbor_linkers_list_.extend( self.linker_indexes_MetalCluster[_node_index_] )
            _neighbor_linkers_list_.remove(i)
            _neighbor_linkers_list_ = list(np.unique(_neighbor_linkers_list_))
            if i in rand_linker:
                _ele_ = [i,'defect', _neighbor_linkers_list_]
            else:
                _ele_ = [i,'normal', _neighbor_linkers_list_]
            self.neighbor_list.append(_ele_)
        
        return
                
        
    def ReverseMonteCarlo(self, SRO = 1, beta = 1.0, N = 100 , MaxErr = 0.01):
        '''
        neighbor_list: 2-D list. [[1, 'Yb', ['Nd', 'Yb']]...]
        SRO: target alpha (short range order)
        N: steps number
        beta: smoothfactor for exp(beta|alpha-SRO|)
        '''
        #old_neighbor_list=copy.deepcopy(neighbor_list)
        t0 = time.time()
        neighbor_list = copy.deepcopy(self.neighbor_list)
        n_metal = len(neighbor_list)
        alpha_Yb_0 = WarrenCowleyParameter(neighbor_list, 'normal', 'defect')
        metric_Yb_0 = abs(alpha_Yb_0-SRO)
        metric_Yb_history = []
        n = 0
        if SRO == None:
            SRO = np.inf
            outputAll = True
            
        while n <= N:# and metric_Yb_0 >= MaxErr:
            n += 1
            ID_1, ID_2 = random.sample(range(0, n_metal), 2)
            if neighbor_list[ID_1][1] != neighbor_list[ID_2][1]:
                new_neighbor_list = SwapNeighborList(neighbor_list, ID_1, ID_2)
                alpha_Yb = WarrenCowleyParameter(new_neighbor_list, 'normal', 'defect')
                metric_Yb = abs(alpha_Yb-SRO)
                if metric_Yb < metric_Yb_0:
                    neighbor_list = new_neighbor_list
                    alpha_Yb_0 = alpha_Yb
                    metric_Yb_0 = metric_Yb
                    if metric_Yb_0 <= MaxErr:
                        print(n, alpha_Yb)
                        break
                elif metric_Yb > metric_Yb_0 and np.exp(-beta*(metric_Yb-metric_Yb_0)) >= random.uniform(0, 1):
                    neighbor_list = new_neighbor_list
                    alpha_Yb_0 = alpha_Yb
                    metric_Yb_0 = metric_Yb
                else:
                    new_neighbor_list = neighbor_list
                    
                metric_Yb_history.append(metric_Yb)
                self.Sub_with_Capping_agent(neighbor_list)
                print(metric_Yb)
                
            
        print("Finish Coordination env analysis, which took %f Seconds" % (time.time()-t0))
            
        return neighbor_list  
        
