#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 21:39:33 2022

@author: jace
"""

import numpy as np

from pymatgen.core.lattice import Lattice
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.core.sites import PeriodicSite, Site
import pymatgen.core.structure as mgStructure
from pymatgen.util.coord import all_distances

from helper import CheckConnectivity, TreeSearch

class DefectMOFStructure():
    
    def __init__(self, linkers, nodes, superCell):
        
        self.original_linkers = linkers
        self.linkers = []
        self.original_nodes = nodes
        self.nodes = []
        
        self.superCell = superCell

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
        for i,molecular_index in enumerate(self.moleculars_indexes):
            _sites_ = [self.original_linkers.sites[index] for index in molecular_index]
            molecular = mgStructure.Structure.from_sites(_sites_)
            self.moleculars.append(molecular)
            
            # for metal, linker, coord mode, put metal in the second 
            linker_coord_array, _metal_dict_ = CheckConnectivity(molecular,self.original_nodes, mode='coord') 
            self.coord_dict[i] = _metal_dict_
            _coord_index_ = np.where((linker_coord_array==True).any(axis=1)==True)[0]
            self.coord_indexes.append(_coord_index_)

    def Add_Capping_agent(self):
        
        return
        
    def Build_supercell_component(self):
        
        linker_type = []
        for linker in self.original_linkers:
            if linker.formula not in linker_type:
                linker_type.append(linker.formula)
            _linkers_ = self.make_supercell(linker, self.superCell, linker.formula)
            self.linkers.extend(_linkers_)

        # cluster metal nodes 
        node_dist, visited = all_distances(self.original_nodes.cart_coords,self.original_nodes.cart_coords), []
        self.original_node_clusters, self.node_clusters = [], []
        
        for i,x in enumerate(self.original_nodes):
            _sites_ = [x]
            if i not in visited:
                for j, y in enumerate(self.original_nodes):
                    if node_dist[i,j] < 5:
                        _sites_.append(y)
                        visited.append(j)
            _node_cluster_ = mgStructure.Structure.from_sites(_sites_)
            self.original_node_clusters.append(_node_cluster_)
                                    
        for node_cluster in self.original_node_clusters:
            node_clusters = self.make_supercell(node_cluster, self.superCell, node_cluster.formula)
            self.node_clusters.extend(node_clusters)
        
    def SRO(self):
        
        return
        
        
    def reverseMC(self):
        
        
        return
        
