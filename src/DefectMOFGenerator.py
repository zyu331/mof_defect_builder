#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:06:27 2022

@author: Zhenzi Yu
"""

import os
import math
import numpy as np
import copy
import random
import time 

from pymatgen.io.cif import CifParser
from mofid.run_mofid import cif2mofid
import pymatgen.analysis.graphs as mgGraph
import pymatgen.core.bonds  as mgBond
import pymatgen.core.structure as mgStructure
from pymatgen.io.vasp.inputs import Poscar

from src.helper import CheckConnectivity, TreeSearch, WriteStructure
from src.DefectMOFStructure import DefectMOFStructure


class DefectMOFStructureBuilder():
    
    def __init__(self, cifFile_Name, output_dir = '' , sepMode = 'StandardIsolated', cutoff = 12.8 ):
        
        self.cifFile_Name = cifFile_Name
        self.output_dir = output_dir
        self.sepMode = sepMode
        self.cutoff = cutoff

        self.original_nodes = None
        self.original_linkers, self.original_linkers_length = None, None
        self.moleculars, self.moleculars_indexes, self.coord_indexes = [], [], [] 

        self.original_Structure = self.ReadStructure(self.cifFile_Name)
        self.SeperateStructure(self.cifFile_Name)
        
        return
    
    def ReadStructure(self,cifFile_Name):
        
        cifFile = CifParser(cifFile_Name)
        structure = cifFile.get_structures()[0]
        
        return structure
    
    def SeperateStructure(self, cifFileName, linkerSepDir = 'linkerSep' ):
        
        
        # use the packge to seperate the linker/nodes and read in
        t0 = time.time()
        cif2mofid(cifFileName ,output_path = os.path.join(self.output_dir,linkerSepDir))
        self.original_nodes = self.ReadStructure(os.path.join(linkerSepDir, self.sepMode, 'nodes.cif'))
        self.original_linkers = self.ReadStructure(os.path.join(linkerSepDir, self.sepMode, 'linkers.cif'))
        self.original_linkers_length = len(self.original_linkers.sites)
        print("SBU seperation finished, which takes %f Seconds" % (time.time()-t0))
        
        linker_bond_array, _ = CheckConnectivity(self.original_linkers,self.original_linkers)
    
        visited = []
        self.coord_dict = {}
        while len(visited) < self.original_linkers_length:
            for x in range(self.original_linkers_length):
                if x not in visited:
                    visited, _molecular_index_ = TreeSearch(x, visited, linker_bond_array)
                    self.moleculars_indexes.append(_molecular_index_)

        for i,molecular_index in enumerate(self.moleculars_indexes):
            _sites_ = [self.original_linkers.sites[index] for index in molecular_index]
            if len(molecular_index) <=2:
                for _site_ in _sites_:
                    self.original_nodes.sites.append(_site_)
            else:       
                molecular = mgStructure.Structure.from_sites(_sites_)
                self.moleculars.append(molecular)
            

        _linker_type_ = [ x.formula for x in self.moleculars]
        print("There are %d types, with a total %d linkers in the MOF Unit Cell" % ( len(np.unique(_linker_type_)), len(self.moleculars)))            

    
        return

    def _DefectDensityControl_(self):
        
        cell = self.original_Structure.lattice.abc
        min_index = cell.index(min(cell))
        max_index = cell.index(max(cell))
        _mid_index = [0,1,2]
        _mid_index.remove(min_index)
        _mid_index.remove(max_index)
        mid_index = _mid_index[0]
        
        def assign_supercell(cellnum):
            cell = [1,1,1]
            if cellnum == 1:
                return cell
            elif cellnum ==2 or cellnum==3 or cellnum==5:
                cell[min_index] = cellnum
            elif cellnum == 6:
                cell[min_index] = 2
                cell[mid_index] = 3
            elif cellnum == 4:
                cell[min_index] = 2
                cell[mid_index] = 2
            else:
                raise "error for super cell assignment"
            return cell

        enumerate_conc = [1/2,1/3,2/3,1/4,3/4,1/5,2/5,3/5,4/5,1/6,5/6,1,2,3,4,5,6]
        corr_defect_num = [1,1,2,1,3,1,2,3,4,1,5,1,2,3,4,5,6]
        corr_cell = [2,3,3,4,4,5,5,5,5,6,6,1,1,1,1,1,1]
        original_conc = 1/len(self.moleculars)
        desired_conc = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        achived_conc = {}
        
        for _conc_goal_ in desired_conc:
            for i,_conc_possible_ in enumerate(enumerate_conc):
                _conc_possible2_ = original_conc*_conc_possible_
                if abs((_conc_possible2_-_conc_goal_)/_conc_goal_) < 0.3:
                    achived_conc[_conc_possible2_] = [_conc_goal_, corr_defect_num[i], corr_cell[i], assign_supercell(corr_cell[i])]
                    break
        
        
        return achived_conc
    
    def StructureGeneration(self, superCell, defectConc, numofdefect):
        
        working_linkers = copy.deepcopy(self.original_linkers)
        working_nodes = copy.deepcopy(self.original_nodes)

        defect_structure = DefectMOFStructure(working_linkers, working_nodes, superCell, defectConc, numofdefect)
        defect_structure.Build_supercell_component()
        defect_structure.Coord_bond_Analysis()
        defect_structure.DefectGen()
        defect_structure.ReverseMonteCarlo()
        
        all_components = defect_structure.linkers.copy()
        all_components.extend(defect_structure.node_clusters)

        return
    
    def LinkerVacancy(self):
        
        self.possible_Defect_Density = self._DefectDensityControl_()
        for key,val in self.possible_Defect_Density.items():
       
            
            self.StructureGeneration(val[3],key,val[1])
        
        return
    
    def DebugVisualization():
        
        return
