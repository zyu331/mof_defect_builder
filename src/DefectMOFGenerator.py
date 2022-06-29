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
from pymatgen.io.cif import CifParser,CifWriter

from helper import CheckConnectivity, TreeSearch, WriteStructure
from DefectMOFStructure import DefectMOFStructure


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
        cif2mofid(cifFileName ,output_path = linkerSepDir)
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
        pbc_num = [ math.ceil( self.cutoff*2/x ) for x in cell ] 
        posible_conc = {}
        for i in range(pbc_num[0]):
            for j in range(pbc_num[1]):
                for k in range(pbc_num[2]):
                    posible_conc[str(i)+str(j)+str(k)] =  [1/((i+1)*(j+1)*(k+1))/len(self.moleculars)*x for x in [1,2,3]] 
        posible_conc
        
        return posible_conc
    
    def StructureGeneration(self, superCell, defectConc):
        
        working_linkers = copy.deepcopy(self.original_linkers)
        working_nodes = copy.deepcopy(self.original_nodes)

        defect_structure = DefectMOFStructure(working_linkers, working_nodes, superCell ,0.05)
        defect_structure.Build_supercell_component()
        defect_structure.Coord_bond_Analysis()
        defect_structure.DefectGen()
        defect_structure.ReverseMonteCarlo()
        
        all_components = defect_structure.linkers.copy()
        all_components.extend(defect_structure.node_clusters)

        return
    
    def LinkerVacancy(self):
        
        self.possible_Defect_Density = self._DefectDensityControl_()
        self.StructureGeneration([1,1,1],0.1)
        
        return
    
    def DebugVisualization():
        
        return

cifFile_Name = 'MOF-801.cif'
cifFile = CifParser(cifFile_Name)
os.system('mv '+cifFile_Name+' original.cif')
structure = cifFile.get_structures()[0]
out = CifWriter(structure)
out.write_file(cifFile_Name)
a = DefectMOFStructureBuilder(cifFile_Name)
a.LinkerVacancy()