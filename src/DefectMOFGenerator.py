#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:06:27 2022

@author: Zhenzi Yu
"""

import os,sys
import numpy as np
import copy
import time 
import json
import shutil

from pymatgen.io.cif import CifParser
from mofid.run_mofid import cif2mofid
import pymatgen.analysis.graphs as mgGraph
import pymatgen.core.bonds  as mgBond
import pymatgen.core.structure as mgStructure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.xyz import XYZ

from src.helper import StructureAnalysis, WriteStructure, DebugVisualization
from src.DefectMOFStructure import DefectMOFStructure

from ase.io import read, write

CHARGE_APP_PATH = 'python /home/zhenzi/src/xyz2mol/'

class DefectMOFStructureBuilder():
   
    def __init__(self, cifFile_Name, input_dir = '.', output_dir = '.', sepMode = 'MetalOxo', cutoff = 12.8 ):
        
        self.cifFile_Name = cifFile_Name
        self.cifFile = os.path.join(input_dir,self.cifFile_Name)
        self.output_dir = output_dir + cifFile_Name.split('.')[0]
        
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.mkdir(self.output_dir)

        self.sepMode = sepMode
        self.cutoff = cutoff

        self.original_Structure = self.ReadStructure(self.cifFile )
        self.SeperateStructure(linkerSepDir = os.path.join(self.output_dir,'linkerSep' ))
        
        return
    
    def ReadStructure(self,cifFile_Name):
        
        cifFile = CifParser(cifFile_Name)
        # bug fixed, cannot allow primitive reduction!
        structure = cifFile.get_structures(primitive=False)[0]
        
        return structure

    def SeperateStructure(self, linkerSepDir):
 
        # use the packge to seperate the linker/nodes and read in
        t0 = time.time()

        _ = cif2mofid(self.cifFile ,output_path = linkerSepDir)
        self.original_linkers = self.ReadStructure(os.path.join(linkerSepDir, self.sepMode, 'linkers.cif'))
        self.original_nodes = self.ReadStructure(os.path.join(linkerSepDir, self.sepMode, 'nodes.cif'))
        print("========== Now working on %s ==========" %(self.cifFile_Name))
        print("1. SBU seperation finished, which takes %f Seconds" % (time.time()-t0))

        self.original_linkers,self.original_nodes,formulas = self.MOFStructurePropertyAssginment(self.original_linkers,self.original_nodes)

        # print and summary
        # TODO: add some check function to make sure the right structure is read into the code 

        print("2. MOF Analysis: There are %d atoms in Metal Clusters, and %d atoms in linkers, with %d linkers of %d types" \
        % ( len(self.original_nodes), len(self.original_linkers), self.linker_num, len(np.unique(formulas))))            

        return None

    def Concat_linkers_nodes(self,original_linkers,original_nodes):
            # Step 2 : add metal site into linker site, combined as a whole 
            processed_structure_sites = []
            for site in original_linkers:
                processed_structure_sites.append(site)       
            for site in original_nodes:
                processed_structure_sites.append(site)
            processed_structure = mgStructure.Structure.from_sites(processed_structure_sites)
            
            return processed_structure

    def MOFStructurePropertyAssginment(self, original_linkers,original_nodes):
        
        # Subgraph detection : linker_cluster_assignment - LCA
        cluster_assignment_linker, pbc_linker, coord_bond_array, cluster_assignment_metal, coord_bond_list, pbc_metal \
        = StructureAnalysis(original_linkers,original_nodes)

        # Neighbour Metal Cluster detection : NbM
        NbM = [x + len(original_linkers) for x in coord_bond_array]
        
        # pass Subgraph detection to the mg structure; 
        # NOTE: this could be combined together with the prvious step, but just for debug purposes, I seperated
        original_linkers.add_site_property('NbM',NbM)
        original_linkers.add_site_property('LCA',cluster_assignment_linker[1])
        original_linkers.add_site_property('pbc_custom',pbc_linker)
        original_linkers.add_site_property('NbL',[np.nan]*len(original_linkers))
        original_linkers.add_site_property('MCA',[np.nan]*len(original_linkers))
        self.linker_num = cluster_assignment_linker[0]

        # add property to nodes   
        original_nodes.add_site_property('NbM',[np.nan]*len(original_nodes)) # add np.nan 
        original_nodes.add_site_property('LCA',[np.nan]*len(original_nodes)) # add np.nan 
        original_nodes.add_site_property('pbc_custom',pbc_metal)
        original_nodes.add_site_property('NbL',coord_bond_list) # NbL: neighbourhodd linker
        original_nodes.add_site_property('MCA',cluster_assignment_metal[1]) # MCA: Metal Cluster assginement 
        original_nodes.add_site_property('is_linker',[False]*len(original_nodes))

        # create seperate molecules based on graph 
        self.molecules = []
        indexes = []
        # detect linker type 
        for i in range(self.linker_num):
            index = np.array( np.where(cluster_assignment_linker[1]==i)[0], dtype=int)
            indexes.append(index)
            sites = [ original_linkers[i] for i in index ]
            molecule = mgStructure.Structure.from_sites(sites)
            self.molecules.append(molecule)
        _formulas_ = [ x.formula for x in self.molecules]
        formulas = []
        for formula in _formulas_:
            if len(formula) > 3:
                formulas.append(formula)
        self.linker_type = np.unique(formulas) 
        
        # construct new MOF structure: add lable distinguishing Metal/linker, add label for linker type, 
        # Step 0: add unique id for sites. In the late process, sites might be removed, so need to defined an id besides index.
        id_linkers = np.arange(len(original_linkers))
        original_linkers.add_site_property('fixed_id',id_linkers)
        id_nodes = np.arange(len(original_nodes))+len(original_linkers)
        original_nodes.add_site_property('fixed_id',id_nodes)


        # Step 1: label linker type 
        linker_label = np.full(len(original_linkers),None) 
        is_linker = np.full(len(original_linkers),True)

        for jj,molecule in enumerate(self.molecules):
            if len(molecule.formula) <=3:
                self.linker_num -= 1
                continue
            linker_type_num = np.where(self.linker_type == molecule.formula)[0]
            linker_label[indexes[jj]] = linker_type_num
        original_linkers.add_site_property('linker_label',linker_label)
        original_nodes.add_site_property('linker_label',[np.nan]*len(original_nodes))
        original_linkers.add_site_property('is_linker',is_linker)


        return original_linkers, original_nodes, formulas


    def _DefectDensityControl_(self):
        
        cell = self.original_Structure.lattice.abc
        upper_lim = 40
        achived_conc = {1/self.linker_num:(1,[1,1,1])}
        max_supercell = [1,1,1]
        for i,dim in enumerate(cell):
            max_supercell[i] = max(1,int(np.floor(upper_lim/dim)))

        for i in range(1,max_supercell[0]+1):
            for j in range(1,max_supercell[1]+1):
                for k in range(1,max_supercell[2]+1):
                    num_of_linker = 1
                    conc = num_of_linker/(self.linker_num*i*j*k)
                    
                    while conc < 0.4:
                        addFlag = True
                        dict_keys = list(achived_conc.keys())
                        for exist_conc in dict_keys:
                            if conc > 0.01 and (exist_conc-conc)/conc < 0.3:
                                exisitng_v = achived_conc[exist_conc][1][0]*achived_conc[exist_conc][1][1]*achived_conc[exist_conc][1][2]
                                new_v =  i*j*k
                                if new_v < 0.6*exisitng_v:
                                    del achived_conc[exist_conc]
                                elif new_v >= exisitng_v:
                                    addFlag = False
                        if addFlag:
                            achived_conc[conc] = (num_of_linker,[i,j,k])

                        num_of_linker += 1
                        conc = num_of_linker/(self.linker_num*i*j*k)
                                    
        return achived_conc
    
    def StructureGeneration(self, superCell, defectConc, numofdefect, linker_type):
        
        working_linkers, working_nodes= copy.deepcopy(self.original_linkers), copy.deepcopy(self.original_nodes)
        working_mof = self.Concat_linkers_nodes(working_linkers, working_nodes)
        
        with open('src/charge_dict.json') as json_file:
            charge_dict = json.load(json_file)

        if self.linker_type[linker_type] in charge_dict.keys():
            charge_comp = charge_dict[self.linker_type[linker_type]]
        else:
            success = False
            for molecule in self.molecules:
                if success:
                    break
                if molecule.formula == self.linker_type[linker_type]:
                    os.system('rm temp.xyz')
                    XYZ(molecule).write_file('temp.xyz')
                    O_count = [ xx for xx in molecule.species if xx.name == 'O']
                    if len(O_count)!= 0:
                        re = os.popen( CHARGE_APP_PATH+'xyz2mol.py temp.xyz --ignore-chiral --charge -'+str(int(len(O_count))/2)).read()
                        if len(re) > 0:
                            charge_comp = -trial_charge
                            success = True
                            break
                        else:
                            vis_structure = molecule
                            DebugVisualization(vis_structure)
                            charge_comp = input("unseen molecular:%s, please specify charge\n"%(self.linker_type[linker_type] ))
                            success = True
                            break
                    for trial_charge in range(0,6):
                        re = os.popen( CHARGE_APP_PATH+'xyz2mol.py temp.xyz --ignore-chiral --charge -'+str(trial_charge)).read()
                        if len(re) > 0:
                            charge_comp = -trial_charge
                            success = True
                            break
            if not success:
                vis_structure = molecule
                DebugVisualization(vis_structure)
                charge_comp = input("unseen molecular:%s, please specify charge\n"%(self.linker_type[linker_type] ))
            try:
                charge_comp = int(charge_comp)
            except:
                raise("please input num")
            charge_dict[self.linker_type[linker_type]] = charge_comp

        with open('src/charge_dict.json', 'w') as fp:
            json.dump(charge_dict, fp)

        defect_structure = DefectMOFStructure(working_mof, defectConc, numofdefect, linker_type, self.output_dir, superCell, charge_comp)
        defect_structure.Build_supercell()
        defect_structure.DefectGen()
        # defect_structure.ReverseMonteCarlo()

        return
    
    def LinkerVacancy(self):
        
        self.possible_Defect_Density = self._DefectDensityControl_()
        for i_linker,linker_type in enumerate(self.linker_type):
            # skip H only and metal only
            if len(linker_type)<=2:
                continue
            for key,val in self.possible_Defect_Density.items():
                print("3. Generate linker vacancy defect structure with %s, at conc. of %.3f" %(linker_type,key))
                self.StructureGeneration(val[1],key,val[0], i_linker)
        
        return
    

