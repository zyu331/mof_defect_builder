#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 21:39:33 2022

@author: jace
"""

from cmath import isnan
import numpy as np
import time 
import copy 
import random
import os
import shutil

from pymatgen.core.lattice import Lattice
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.core.sites import PeriodicSite, Site
import pymatgen.core.structure as mgStructure
from pymatgen.util.coord import all_distances

from src.helper import WarrenCowleyParameter, SwapNeighborList, addOH, addH2O, addHOHOH, addX, WriteStructure, DebugVisualization, merge_sites, NbMAna
from src.cappingAgent import water, water2, dummy, h2ooh, oh

import warnings
warnings.filterwarnings("ignore", message="Sites with different site property NbM are merged. So property is set to none")

class DefectMOFStructure():
    
    def __init__(self, mof, defectConc, numofdefect, linker_type, output_dir, superCell, charge_comp, capping_mod = 'OH'):
        
        self.original_mof = mof
        self.defectConc = defectConc
        self.numofdefect = numofdefect
        self.linker_type = linker_type
        self.charge_comp = charge_comp
        self.output_dir = os.path.join( output_dir , str(self.defectConc)[0:4])
        self.capping_mod = capping_mod
        if not os.path.isdir( self.output_dir):
            os.mkdir(self.output_dir)
        self.superCell = superCell

        self.ReverseMonteCarlo_flag = False

    def __mul__(self, structure, scaling_matrix):
        
        scale_matrix = np.array(scaling_matrix, np.int)
        if scale_matrix.shape != (3, 3):
            scale_matrix = np.array(scale_matrix * np.eye(3), np.int)
        new_lattice = Lattice(np.dot(scale_matrix, structure._lattice.matrix))
        new_moleculars = structure
        # f_lat = lattice_points_in_supercell(scale_matrix)
        # c_lat = new_lattice.get_cartesian_coords(f_lat)
        for axis,x in enumerate(scaling_matrix):
            original_atom_num = len(new_moleculars)
            LCA_count = np.nanmax([x.LCA for x in new_moleculars if x.LCA is not None])+1
            MCA_count = np.nanmax([x.MCA for x in new_moleculars if x.MCA is not None])+1
            scale = np.array([1.0,1.0,1.0])
            scale[axis] = 1/x
            count = 0
            _new_sites_seperate_molecular = []
            while count < x:
                vec = np.array([0.0,0.0,0.0])
                vec[axis]  = count * scale[axis]
                for _site_ in new_moleculars:
                    # bug fixed, dont touch/modify the original site! operate on a copy!
                    site = copy.deepcopy(_site_)
                    if site.NbM is np.nan:
                        site.properties['NbM'] = np.nan
                    else:
                        site.properties['NbM'] = site.NbM + count*original_atom_num 
                    if site.LCA is np.nan:
                        site.properties['LCA'] = np.nan
                    else:
                        if site.pbc_custom[axis] == 'small':
                            site.properties['LCA'] = site.LCA + count*LCA_count
                        elif site.pbc_custom[axis] == 'large' and count != x-1:
                            site.properties['LCA'] = site.LCA + (count+1)*LCA_count
                        elif site.pbc_custom[axis] == 'large' and count == x-1:
                            site.properties['LCA'] = site.LCA
                        else:
                            site.properties['LCA'] = site.LCA + count*LCA_count
                    if site.NbL is np.nan:
                        site.properties['NbL'] = np.nan
                    else:
                        site.properties['NbL'] = [ xx + count*original_atom_num for xx in site.NbL]
                    if site.MCA is np.nan:
                        site.properties['MCA'] = np.nan
                    else:
                        if site.pbc_custom[axis] == 'small':
                            site.properties['MCA'] = site.MCA + count*MCA_count
                        elif site.pbc_custom[axis] == 'large' and count != x-1:
                            site.properties['MCA'] = site.MCA + (count+1)*MCA_count
                        elif site.pbc_custom[axis] == 'large' and count == x-1:
                            site.properties['MCA'] = site.MCA
                        else:
                            site.properties['MCA'] = site.MCA + count*MCA_count
                    site.properties['fixed_id'] = site.fixed_id + count*original_atom_num 
                    s = PeriodicSite(
                        site.species,
                        np.multiply(site.frac_coords,scale) + np.array(vec),
                        new_lattice,
                        properties=site.properties,
                        coords_are_cartesian=False,
                        to_unit_cell=False,
                        skip_checks=True,
                    )

                    _new_sites_seperate_molecular.append(s)
                count += 1
            new_moleculars = mgStructure.Structure.from_sites(_new_sites_seperate_molecular)
        # TODO: the charge current is not read and pass, need to be fixed in the future (should be an easy fix)
        #new_charge = structure._charge * np.linalg.det(scale_matrix) if structure._charge else None 
        return new_moleculars

    def make_supercell(self, molecular, scaling_matrix, to_unit_cell: bool = True):
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
        s_sep = self.__mul__(molecular , scaling_matrix)
        
        return s_sep
        
    def Sub_with_Capping_agent(self,mof, deleted_linker):
            
        # try to cap OMS with something
############################################################################################
#       Important notice: since the linkers have been delete please use fixed_id!!!!  
# ##########################################################################################        
        self.output_sites = []
        # Preperation 1: redo the subgraph for deleted_linker 
        # TODO: could improve but it should be fine 
        LCA_dict_notuniqueid = {}
        for index,site in enumerate(deleted_linker):
            if site.LCA not in LCA_dict_notuniqueid.keys():
                LCA_dict_notuniqueid[ site.LCA ] = []
                LCA_dict_notuniqueid[ site.LCA ].append(index)
            else:
                LCA_dict_notuniqueid[ site.LCA ].append(index)        

        # Preperation 2: check the charge 
        if self.charge_comp != 0:
            print("Note: The linker has %i net charge"%self.charge_comp)
        else:
            print("Note: The linker has ZERO net charge")

        # add capping agent
        
        for LCA in LCA_dict_notuniqueid.keys():
            added_charge = 0
            _index_linker_ = np.array(LCA_dict_notuniqueid[LCA])

            linker_coord_list = NbMAna(mof, deleted_linker, _index_linker_)
            # put OO first 
            linker_coord_list.sort(reverse=True)
            for (type, coord_linker, coord_linker_neighbour, metal) in linker_coord_list:
                output_sites_T = []
                if added_charge < abs(self.charge_comp):
                    if type == 'OO' and self.capping_mod == 'OH':
                        output_sites_T.extend(addHOHOH(coord_linker, coord_linker_neighbour, metal))
                        added_charge += 1
                    elif self.capping_mod == 'OH':
                        output_sites_T.extend(addOH(coord_linker, coord_linker_neighbour, metal))
                        added_charge += 1
                    else:
                        output_sites_T.extend(addX(coord_linker, coord_linker_neighbour, metal))
                        added_charge += 1
                else:
                    output_sites_T.extend(addH2O(coord_linker, coord_linker_neighbour, metal))

                for __site__ in output_sites_T:
                    self.output_sites.append(__site__)

        addedAtoms = merge_sites(mgStructure.Structure.from_sites(self.output_sites))
        WriteStructure(self.output_dir, addedAtoms, name = 'CONTCAR'+'_'+str(self.linker_type)+'_debug_addedAtoms', sort = True)

        WriteStructure(self.output_dir, deleted_linker, name = 'CONTCAR'+'_'+str(self.linker_type)+'_debug_deleted_linker', sort = True)     

        for __site__ in mof:
            self.output_sites.append(__site__)
        self.outStructure = merge_sites(mgStructure.Structure.from_sites(self.output_sites))
        if abs(added_charge) != abs(self.charge_comp):
            WriteStructure(self.output_dir, self.outStructure, name = 'CONTCAR'+'_'+str(self.linker_type)+'_chargeError', sort = True)
        else:
            WriteStructure(self.output_dir, self.outStructure, name = 'CONTCAR'+'_'+str(self.linker_type), sort = True)
        return 
        
    def Build_supercell(self):
        
        t0 = time.time()
        
        """
        build MOF supercell
        """
        self.mof_superCell = self.original_mof
        self.mof_superCell = self.make_supercell(copy.deepcopy(self.original_mof), self.superCell)

        print("3.1 Finished super cell build, which took %f seconds" % (time.time()-t0))
        
    def DefectGen(self):

        # extract the right linkers from the supercell
        _linker_info_LCA_ = np.array([ s.LCA for s in self.mof_superCell])
        _linker_info_linker_label_ = np.array([ s.linker_label for s in self.mof_superCell])
            # TODO: define the varible type, int or str? could introduce errors! 
        all_index = np.where(_linker_info_linker_label_==self.linker_type)[0]
        possible_linker_LCA = np.unique(_linker_info_LCA_[all_index])
        
        # test if the numofdefect is possible
        if self.numofdefect > len(possible_linker_LCA):
            print( "Due to there are multiple linkers, the specified defect cannot be reached,\
                 the current conc.is%.2fm changed to%.2f"\
                    %(self.defectConc,self.defectConc*len(possible_linker_LCA)/self.numofdefect) )
            self.numofdefect = len(possible_linker_LCA)

        sampled_linker_to_be_delete = np.random.choice(possible_linker_LCA,size=self.numofdefect, replace=False )
        self.vacant_mof = copy.deepcopy(self.mof_superCell)

        # debug use to visulize all molecules
        # for i in range(0,4):
        #     debugsite = []
        #     for site in self.mof_superCell: 
        #         if site.LCA ==i :
        #             debugsite.append(site)
        #     debug = mgStructure.Structure.from_sites(debugsite)
        #     WriteStructure(self.output_dir, debug, name = 'CONTCAR_'+str(i), sort = True)


        # delete site
        deleted_sites = []
        for site in self.mof_superCell:
            if site.LCA in sampled_linker_to_be_delete:
                current_index = np.array([s.fixed_id for s in self.vacant_mof])
                site_tobe_delete_index = np.where(current_index == site.fixed_id)[0]
                if len(site_tobe_delete_index) > 1:
                    raise("error not unique index")
                else:
                    site_tobe_delete_index = site_tobe_delete_index[0]
                deleted_sites.append(self.vacant_mof[site_tobe_delete_index])
                self.vacant_mof.remove_sites([site_tobe_delete_index])
            
        self.deleted_linker = mgStructure.Structure.from_sites(deleted_sites)

        WriteStructure(self.output_dir, self.vacant_mof, name = 'POSCAR_'+str(self.linker_type), sort = True)
        # capping OMS
        try: 
            self.Sub_with_Capping_agent( copy.deepcopy(self.vacant_mof), copy.deepcopy(self.deleted_linker))
        except:
            shutil.move(self.output_dir, self.output_dir+'_debug')
        
        return
                
        
    def ReverseMonteCarlo(self, candidate_index_list = None, SRO = 0, beta = 1.0, N = 10000 , MaxErr = 0.001):
        '''
        neighbor_list: 2-D list. [[1, 'Yb', ['Nd', 'Yb']]...]
        SRO: target alpha (short range order)
        N: steps number
        beta: smoothfactor for exp(beta|alpha-SRO|)
        '''

        neighbor_list = copy.deepcopy(self.neighbor_list)
        
        #old_neighbor_list=copy.deepcopy(neighbor_list)
        if self.ReverseMonteCarlo_flag == False:
            self.Sub_with_Capping_agent(neighbor_list)    
            print("Finish Coordination env analysis, which took %f Seconds" % (time.time()-t0))
            
            return neighbor_list  
            
        t0 = time.time()
        if candidate_index_list == None:
            candidate_index_list = self.candidate_index_list


        if len(candidate_index_list)<=2:
            return neighbor_list
        n_metal = len(neighbor_list)
        alpha_Yb_0 = WarrenCowleyParameter(neighbor_list, 'normal', 'defect')
        metric_Yb_0 = abs(alpha_Yb_0-SRO)
        alpha_Yb_history = []
        n = 0
        if SRO == None:
            SRO = np.inf
            outputAll = True
            
        while n <= N:# and metric_Yb_0 >= MaxErr:
            n += 1
            ID_1, ID_2 = random.sample(candidate_index_list, 2)
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
                if alpha_Yb not in alpha_Yb_history:
                    alpha_Yb_history.append(alpha_Yb)
                    print(alpha_Yb)
                
        self.Sub_with_Capping_agent(neighbor_list)    
        print("Finish Coordination env analysis, which took %f Seconds" % (time.time()-t0))
            
        return neighbor_list  
        
def XRD(self):
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    xrd = XRDCalculator()
    pattern = xrd.get_pattern(self.outStructure)
