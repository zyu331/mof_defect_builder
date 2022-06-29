#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 20:50:37 2022

@author: jace
"""
import numpy as np
import os
import copy

import pymatgen.core.bonds  as mgBond
from pymatgen.io.vasp.inputs import Poscar
import pymatgen.core.structure as mgStructure
from pymatgen.core.bonds import CovalentBond, get_bond_length
from pymatgen.util.coord import all_distances, get_angle, lattice_points_in_supercell
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.operations import SymmOp

import cappingAgent

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
    
def WarrenCowleyParameter(neighbor_list, center_atom, noncenter_atom):
    '''
    neighbor_list: 2-D list. [[1, 'Yb', ['Nd', 'Yb']]...]
    center_atom: reference atom. 'Yb' or 'Nd'
    '''
    n_metal = len(neighbor_list)
    # n_neighbor = len(neighbor_list[0][-1])
    # find center atom list
    center_atoms =[]
    for neighbor in neighbor_list:
        if neighbor[1] == center_atom:
            center_atoms.append(neighbor)
    n_center_atom = len(center_atoms)
    # neighbor_list_names = ['neighbor_'+str(i+1) for i in range(n_neighbor)]
    probs = []
    for _index_, _type_, _neighbor_list_ in neighbor_list:
        # list_name = [x[-1][i] for x in center_atoms]
        n_center_atom = [ neighbor_list[x][1] for x in _neighbor_list_ ].count(center_atom)
        n_noncenter_atom = [ neighbor_list[x][1] for x in _neighbor_list_ ].count(noncenter_atom)
        if (n_center_atom + n_noncenter_atom) != len(_neighbor_list_):
            raise "neighbor error"
        if n_center_atom == 0:
            prob = 0
        else:
            prob = (n_center_atom - n_noncenter_atom)/n_center_atom
        probs.append(prob)
    prob_BgivenA = np.average(probs)
    prob_A = n_center_atom/n_metal
    prob_B = 1-prob_A
#    alpha = 1-prob_A*prob_BgivenA/prob_B
    alpha = 1-prob_BgivenA/prob_B
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

def substitute_funcGroup_old( molecular, index: int, func_group, delList, addList, bond_order: int = 1):

    # Find the nearest neighbor that is not a terminal atom.
    all_non_terminal_nn = []
    for nn, dist, _, _ in molecular.get_neighbors(molecular[index], 3):
        # Check that the nn has neighbors within a sensible distance but
        # is not the site being substituted.
        for inn, dist2, _, _ in molecular.get_neighbors(nn, 3):
            if inn != molecular[index] and dist2 < 1.2 * get_bond_length(nn.specie, inn.specie):
                all_non_terminal_nn.append((nn, dist))
                break

    if len(all_non_terminal_nn) == 0:
        raise RuntimeError("Can't find a non-terminal neighbor to attach functional group to.")

    non_terminal_nn = min(all_non_terminal_nn, key=lambda d: d[1])[0]

    # Set the origin point to be the coordinates of the nearest
    # non-terminal neighbor.
    origin = non_terminal_nn.coords

    # Pass value of functional group--either from user-defined or from
    # functional.json
    fgroup = func_group

    # If a bond length can be found, modify func_grp so that the X-group
    # bond length is equal to the bond length.
    try:
        bl = get_bond_length(non_terminal_nn.specie, fgroup[1].specie, bond_order=bond_order)
    # Catches for case of incompatibility between Element(s) and Species(s)
    except TypeError:
        bl = None

    if bl is not None:
        fgroup = fgroup.copy()
        vec = fgroup[0].coords - fgroup[1].coords
        vec /= np.linalg.norm(vec)
        fgroup[0] = "X", fgroup[1].coords + float(bl) * vec

    # Align X to the origin.
    x = fgroup[0]
    fgroup.translate_sites(list(range(len(fgroup))), origin - x.coords)

    # Find angle between the attaching bond and the bond to be replaced.
    v1 = fgroup[1].coords - origin
    v2 = molecular[index].coords - origin
    angle = get_angle(v1, v2)

    if 1 < abs(angle % 180) < 179:
        # For angles which are not 0 or 180, we perform a rotation about
        # the origin along an axis perpendicular to both bonds to align
        # bonds.
        axis = np.cross(v1, v2)
        op = SymmOp.from_origin_axis_angle(origin, axis, angle)
        fgroup.apply_operation(op)
    elif abs(abs(angle) - 180) < 1:
        # We have a 180 degree angle. Simply do an inversion about the
        # origin
        for i, fg in enumerate(fgroup):
            fgroup[i] = (fg.species, origin - (fg.coords - origin))

    # Remove the atom to be replaced, and add the rest of the functional
    # group.
    delList.append(molecular[index])
    for _site_ in fgroup[1:]:
        s_new = PeriodicSite(_site_.species, _site_.coords, molecular.lattice, coords_are_cartesian=True)
        addList.append(s_new)
    
    return delList, addList


def substitute_funcGroup( mode, molecular, indexes: list, bond_order: int = 1, oldMolecular = None):
    # Find the nearest neighbor that is not a terminal atom.
    if mode == 'N':
        """
        find the neighbour CC coord to N
        """
        if len(indexes)!=1:
            raise "wrong N input, more than two N atoms"

        for index in indexes:
            all_non_terminal_nn = []


            for nn, dist, _, _ in molecular.get_neighbors(molecular[index], 1.8):
                # Check that the nn has neighbors within a sensible distance but
                # is not the site being substituted.
                for inn, dist2, _, _ in molecular.get_neighbors(nn, 1.8):
                    if inn != molecular[index] and dist2 < 1.2 * get_bond_length(nn.specie, inn.specie):
                        all_non_terminal_nn.append((nn, dist))
                        break
                    
        if len(all_non_terminal_nn) < 2:
            raise RuntimeError("Can't find a proper env to add functional group to.")
        elif len(all_non_terminal_nn) > 2:
            raise "more than two NN for OO atoms"
        
        sites = []
        if oldMolecular:
            oldMolecular.sites.append(PeriodicSite('O', molecular[indexes[0]].coords, molecular.lattice, coords_are_cartesian = True))
        else:
            sites.append(PeriodicSite('O', molecular[indexes[0]].coords, molecular.lattice, coords_are_cartesian = True))
        for x in all_non_terminal_nn:
            _site_ = x[0]
            _coords_ = molecular[indexes[0]].coords + (_site_.coords - molecular[indexes[0]].coords)*0.96/x[1]
            _site_pbc_ = PeriodicSite('H', _coords_ , molecular.lattice, coords_are_cartesian = True)
            if oldMolecular:
                oldMolecular.sites.append(_site_pbc_)
            else:
                sites.append(_site_pbc_)
        if oldMolecular:
            new_ad = oldMolecular
        else:
            new_ad = mgStructure.Structure.from_sites(sites)
             
        
    elif mode == 'OO':
        """
        find the neighbour C coord to OO
        """
        if len(indexes)!=2:
            raise "wrong OO input, more than two Oxygen atoms"
        OO_group = []
        for index in indexes:
            all_non_terminal_nn = []
            OO_group.append(molecular[index])   

            for nn, dist, _, _ in molecular.get_neighbors(molecular[index], 2):
                # Check that the nn has neighbors within a sensible distance but
                # is not the site being substituted.
                for inn, dist2, _, _ in molecular.get_neighbors(nn, 2):
                    if inn != molecular[index] and dist2 < 1.2 * get_bond_length(nn.specie, inn.specie):
                        all_non_terminal_nn.append((nn, dist))
                        break
                    
        OO_coords = [x.coords[0] for x in OO_group]
        for x in all_non_terminal_nn:
            _flag_ = any([ all(x[0].coords == y) for y in OO_coords])
            if _flag_:
                all_non_terminal_nn.remove(x)
                
        if len(all_non_terminal_nn) < 1:
            raise RuntimeError("Can't find a proper env to add functional group to.")
        elif len(all_non_terminal_nn) > 1:
            raise "more than two NN for OO atoms"
        
                
        """
        first add H2O, then add the additional two hydrogen by aligh COM and rotate 
        pbc fix to find the mid point is pain, and improvement is needed 
        
        """
              
        dist_OO_pbc = molecular[indexes[0]].distance(molecular[indexes[1]])
        dist_OO = molecular[indexes[1]].coords - molecular[indexes[0]].coords
        for i,x in enumerate(dist_OO):
            if abs(x) < dist_OO_pbc:
                dist_OO[i] = x
            else:
                dist_OO[i] = -x
        mid = molecular[indexes[0]].coords + dist_OO_pbc/2/np.sqrt(sum(dist_OO**2))* dist_OO
        dist_CO_pbc = all_non_terminal_nn[0][0].distance(molecular[indexes[1]])
        dist_CO = all_non_terminal_nn[0][0].coords - molecular[indexes[1]].coords
        for i,x in enumerate(dist_CO):
            if abs(x) < dist_CO_pbc:
                dist_CO[i] = x
            else:
                dist_CO[i] = -x        
        _coord_ = mid + 0.16844981387963906*dist_CO_pbc/np.sqrt(sum(dist_CO**2))* dist_CO


        h2ooh = cappingAgent.h2ooh()
        scale = np.sqrt(sum((h2ooh[0].coords - h2ooh[1].coords)**2))/dist_OO_pbc
        for i,x in enumerate(h2ooh):
            _v = x.coords - h2ooh[3].coords
            x.coords = h2ooh[3].coords + scale*_v
        # Align X to the origin. use the mid H atom as the origin
        x = h2ooh[3].coords
        h2ooh.translate_sites(list(range(len(h2ooh))), _coord_ - x)
    
        # Find angle between the attaching bond and the bond to be replaced.
        v1 = h2ooh[0].coords - _coord_
        v2 = molecular[index].coords - _coord_
        for i,_v in enumerate(v2): 
            if abs(_v) > dist_OO_pbc:
                v2[i] = -_v
        angle = get_angle(v1, v2)
    
        if 1 < abs(angle % 180) < 179:
            # For angles which are not 0 or 180, we perform a rotation about
            # the origin along an axis perpendicular to both bonds to align
            # bonds.
            axis = np.cross(v1, v2)
            op = SymmOp.from_origin_axis_angle(_coord_, axis, angle)
            h2ooh.apply_operation(op)
        elif abs(abs(angle) - 180) < 1:
            # We have a 180 degree angle. Simply do an inversion about the
            # origin
            for i, fg in enumerate(h2ooh):
                h2ooh[i] = (fg.species, _coord_ - (fg.coords - _coord_))        
            
                # Set the origin point to be the coordinates of the nearest
                # non-terminal neighbor.
        """
        min H atoms to the linker; max H atoms to the metal
        """

        axis =  h2ooh[0].coords -  h2ooh[1].coords
        distant_0, distant_1 = np.inf, np.inf       
        h2ooh_origin = copy.deepcopy(h2ooh)
        
        for i in range(0,360):
            _h2ooh_ = copy.deepcopy(h2ooh_origin)
            op = SymmOp.from_origin_axis_angle(_coord_, axis, i)
            _h2ooh_.apply_operation(op)
            site_0 = PeriodicSite(_h2ooh_.sites[2].species,_h2ooh_.sites[2].coords, molecular.lattice, coords_are_cartesian = True)
            site_1 = PeriodicSite(_h2ooh_.sites[4].species,_h2ooh_.sites[4].coords, molecular.lattice, coords_are_cartesian = True)
            _distance_0 = all_non_terminal_nn[0][0].distance(site_0)
            _distance_1 = all_non_terminal_nn[0][0].distance(site_1)
            if _distance_0 < distant_0:
                h2ooh[2] = _h2ooh_[2]
                distant_0 = _distance_0
            if _distance_1 < distant_1:
                h2ooh[4] = _h2ooh_[4]
                distant_1 = _distance_1
        
        sites = []
        for _site_ in h2ooh.sites:
            _site_.lattice = molecular.lattice
            _site_pbc_ = PeriodicSite(_site_.species,_site_.coords, molecular.lattice, coords_are_cartesian = True)
            sites.append(_site_pbc_)
            
        if oldMolecular:
            for _site_ in sites:
                oldMolecular.sites.append(_site_)
            new_ad = oldMolecular
        else:
            new_ad = mgStructure.Structure.from_sites(sites)
            new_ad.lattice = molecular.lattice
        
    return new_ad