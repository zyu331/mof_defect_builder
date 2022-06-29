#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 21:48:48 2021

@author: jace

This package is designed to generate initial guess of defect MOFS structures

"""

from src.DefectMOFGenerator import DefectMOFStructureBuilder
import os
from pymatgen.io.cif import CifParser,CifWriter
def main():
    """
    Default file path system:
        - Input/original cifs: 
            'cifs/'
        - temp file storage, including OMS info and different linker/metal seperation options:
            'workingDir/'
        - final output:            'results/MOFName/'
            
    """
    
    cifFolder = 'cifs/'
    cifFile_Name = 'OCUNAC_manual_MIL101.cif'
    cifFile = CifParser(cifFolder + cifFile_Name)
    # os.system('mv '+cifFolder+cifFile_Name+' '+ cifFolder+'original.cif')
    structure = cifFile.get_structures()[0]
    out = CifWriter(structure)
    out.write_file('src/'+cifFile_Name)
    a = DefectMOFStructureBuilder(cifFolder+cifFile_Name)
    a.LinkerVacancy()

if __name__ == '__main__':
    main()