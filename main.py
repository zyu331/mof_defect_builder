#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 21:48:48 2021

@author: jace

This package is designed to generate initial guess of defect MOFS structures

"""

from src.DefectMOFGenerator import DefectMOFStructureBuilder
import os
from ase.io import read, write

def main(formatting):
    """
    Default file path system:
        - Input/original cifs: 
            'cifs/'
    """

    if formatting :
        cif_path = 'cifs'
        cifs = os.listdir(cif_path)
        cifs.sort()
        for cif in cifs:
            mof = read(os.path.join(cif_path, cif))
            write(os.path.join(cif_path, cif), mof)
    
    cifFolder = 'cifs/'
    cifList = os.listdir(cifFolder)
    stat_file = ['archive','cifs','output','main.py','README.md','requirements.txt','src']
    for cifName in cifList:
        op = DefectMOFStructureBuilder( cifName, input_dir = cifFolder, output_dir= 'output/')
        op.LinkerVacancy()    

if __name__ == '__main__':
    main(True)
