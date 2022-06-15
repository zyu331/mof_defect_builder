#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 21:48:48 2021

@author: jace

This package is designed to generate initial guess of defect MOFS structures

"""


      

from DefectGenerator.CifDefectGenerator import CifDefectGenerator
import os

def main():
    """
    Default file path system:
        - Input/original cifs: 
            'cifs/'
        - temp file storage, including OMS info and different linker/metal seperation options:
            'workingDir/'
        - final output:            'results/MOFName/'
            
    """
    
    path = r'cifs'
    file = 'ATOXEN_clean.cif'
    resultDir = 'results'
    if not os.path.exists(resultDir):
        os.system('mkdir '+resultDir)

    oprator = CifDefectGenerator(path,file,resultDir)
    oprator.Linkervacancy()

if __name__ == '__main__':
    main()