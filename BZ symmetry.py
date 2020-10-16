# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:41:32 2020

@author: fjfar
"""

from pyprocar import ProcarSelect, ProcarParser, UtilsProcar, repair
import numpy as np

# parses each symmetry operation used by VASP
def outcarParse_Operators(outcar):
    with open(outcar) as f:
        txt = f.readlines()

    for i,line in enumerate(txt):
        if 'irot' in line:
            begin_table = i+1
        if 'Subroutine' in line:
            end_table = i-1

    operators = np.zeros((end_table-begin_table, 9))
    for i,line in enumerate(txt[begin_table:end_table]):
        str_list = line.split()
        num_list = [float(s) for s in str_list]
        operator = np.array(num_list)
        operators[i,:] = operator
    
    return operators

#finds rotation matrix from axis vector (normalized) and angle
def findR(operator, reciprocal_lattice):
    det_A = operator[1]
    #convert alpha to radians
    alpha = np.pi*operator[2]/180.
    #get rotation axis
    x = operator[3]
    y = operator[4]
    z = operator[5]
        
    R = np.array([
        [np.cos(alpha) + x**2*(1-np.cos(alpha)), x*y*(1-np.cos(alpha))-z*np.sin(alpha), x*z*(1-np.cos(alpha)) + y*np.sin(alpha)],
        [y*x*(1-np.cos(alpha)) + z*np.sin(alpha), np.cos(alpha) + y**2*(1-np.cos(alpha)), y*z*(1-np.cos(alpha)) - x*np.sin(alpha)],
        [z*x*(1-np.cos(alpha)) - y*np.sin(alpha), z*y*(1-np.cos(alpha)) + x*np.sin(alpha), np.cos(alpha) + z**2*(1-np.cos(alpha))]
    ])*det_A
    
    R = np.dot(np.dot(np.linalg.inv(reciprocal_lattice), R), reciprocal_lattice)
    R = np.round_(R, decimals=3)
    
    return R

#this function applies each symmetry operation to each kpoint
def apply_symmetries(kpoints, bands, spd, rotations):
    klist = []
    bandlist = []
    spdlist = []
    #for each symmetry operation
    for i,_ in enumerate(rotations): 
        #for each point
        for j,_ in enumerate(kpoints):
            #apply symmetry operation to kpoint
            sympoint_vector = np.dot(rotations[i], kpoints[j])
            sympoint = sympoint_vector.tolist()
            
            if sympoint not in klist:
                klist.append(sympoint)

                band = bands[j].tolist()
                bandlist.append(band)
                spd = spd[j].tolist()
                spdlist.append(spd)
            
    new_kpoints = np.array(klist)
    new_bands = np.array(bandlist)
    new_spd = np.array(spdlist)
    
    return new_kpoints, new_bands, new_spd  



# get files
procar_sym = input('Enter the filename/path for the PROCAR file:')
repair(procar_sym, procar_sym)
outcar = input('Enter the filename/path for the OUTCAR file:')
repair(outcar, outcar)

# get kpoints, bands, spd from PROCAR
print('Getting structure data...')

procarFile = ProcarParser()
procarFile.readFile(procar_sym, False)
data_sym = ProcarSelect(procarFile, deepCopy=True)

kpoints_sym = data_sym.kpoints
bands_sym = data_sym.bands
spd_sym = data_sym.spd

# get the symmetry operations
print('Finding symmetries...')
outcarparser = UtilsProcar()
reciprocal_lattice = np.transpose(outcarparser.RecLatticeOutcar(outcar))
operators = outcarParse_Operators(outcar)
rotations = np.array([findR(op, reciprocal_lattice) for op in operators])

# apply symmetry operations and boundary conditions
print('Applying symmetry operations...')
kpoints_full, bands_full, spd_full = apply_symmetries(kpoints_sym, bands_sym, spd_sym, rotations)

bound_ops = -1.0*(kpoints_full > 0.5) + 1.0*(kpoints_full < -0.5)
kpoints_full += bound_ops


