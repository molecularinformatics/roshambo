#
# prePCA.py
# Example code demonstrating PCA-based preprocessing for molecules
#
# This code, or one similar, should be run on molecules prior to
# processing with PAPER to do inertial overlay
# 
# Modifies each of its input files (specified on the command line)
# reorienting each molecule along its principal component axes
# Depends on the OpenEye Python toolkits
#  
# Author: Imran Haque, 2009
# Copyright 2009, Stanford University
#
# This file is licensed under the terms of the GPL. Please see
# the COPYING file in the accompanying source distribution for
# full license terms.


from openeye.oechem import *
import sys
import math
from numpy import *
from numpy.linalg import svd,det

def pcaProjectMol(mol):
    """ Reorient a molecule along its principal axes to reduce the size of the bounding box"""
    x = matrix([mol.GetCoords(x) for x in mol.GetAtoms()])
    (u,s,vh) = svd(x)
    print ("S:",s)
    # Check for negative determinant to avoid reflection
    if det(vh) < 0:
        vh = -vh
    #print vh.T*vh

    for atom in mol.GetAtoms():
        cords = matrix(mol.GetCoords(atom)).transpose()
        newcords = vh*cords
        mol.SetCoords(atom,newcords)

def centerMol(mol):
    centroid = zeros((1,3))
    count = 0
    for atom in mol.GetAtoms():
        centroid = centroid + array(mol.GetCoords(atom))
        count += 1
    centroid = centroid / count
    #print "centroid:",centroid
    for atom in mol.GetAtoms():
        #print "old coordinate:",array(mol.GetCoords(atom))
        #print "new coordinate:",(array(mol.GetCoords(atom)-centroid))
        cord = mol.GetCoords(atom)
        #print cord
        #print centroid
        newcord = (cord[0] - centroid[0][0],cord[1] - centroid[0][1],cord[2] - centroid[0][2])
        mol.SetCoords(atom,newcord)
        #print "new coordinate:",array(mol.GetCoords(atom))

def main():
    ifs = oemolistream()
    ofs = oemolostream()
    mol = OEMol()
    for f in sys.argv[1:]:
        ifs.open(f)
        OEReadMolecule(ifs,mol)
        ifs.close()
            
        OEAssignBondiVdWRadii(mol)
        centerMol(mol)
        pcaProjectMol(mol)
        
        ofs.open(f)
        OEWriteMolecule(ofs,mol)
        ofs.close()

try:
    import psyco
    psyco.full()
except:
    pass

main()
