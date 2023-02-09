/*
 * cpuPAPER/inputFileReader.cpp
 * Loads molecules from disk using OpenBabel
 *
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */


#include "cudaVolumeTypes.h"
#include "config_input.h"
#include <list>
#include <set>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdlib>
#include "inputPreprocessor.h"
#include <openbabel/mol.h>
#include <openbabel/obconversion.h>
#include <openbabel/obiter.h>
#include <openbabel/data.h>

using namespace std;
using namespace OpenBabel;

CUDAmol obMolToCUDAmol(OBMol& obmol) 
{ //{{{
    CUDAmol mol;
    #ifdef IGNORE_HYDROGENS
    mol.natoms = obmol.NumHvyAtoms();
    #else
    mol.natoms = obmol.NumAtoms();
    #endif
    mol.atoms = new float4[mol.natoms];
    unsigned int idx = 0;
    for (OBMolAtomIter a(obmol); a; a++) {
        #ifdef IGNORE_HYDROGENS
        if (a->GetAtomicNum() == 1) continue;
        #endif
        double* coords = a->GetCoordinate();
        mol.atoms[idx].x = coords[0];
        mol.atoms[idx].y = coords[1];
        mol.atoms[idx].z = coords[2];
        //printf("Placing atom %d at coordinates (%f,%f,%f)\n",idx,coords[0],coords[1],coords[2]);
        #ifdef ALL_CARBON_RADII
        mol.atoms[idx].w = 1.70f;
        #else
        mol.atoms[idx].w = etab.GetVdwRad(a->GetAtomicNum());
        #endif
        idx++;
    }
    return mol;
} //}}}

float3 atomset_centroid(OBMol& mol,const set<int>& atoms) {
    float3 cen;
    cen.x = cen.y = cen.z = 0.0f;
    for (set<int>::iterator i = atoms.begin(); i!= atoms.end(); i++) {
       OBAtom* atom = mol.GetAtom(*i);
       double* coords = atom->GetCoordinate();
       cen.x += coords[0];
       cen.y += coords[1];
       cen.z += coords[2]; 
    }
    cen.x /= atoms.size();
    cen.y /= atoms.size();
    cen.z /= atoms.size();

    return cen;
}

extern "C" void molFromFile(string filename,list<CUDAmol>& cmols,list<list<float3> >& ring_centroids) {
    OBConversion conv;
    ifstream ifs(filename.c_str(),ifstream::in|ifstream::binary);
    OBFormat* inFormat = conv.FormatFromExt(filename.c_str());
    conv.SetInFormat(inFormat);
    OBMol mol;
    //printf("Loading molecule %s\n",filename.c_str());
    // The following code needs to change to deal with multi-conformer molecules
    if (!conv.Read(&mol,&ifs)) {
        cout << "FATAL ERROR: could not read file "<<filename<<endl;
        exit(1);
    }
    list<set<int> > ringsystems = find_ring_systems(mol);
    cmols.push_back(obMolToCUDAmol(mol));
    list<float3> centroids;
    for (list<set<int> >::iterator i = ringsystems.begin(); i != ringsystems.end(); i++) {
        centroids.push_back(atomset_centroid(mol,*i));
    }
    ring_centroids.push_back(centroids);
    //printf("Done loading molecule\n\n");
    return;
}
