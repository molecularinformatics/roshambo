/*
 * inputFileReader.cpp
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
#include <typeinfo>
#include "inputPreprocessor.h"
#include <openbabel/mol.h>
#include <openbabel/obconversion.h>
#include <openbabel/obiter.h>
#include <openbabel/data.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/PeriodicTable.h>

using namespace std;
using namespace OpenBabel;
using namespace RDKit;

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

dCUDAmol obMolTodCUDAmol(OBMol& obmol) 
{ //{{{
    dCUDAmol mol;
    #ifdef IGNORE_HYDROGENS
    mol.natoms = obmol.NumHvyAtoms();
    #else
    mol.natoms = obmol.NumAtoms();
    #endif
    mol.x = new float[mol.natoms];
    mol.y = new float[mol.natoms];
    mol.z = new float[mol.natoms];
    mol.a = new float[mol.natoms];

    const float partialalpha = 2.41798793102f;
    unsigned int idx = 0;
    for (OBMolAtomIter a(obmol); a; a++) {
        #ifdef IGNORE_HYDROGENS
        if (a->GetAtomicNum() == 1) continue;
        #endif
        double* coords = a->GetCoordinate();
        mol.x[idx] = coords[0];
        mol.y[idx] = coords[1];
        mol.z[idx] = coords[2];
        #ifdef ALL_CARBON_RADII
        float rad = 1.70f;
        #else
        float rad = etab.GetVdwRad(a->GetAtomicNum());
        #endif
        mol.a[idx] = partialalpha/(rad*rad);
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

CUDAmol rdMolToCUDAmol(RDKit::ROMol* rdmol)
{
    CUDAmol mol;
    if (!rdmol) throw std::invalid_argument("Invalid molecule object");
    #ifdef IGNORE_HYDROGENS
    const unsigned int natoms = rdmol->getNumHeavyAtoms();
    #else
    const unsigned int natoms = rdmol->getNumAtoms();
    #endif
    mol.natoms = natoms;
    mol.atoms = new float4[mol.natoms];
    const auto &conf = rdmol->getConformer();
    unsigned int idx = 0;
    const PeriodicTable *table = PeriodicTable::getTable();
    for (unsigned int i = 0; i < natoms; ++i) {
        const auto& atom = rdmol->getAtomWithIdx(i);
        const auto& coords = conf.getAtomPos(i);
        mol.atoms[idx].x = coords.x;
        mol.atoms[idx].y = coords.y;
        mol.atoms[idx].z = coords.z;
        #ifdef ALL_CARBON_RADII
        mol.atoms[idx].w = 1.70f;
        #else
        mol.atoms[idx].w = table->getRvdw(atom->getAtomicNum());
        #endif
        ++idx;
    }
    return mol;
}

dCUDAmol rdMolTodCUDAmol(RDKit::ROMol* rdmol)
{
    dCUDAmol mol;
    if (!rdmol) throw std::invalid_argument("Invalid molecule object");
    #ifdef IGNORE_HYDROGENS
    const unsigned int natoms = rdmol->getNumHeavyAtoms();
    #else
    const unsigned int natoms = rdmol->getNumAtoms();
    #endif
    mol.natoms = natoms;
    mol.x = new float[natoms];
    mol.y = new float[natoms];
    mol.z = new float[natoms];
    mol.a = new float[natoms];

    const float partialalpha = 2.41798793102f;
    const auto &conf = rdmol->getConformer();
    unsigned int idx = 0;
    const PeriodicTable *table = RDKit::PeriodicTable::getTable();
    for (unsigned int i = 0; i < natoms; ++i) {
        const auto& atom = rdmol->getAtomWithIdx(i);
        const auto& coords = conf.getAtomPos(i);
        #ifdef IGNORE_HYDROGENS
        if (table->getRvdw(atom->getAtomicNum()) == 1) continue;
        #endif
        mol.x[idx] = coords.x;
        mol.y[idx] = coords.y;
        mol.z[idx] = coords.z;
        #ifdef ALL_CARBON_RADII
        float rad = 1.70f;
        #else
        float rad = table->getRvdw(atom->getAtomicNum());
        #endif
        mol.a[idx] = partialalpha / (rad * rad);
        ++idx;
    }
    return mol;
}

float3 atomset_centroid_rdkit(RDKit::ROMol* rdmol, const std::set<int>& atoms) {
    float3 cen;
    cen.x = cen.y = cen.z = 0.0f;
    const auto &conf = rdmol->getConformer();

    for (const auto& idx : atoms) {
        const auto& coords = conf.getAtomPos(idx);
        cen.x += coords.x;
        cen.y += coords.y;
        cen.z += coords.z;
    }
    const int num_atoms = static_cast<float>(atoms.size());
    cen.x /= num_atoms;
    cen.y /= num_atoms;
    cen.z /= num_atoms;

    return cen;
}

list<set<int>> find_ring_systems_rdkit(RDKit::ROMol* rdmol, bool includeSpiro=false) {
    RDKit::RingInfo *ring_info = rdmol->getRingInfo();
    list<set<int>> systems;
    for (const auto& ring : ring_info->atomRings()) {
        set<int> ringAts(ring.begin(), ring.end());
        list<set<int>> nSystems;
        for (const auto& system : systems) {
            int nInCommon = count_if(system.begin(), system.end(),
                                     [&](int i){ return ringAts.count(i) > 0; });
            if (nInCommon && (includeSpiro || nInCommon > 1)) {
                ringAts.insert(system.begin(), system.end());
            } else {
                nSystems.push_back(system);
            }
        }
        nSystems.push_back(ringAts);
        systems = nSystems;
    }
    return systems;
}

extern "C" void molFromFile(string filename,list<CUDAmol>& cmols,list<dCUDAmol>& dcmols,list<list<float3> >& ring_centroids) {
    OBConversion conv;
    ifstream ifs(filename.c_str(),ifstream::in|ifstream::binary);
    OBFormat* inFormat = conv.FormatFromExt(filename.c_str());
    conv.SetInFormat(inFormat);
    OBMol mol;
    // printf("Loading molecule %s\n",filename.c_str());
    // The following code needs to change to deal with multi-conformer molecules
    if (!conv.Read(&mol,&ifs)) {
        cout << "FATAL ERROR: could not read file "<<filename<<endl;
        exit(1);
    }
    list<set<int> > ringsystems = find_ring_systems(mol);
    cmols.push_back(obMolToCUDAmol(mol));
    dcmols.push_back(obMolTodCUDAmol(mol));
    list<float3> centroids;
    for (list<set<int> >::iterator i = ringsystems.begin(); i != ringsystems.end(); i++) {
        centroids.push_back(atomset_centroid(mol,*i));
    }
    ring_centroids.push_back(centroids);
    return;
}

extern "C" void molFromRDKit(RDKit::ROMol* rdmol,list<CUDAmol>& cmols,
                             list<dCUDAmol>& dcmols,list<list<float3> >& ring_centroids)
{
    list<set<int> > ringsystems = find_ring_systems_rdkit(rdmol);
    cmols.push_back(rdMolToCUDAmol(rdmol));
    dcmols.push_back(rdMolTodCUDAmol(rdmol));
    list<float3> centroids;
    for (list<set<int> >::iterator i = ringsystems.begin(); i != ringsystems.end(); i++) {
        centroids.push_back(atomset_centroid_rdkit(rdmol,*i));
    }
    ring_centroids.push_back(centroids);
    return;
}
