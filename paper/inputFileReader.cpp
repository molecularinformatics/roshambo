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
#include <GraphMol/ROMol.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/PeriodicTable.h>

using namespace std;
using namespace RDKit;


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
    const PeriodicTable *table = RDKit::PeriodicTable::getTable();
    for (unsigned int i = 0; i < rdmol->getNumAtoms(); ++i) {
        const auto& atom = rdmol->getAtomWithIdx(i);
        #ifdef IGNORE_HYDROGENS
        if (table->getRvdw(atom->getAtomicNum()) == 1.2) continue;
        #endif
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
    for (unsigned int i = 0; i < rdmol->getNumAtoms(); ++i) {
        const auto& atom = rdmol->getAtomWithIdx(i);
        #ifdef IGNORE_HYDROGENS
        if (table->getRvdw(atom->getAtomicNum()) == 1.2) continue;
        #endif
        const auto& coords = conf.getAtomPos(i);
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
