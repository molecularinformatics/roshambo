/* 
 * cpuPAPER/oeInterface.cpp
 * Loads molecule files from disk using OpenEye libraries.
 *
 *
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */
// includes, project
#include "openeye.h"
#include "oechem.h"
#include "oeshape.h"
#include "cudaVolumeTypes.h"
#include <stdio.h>
using namespace OEChem;
using namespace OESystem;
using namespace OEShape;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
extern "C" CUDAmol molfileToCUDAmol(char* filename);
CUDAmol oeMolToCUDAmol(OEMol& mol);
extern "C" void CUDAmolToFile(char* model_file,char* out_file,CUDAmol& cmol);
extern "C" void combinedCUDAmolToFile(char* model_file1,char* model_file2,char* outfile1,char* outfile2, CUDAmol& cmol,uint natoms1);

extern "C" CUDAmol molfileToCUDAmol(char* filename) {
    OEMol mol;
    oemolistream ims;
    ims.open(filename);
    ims >> mol;
    printf("File %s: total OE volume: %f\n",filename,OECalcVolume(mol));
    CUDAmol cmol = oeMolToCUDAmol(mol);
    return cmol;
}

CUDAmol oeMolToCUDAmol(OEMol& mol) {
    CUDAmol cmol;
    cmol.natoms = mol.NumAtoms();
    cmol.atoms = new float4[cmol.natoms];
    float cord[3];
    uint i = 0;
    for (OEIter<OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        mol.GetCoords(atom,cord);
        cmol.atoms[i].x = cord[0];
        cmol.atoms[i].y = cord[1];
        cmol.atoms[i].z = cord[2];
        cmol.atoms[i].w = atom->GetRadius();
        i++;
        //printf("Atom %d @ [%f,%f,%f]\n",i-1,cord[0],cord[1],cord[2]);
    }
    return cmol;
}

extern "C" void CUDAmolToFile(char* model_file,char* out_file,CUDAmol& cmol) {
    OEMol mol;
    model_file = 0; // Shut up the compiler
    for (uint i = 0; i < cmol.natoms; i++) {
        OEAtomBase* atom = mol.NewAtom(6);
        float4 catom = cmol.atoms[i];
        float xyz[3] = {catom.x,catom.y,catom.z};
        mol.SetCoords(atom,xyz);
    }
    oemolostream oms;
    oms.open(out_file);
    oms << mol;
    oms.close();
    return;
    /*OEMol mol;
    oemolistream ims;
    ims.open(model_file);
    ims >> mol;
    int i = 0;
    for (OEIter<OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        float4 catom = cmol.atoms[i];
        float xyz[3] = {catom.x,catom.y,catom.z};
        mol.SetCoords(atom,xyz);
        i++;
    }
    oemolostream oms;
    oms.open(out_file);
    oms << mol;
    oms.close();*/
}

extern "C" void combinedCUDAmolToFile(char* model_file1,char* model_file2,char* outfile1,char* outfile2, CUDAmol& cmol,uint natoms1) {
    CUDAmol cmol1 = cmol;
    CUDAmol cmol2 = cmol;
    cmol1.natoms = natoms1;
    cmol2.natoms -= natoms1;
    cmol2.atoms += natoms1;
    CUDAmolToFile(model_file1,outfile1,cmol1);    
    CUDAmolToFile(model_file2,outfile2,cmol2);    
}
