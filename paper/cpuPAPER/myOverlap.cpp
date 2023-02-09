/* 
 * cpuPAPER/myOverlap.cpp
 * Utility program to calculate hard-sphere overlap volume between
 * two molecules, without reorientation.
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

#include "cudaVolumeTypes.h"
#include "hostVolume.h"
//#include "oeInterface.cpp"
#include <iostream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
using namespace std;

#include "config.h"

inline float min(float a,float b) {
    return (a>b) ? b: a;
}

inline float max(float a,float b) {
    return (a>b)? a : b;
}

inline float distsqr(const float x,const float y,const float z,const float4& b) {
    float tmpx = b.x - x;
    float tmpy = b.y - y;
    float tmpz = b.z - z;
    return tmpx*tmpx + tmpy*tmpy + tmpz*tmpz;
}

CUDAmol xyzwFileToCUDAmol(char* fn) {
    FILE* f=fopen(fn,"rt");
    char buf[1024];
    fgets(buf,1023,f);
    CUDAmol mol;
    mol.natoms = atoi(buf);
    mol.atoms = new float4[mol.natoms];
    int nonHcount = 0;
    for (int i = 0; i < mol.natoms; i++) {
        fgets(buf,1023,f);
        sscanf(buf,"%f %f %f %f",&(mol.atoms[i].x),&(mol.atoms[i].y),&(mol.atoms[i].z),&(mol.atoms[i].w));
        if (mol.atoms[i].w != 1.20f)
            nonHcount++;
        //cout << "Read atom ("<<mol.atoms[i].x<<","<<mol.atoms[i].y<<","<<mol.atoms[i].z<<","<<mol.atoms[i].w<<")\n";
    }

    #ifdef IGNORE_HYDROGENS
    float4* nonH = new float4[nonHcount];
    for (int i = 0,j=0; i < mol.natoms; i++) {
        if (mol.atoms[i].w != 1.20f) {
            memcpy(nonH+j,mol.atoms+i,16);
            j++;
        }
    }
    //printf("Read %d atoms from file; retained %d non-hydrogen atoms\n",mol.natoms,nonHcount);
    delete[] mol.atoms;
    mol.atoms = nonH;
    mol.natoms = nonHcount;
    #endif
    
    #ifdef ALL_CARBON_RADII
    for (int i = 0; i < mol.natoms; i++)
        mol.atoms[i].w = 1.70f;
    #endif
    
    return mol;
}
// Calculates overlap volume on a 1/3 angstrom grid
float hostOverlapVolumeHS(const CUDAmol& ref,const CUDAmol& fit) {
    // Calculate bounding box
    float xmin,xmax,ymin,ymax,zmin,zmax;
    xmin= ref.atoms[0].x-ref.atoms[0].w;
    ymin= ref.atoms[0].y-ref.atoms[0].w;
    zmin= ref.atoms[0].z-ref.atoms[0].w;
    xmax= ref.atoms[0].x+ref.atoms[0].w;
    ymax= ref.atoms[0].y+ref.atoms[0].w;
    zmax= ref.atoms[0].z+ref.atoms[0].w;

    for (uint i = 1; i < ref.natoms; i++) {
        xmin = min(xmin,ref.atoms[i].x-ref.atoms[i].w);
        ymin = min(ymin,ref.atoms[i].y-ref.atoms[i].w);
        zmin = min(zmin,ref.atoms[i].z-ref.atoms[i].w);
        xmax = max(xmax,ref.atoms[i].x+ref.atoms[i].w);
        ymax = max(ymax,ref.atoms[i].y+ref.atoms[i].w);
        zmax = max(zmax,ref.atoms[i].z+ref.atoms[i].w);
    }
    for (uint i = 0; i < fit.natoms; i++) {
        xmin = min(xmin,fit.atoms[i].x-fit.atoms[i].w);
        ymin = min(ymin,fit.atoms[i].y-fit.atoms[i].w);
        zmin = min(zmin,fit.atoms[i].z-fit.atoms[i].w);
        xmax = max(xmax,fit.atoms[i].x+fit.atoms[i].w);
        ymax = max(ymax,fit.atoms[i].y+fit.atoms[i].w);
        zmax = max(zmax,fit.atoms[i].z+fit.atoms[i].w);
    }

    const float res = 0.33f;
    uint count = 0;
    for (float xcor = xmin; xcor < xmax; xcor += res) {
    for (float ycor = ymin; ycor < ymax; ycor += res) {
    for (float zcor = zmin; zcor < zmax; zcor += res) {
        bool refAtomFound = false;
        for (uint i = 0; i < ref.natoms && !refAtomFound; i++)
            refAtomFound = (ref.atoms[i].w * ref.atoms[i].w >= distsqr(xcor,ycor,zcor,ref.atoms[i]));
        if (!refAtomFound) continue;
        bool fitAtomFound = false;
        for (uint i = 0; i < fit.natoms && !fitAtomFound; i++)
            fitAtomFound = (fit.atoms[i].w * fit.atoms[i].w >= distsqr(xcor,ycor,zcor,fit.atoms[i]));
        // if we're here, there is a ref atom in range, so just check if there's a fit atom too
        if (fitAtomFound) count++;
    }}}

    return count*res*res*res;
    
}

int main(int argc,char** argv) {
    if (argc < 3) {
        cout << "myOverlap [reference file] [fit file]\n";
        return 1;
    }
    CUDAmol refmol = xyzwFileToCUDAmol(argv[1]);
    //cout << "Reference molecule: "<<refmol.natoms<<" atoms"<<endl;
    CUDAmol fitmol = xyzwFileToCUDAmol(argv[2]);
    //cout << "Fit molecule: "<<fitmol.natoms<<" atoms"<<endl;
    Grid nullgrid;
    
    //cout << getOverlapVolume(refmol,fitmol,nullgrid)<<endl;
    cout << hostOverlapVolumeHS(refmol,fitmol)<<endl;

return 0;
    
}
