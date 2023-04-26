/*
 * inputModule.cu
 * Molecule-file frontend to PAPER
 *  - loads files from disk (using inputFileReader)
 *  - sets up starting positions based on config_input
 *    (possibly using inputPreprocessor if using ring-centered modes)
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
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "inputFileReader.h"
#include "transformTools.h"

using namespace std;

float3 min(const float4& a,const float3& b) {
    float3 rv;
    rv.x = min(a.x,b.x);
    rv.y = min(a.y,b.y);
    rv.z = min(a.z,b.z);
    return rv;
}
float3 max(const float4& a,const float3& b) {
    float3 rv;
    rv.x = max(a.x,b.x);
    rv.y = max(a.y,b.y);
    rv.z = max(a.z,b.z);
    return rv;
}
float3 operator-(const float3& a,const float3& b) {
    float3 rv;
    rv.x = a.x-b.x;
    rv.y = a.y-b.y;
    rv.z = a.z-b.z;
    return rv;
}

void boundingBox(const CUDAmol& cmol,float3& lb,float3& ub) {
    lb.x = ub.x = cmol.atoms[0].x;
    lb.y = ub.y = cmol.atoms[0].y;
    lb.z = ub.z = cmol.atoms[0].z;
    for (uint i = 1; i < cmol.natoms; i++) {
        lb = min(cmol.atoms[i],lb);
        ub = max(cmol.atoms[i],ub);
    //fprintf(stderr,"mol bounding box: [%f,%f,%f] - [%f,%f,%f]\n",lb.x,lb.y,lb.z,ub.x,ub.y,ub.z);
    }
}

// Intermediate data representation for the input parser
class molAndTransform {
    public:
    ~molAndTransform() {
        delete[] cmol.atoms;
        delete[] dcmol.x;
        delete[] dcmol.y;
        delete[] dcmol.z;
        delete[] dcmol.a;
    }
    CUDAmol cmol;
    dCUDAmol dcmol;
    uint molid;
    float3 com;
    float transform[7];
};

struct transform {
    float xf[7];
};

list<struct transform> initializeStartingPoints(CUDAmol fitmol,CUDAmol refmol,const list<float3>& refmolRingCentroids,const list<float3>& fitmolRingCentroids);
list<struct transform> initializeStartingPoints(CUDAmol fitmol,CUDAmol refmol) {
    list<float3> emptylist;
    return initializeStartingPoints(fitmol,refmol,emptylist,emptylist);
}

CUDAmol xyzwFileToCUDAmol(const char* fn) { //{{{
    FILE* f=fopen(fn,"rt");
    if (!f) {
        printf("Could not open file %s!\n",fn);
        exit(1);
    }

    char buf[1024];
    fgets(buf,1023,f);
    CUDAmol mol;
    mol.natoms = atoi(buf);
    mol.atoms = new float4[mol.natoms];
    int nonHcount = 0;
    for (unsigned int i = 0; i < mol.natoms; i++) {
        fgets(buf,1023,f);
        sscanf(buf,"%f %f %f %f",&(mol.atoms[i].x),&(mol.atoms[i].y),&(mol.atoms[i].z),&(mol.atoms[i].w));
        if (mol.atoms[i].w != 1.20f)
            nonHcount++;
        //cout << "Read atom ("<<mol.atoms[i].x<<","<<mol.atoms[i].y<<","<<mol.atoms[i].z<<","<<mol.atoms[i].w<<")\n";
    }

    #ifdef IGNORE_HYDROGENS
    float4* nonH = new float4[nonHcount];
    for (uint i = 0,j=0; i < mol.natoms; i++) {
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
    for (uint i = 0; i < mol.natoms; i++)
        mol.atoms[i].w = 1.70f;
    #endif
    fclose(f);   
    return mol;
} //}}}

dCUDAmol xyzwFileTodCUDAmol(const char* fn) { //{{{
    FILE* f=fopen(fn,"rt");
    if (!f) {
        printf("Could not open file %s!\n",fn);
        exit(1);
    }
    char buf[1024];
    fgets(buf,1023,f);
    dCUDAmol mol;
    // These buffers may be longer than necessary if we ignore hydrogens that are in the file. No worries.
    mol.natoms = atoi(buf);
    mol.x = new float[mol.natoms];
    mol.y = new float[mol.natoms];
    mol.z = new float[mol.natoms];
    mol.a = new float[mol.natoms];

    // pi * (4pi/6sqrt(2))^-(2/3)
    const float partialalpha = 2.41798793102f;
    uint atom=0;
    for (unsigned int line = 0; line < mol.natoms; line++) {
        fgets(buf,1023,f);
        float x,y,z,rad;
        sscanf(buf,"%f %f %f %f",&x,&y,&z,&rad);
        #ifdef IGNORE_HYDROGENS
        if (rad == 1.20f) continue;
        #endif
        #ifdef ALL_CARBON_RADII
        rad = 1.70f;
        #endif  
        mol.x[atom] = x;
        mol.y[atom] = y;
        mol.z[atom] = z;
        mol.a[atom] = partialalpha/(rad*rad);

        atom++;
    }
    // atom may be less than natoms if we skipped some protons
    mol.natoms = atom;
    fclose(f);
    return mol;
} //}}}

dCUDAmol dupdCUDAmol(const dCUDAmol& smol) {
    dCUDAmol dmol;
    dmol.natoms = smol.natoms;
    dmol.x = new float[dmol.natoms];
    dmol.y = new float[dmol.natoms];
    dmol.z = new float[dmol.natoms];
    dmol.a = new float[dmol.natoms];
    memcpy(dmol.x,smol.x,dmol.natoms*sizeof(float));
    memcpy(dmol.y,smol.y,dmol.natoms*sizeof(float));
    memcpy(dmol.z,smol.z,dmol.natoms*sizeof(float));
    memcpy(dmol.a,smol.a,dmol.natoms*sizeof(float));
    return dmol;
}
CUDAmol dupCUDAmol(const CUDAmol& smol) {
    CUDAmol dmol;
    dmol.natoms = smol.natoms;
    dmol.atoms = new float4[dmol.natoms];
    memcpy(dmol.atoms,smol.atoms,dmol.natoms*sizeof(float4));
    return dmol;
}

float3 centerOfMass(const CUDAmol& mol) { //{{{
    //TODO: proper CoM calculation

    // Does a simple mean position (assumes all atoms have same weight)
    float x,y,z;
    x=mol.atoms[0].x;
    y=mol.atoms[0].y;
    z=mol.atoms[0].z;
    for (uint i = 1; i< mol.natoms; i++) {
        x+=mol.atoms[i].x;
        y+=mol.atoms[i].y;
        z+=mol.atoms[i].z;
    }
    float3 com;
    com.x = x/mol.natoms;
    com.y = y/mol.natoms;
    com.z = z/mol.natoms;
    
    return com;   
} //}}}
void remove_com(CUDAmol& mol,float3 com) { //{{{
    for (uint i = 0; i < mol.natoms; i++) {
        float4 atom = mol.atoms[i];
        mol.atoms[i] = make_float4(atom.x-com.x,atom.y-com.y,atom.z-com.z,atom.w);
    }
} //}}}
void remove_com(dCUDAmol& mol,float3 com) { //{{{
    for (uint i =0; i< mol.natoms; i++) {
        mol.x[i] -= com.x;
        mol.y[i] -= com.y;
        mol.z[i] -= com.z;
    }
} //}}}

// This function needs to change for multiconformer fit molecules
// This function also needs to change for per-molecule starting positions
list<molAndTransform*> xyzwFileToMolAndStarts(string filename,uint molid,const CUDAmol& refmol) {
    CUDAmol cmol = xyzwFileToCUDAmol(filename.c_str());
    dCUDAmol dcmol = xyzwFileTodCUDAmol(filename.c_str());
    float3 com_fit = centerOfMass(cmol);
    remove_com(cmol,com_fit);
    remove_com(dcmol,com_fit);

    /*printf("Loaded from disk fit mol id %d:\n",molid);
    for (int i =0; i < dcmol.natoms; i++) {
        printf("atom %d: [%f,%f,%f,%f]\n",i,dcmol.x[i],dcmol.y[i],dcmol.z[i],dcmol.a[i]);
    }
    */
    list<struct transform> startingPoints = initializeStartingPoints(cmol,refmol);
    list<molAndTransform*> molStarts;
    for (list<struct transform>::iterator iter = startingPoints.begin(); iter!= startingPoints.end(); iter++) {
        molAndTransform* mat = new molAndTransform;
        mat->cmol = dupCUDAmol(cmol);
        mat->dcmol = dupdCUDAmol(dcmol);
        mat->molid = molid;
        mat->com = com_fit;
        memcpy(mat->transform,iter->xf,7*sizeof(float));
        molStarts.push_back(mat);
    }

    return molStarts;
}

list<molAndTransform*> rdmolToMolAndStarts(RDKit::ROMol* molecule,uint molid,const CUDAmol& refmol,const list<float3>& refmolRingCentroids) {
    list<CUDAmol> cmolConformers;
    list<dCUDAmol> dcmolConformers;
    list<list<float3> > ringCentroids;
    molFromRDKit(molecule,cmolConformers,dcmolConformers,ringCentroids);

    CUDAmol cmol = cmolConformers.front();
    cmolConformers.pop_front();
    float3 com_fit = centerOfMass(cmol);

    dCUDAmol dcmol = dcmolConformers.front();
    dcmolConformers.pop_front();
    remove_com( cmol,com_fit);
    remove_com(dcmol,com_fit);

    list<float3> fitmolRingCentroids;
    // Add a com_fit-compensated centroid to the list of ring centroids for each one found
    for (list<float3>::iterator i = ringCentroids.front().begin(); i != ringCentroids.front().end(); i++) {
        fitmolRingCentroids.push_back(*i - com_fit);
    }
    ringCentroids.pop_front();

    /*printf("Loaded from disk fit mol id %d:\n",molid);
    for (int i =0; i < dcmol.natoms; i++) {
        printf("atom %d: [%f,%f,%f,%f]\n",i,dcmol.x[i],dcmol.y[i],dcmol.z[i],dcmol.a[i]);
    }
    */
    list<struct transform> startingPoints = initializeStartingPoints(cmol,refmol,refmolRingCentroids,fitmolRingCentroids);
    list<molAndTransform*> molStarts;
    for (list<struct transform>::iterator iter = startingPoints.begin(); iter!= startingPoints.end(); iter++) {
        molAndTransform* mat = new molAndTransform;
        mat->cmol = dupCUDAmol(cmol);
        mat->dcmol = dupdCUDAmol(dcmol);
        mat->molid = molid;
        mat->com = com_fit;
        memcpy(mat->transform,iter->xf,7*sizeof(float));
        molStarts.push_back(mat);
    }

    return molStarts;
}

void molxfsTodcMMs(list<molAndTransform*> molxflist,dCUDAMultimol& hostMM,dCUDAMultimol& devMM) {
    uint maxatoms = 0;
    uint count = molxflist.size();
    typedef list<molAndTransform*> listMAT;

    for (listMAT::iterator iter = molxflist.begin(); iter != molxflist.end(); iter++) {
        uint count = (*iter)->cmol.natoms;
        maxatoms = (count > maxatoms) ? count: maxatoms;
    }

    // Initialize basic fields
    devMM.isDeviceMM = true;
    devMM.nmols = count;
    devMM.maxatoms = maxatoms;
    hostMM.isDeviceMM = false;
    hostMM.nmols = count;
    hostMM.maxatoms = maxatoms;
    
    // Allocate device arrays to hold molecules, atom counts, molecule ids, and transforms
    // Need 4*count rows of at least maxatoms floats each
    cudaError_t cudaError;
    cudaError = cudaMallocPitch((void**)&(devMM.mols),&(devMM.pitch),maxatoms*sizeof(float),4*count);
    if (cudaError != cudaSuccess) {
        printf("Error allocating %lu bytes of device memory for fit molecules: %d\n",4*count*maxatoms*sizeof(float),cudaError);
        exit(3);
    }
    cudaError = cudaMallocPitch((void**)&(devMM.transforms),&(devMM.transform_pitch),7*sizeof(float),count);
    if (cudaError != cudaSuccess) {
        printf("Error allocating device memory for fit transforms: %d\n",cudaError);
        exit(3);
    }
    // The kernel code expects pitch to be in units of sizeof(float)
    devMM.pitch /= sizeof(float);
    devMM.transform_pitch /= sizeof(float);
    cudaError = cudaMalloc((void**)&(devMM.atomcounts),count*sizeof(uint));
    if (cudaError != cudaSuccess) {
        printf("Error allocating device memory for fit atomcounts: %d\n",cudaError);
        exit(3);
    }
    cudaError = cudaMalloc((void**)&(devMM.molids),count*sizeof(uint));
    if (cudaError != cudaSuccess) {
        printf("Error allocating device memory for fit molids: %d\n",cudaError);
        exit(3);
    }

    // Allocate host arrays with same geometry as device arrays
    hostMM.pitch = devMM.pitch;
    hostMM.transform_pitch = devMM.transform_pitch;
    hostMM.mols = (float*)malloc(4*count*hostMM.pitch*sizeof(float));
    if (cudaError != cudaSuccess) {
        printf("Error allocating host memory for fit mols: %d\n",cudaError);
        exit(3);
    }
    hostMM.transforms = (float*)malloc(count*hostMM.transform_pitch*sizeof(float));
    if (cudaError != cudaSuccess) {
        printf("Error allocating host memory for fit transforms: %d\n",cudaError);
        exit(3);
    }
    hostMM.atomcounts = (uint*)malloc(count*sizeof(uint));
    if (cudaError != cudaSuccess) {
        printf("Error allocating host memory for fit atomcounts: %d\n",cudaError);
        exit(3);
    }
    hostMM.molids = (uint*)malloc(count*sizeof(uint));
    if (cudaError != cudaSuccess) {
        printf("Error allocating host memory for fit molids: %d\n",cudaError);
        exit(3);
    }
    
    // Copy molecule data into the host arrays
    uint molidx=0;
    for (listMAT::iterator iter = molxflist.begin(); iter!=molxflist.end(); iter++, molidx++) {
        // Load coordinates and alphas
        memcpy(hostMM.mols +  4*molidx*hostMM.pitch + 0*hostMM.pitch,(*iter)->dcmol.x,(*iter)->dcmol.natoms * sizeof(float));
        memcpy(hostMM.mols +  4*molidx*hostMM.pitch + 1*hostMM.pitch,(*iter)->dcmol.y,(*iter)->dcmol.natoms * sizeof(float));
        memcpy(hostMM.mols +  4*molidx*hostMM.pitch + 2*hostMM.pitch,(*iter)->dcmol.z,(*iter)->dcmol.natoms * sizeof(float));
        memcpy(hostMM.mols +  4*molidx*hostMM.pitch + 3*hostMM.pitch,(*iter)->dcmol.a,(*iter)->dcmol.natoms * sizeof(float));
        // Load transform
        memcpy(hostMM.transforms + molidx*hostMM.transform_pitch,(*iter)->transform,7*sizeof(float));
        // Load molecule id
        hostMM.molids[molidx] = (*iter)->molid;
        // Load atom count
        hostMM.atomcounts[molidx] = (*iter)->dcmol.natoms;
        //printf("Loaded into hostMM.molids,atomcounts[%d]: %d,%d\n",molidx,(*iter)->molid,(*iter)->dcmol.natoms);
    }

    //printf("nmols = %d\n",hostMM.nmols);

    // Copy host arrays to device arrays
    cudaMemcpy(devMM.mols,hostMM.mols,4*hostMM.nmols*hostMM.pitch*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(devMM.transforms,hostMM.transforms,hostMM.nmols*hostMM.transform_pitch*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(devMM.atomcounts,hostMM.atomcounts,hostMM.nmols*sizeof(uint),cudaMemcpyHostToDevice);
    cudaMemcpy(devMM.molids,hostMM.molids,hostMM.nmols*sizeof(uint),cudaMemcpyHostToDevice);
     
}

uint molxfsTocMMs(list<molAndTransform*> molxflist,CUDAmol** fitmols) {
    uint maxatoms = 0;
    uint count = molxflist.size();
    typedef list<molAndTransform*> listMAT;

    for (listMAT::iterator iter = molxflist.begin(); iter != molxflist.end(); iter++) {
        uint count = (*iter)->cmol.natoms;
        maxatoms = (count > maxatoms) ? count: maxatoms;
    }

    (*fitmols) = (CUDAmol*)malloc(count*sizeof(CUDAmol));

    uint molidx = 0;
    for (listMAT::iterator iter = molxflist.begin(); iter!=molxflist.end(); iter++, molidx++) {
        (*fitmols)[molidx].natoms = (*iter)->cmol.natoms;
        (*fitmols)[molidx].atoms = (float4*)malloc( ((*iter)->cmol.natoms)*sizeof(float4));
        memcpy((*fitmols)[molidx].atoms,(*iter)->cmol.atoms,((*iter)->cmol.natoms)*sizeof(float4));
    }
    
    return count;

}

// Allocates host and device CUDAMultimols to fit the given list of dCUDAmols, and copies data into multimols
// Legacy code that does not initialize the transform field of the dcMM
// Use only for a reference molecule
void dCUDAmolToMultimol(dCUDAmol& mol,dCUDAMultimol& hostMM,dCUDAMultimol& devMM) { //{{{
    uint maxatoms = mol.natoms;
    uint count = 1;
    
    // Initialize basic fields
    devMM.isDeviceMM = true;
    devMM.nmols = count;
    devMM.maxatoms = maxatoms;
    devMM.transforms = 0;
    devMM.transform_pitch = 0;
    hostMM.isDeviceMM = false;
    hostMM.nmols = count;
    hostMM.maxatoms = maxatoms;
    hostMM.transforms = 0;
    hostMM.transform_pitch = 0;

    // Allocate device arrays to hold molecules and atom counts
    // Need 4*count rows of at least maxatoms floats each
    cudaMallocPitch((void**)&(devMM.mols),&(devMM.pitch),maxatoms*sizeof(float),4*count);
    // The kernel code expects pitch to be in units of sizeof(float)
    devMM.pitch /= sizeof(float);
    cudaMalloc((void**)&(devMM.atomcounts),count*sizeof(uint));
    // TODO: check for CUDA malloc error

    // Allocate host array with same geometry as device array
    hostMM.pitch = devMM.pitch;
    hostMM.mols = (float*)malloc(4*count*hostMM.pitch*sizeof(float));
    hostMM.atomcounts = (uint*)malloc(count*sizeof(uint));
    // TODO: check for malloc error

    // Copy molecules and atom counts into host array {{{
    hostMM.atomcounts[0] = mol.natoms;

    // Find the row pitch in terms of number of floats
    uint rowpitch = hostMM.pitch;

    memcpy((hostMM.mols + 0*rowpitch),mol.x,mol.natoms*sizeof(float));
    memcpy((hostMM.mols + 1*rowpitch),mol.y,mol.natoms*sizeof(float));
    memcpy((hostMM.mols + 2*rowpitch),mol.z,mol.natoms*sizeof(float));
    memcpy((hostMM.mols + 3*rowpitch),mol.a,mol.natoms*sizeof(float));
    // }}}
    
    // Copy molecules and atom counts into device array from host array
    cudaMemcpy(devMM.atomcounts,hostMM.atomcounts,count*sizeof(uint),cudaMemcpyHostToDevice);
    cudaMemcpy(devMM.mols,hostMM.mols,4*count*devMM.pitch*sizeof(float),cudaMemcpyHostToDevice);
    
    return;
} //}}}

list<struct transform> initializeStartingPoints(CUDAmol fitmol,CUDAmol refmol,const list<float3>& refmolRingCentroids,const list<float3>& fitmolRingCentroids) { //{{{
    list<struct transform> startingPoints;
    // 1 start - null
    #if INIT_MODE == 0
    {
    float points[] = {0,0,0,1,0,0,0};
    struct transform t;
    memcpy(t.xf,points,7*sizeof(float));
    startingPoints.push_back(t);
    return startingPoints;
    }
    // 4 starts - null and 180deg along each axis
    #elif INIT_MODE==1
    {
    float points[4][7] = { {0,0,0,1,0,0,0},
                         {0,0,0,0,1,0,0},
                         {0,0,0,0,0,1,0},
                         {0,0,0,0,0,0,1}
                       };
    for (int i = 0; i < 4; i++) {
        struct transform t;
        memcpy(t.xf,points[i],7*sizeof(float));
        startingPoints.push_back(t);
    }
    return startingPoints;
    }
    // 12 starts - null, 90 deg along each, 90deg along every pair
    #elif INIT_MODE==2
    {
    const float halfrt2 = 0.707106781f;
    float points[12][7] = { {0,0,0,1,0,0,0}, // null
                         {0,0,0,0,1,0,0}, // 180 deg
                         {0,0,0,0,0,1,0}, // 180 deg
                         {0,0,0,0,0,0,1}, // 180 deg
                         {0,0,0,halfrt2,halfrt2,0,0}, // 90 deg
                         {0,0,0,halfrt2,0,halfrt2,0}, // 90 deg
                         {0,0,0,halfrt2,0,0,halfrt2}, // 90 deg
                         {0,0,0,-0.5,0.5,0.5,-0.5},   // 90+90
                         {0,0,0,0.5,-0.5,0.5,-0.5},   // 90+90
                         {0,0,0,0.5,0.5,0.5,-0.5},    // 90+90
                         {0,0,0,0.5,-0.5,-0.5,-0.5},  // 90+90
                         {0,0,0,0.5,0.5,-0.5,-0.5}    // 90+90
                       };
    for (int i = 0; i < 12; i++) {
        struct transform t;
        memcpy(t.xf,points[i],7*sizeof(float));
        startingPoints.push_back(t);
    }
    return startingPoints;
    }
    // 4 rotational starts x 17 translational
    #elif INIT_MODE == 3
    {
    float points[4][7] = { {0,0,0,1,0,0,0},
                         {0,0,0,0,1,0,0},
                         {0,0,0,0,0,1,0},
                         {0,0,0,0,0,0,1}
                       };
    for (int i = 0; i < 4; i++) {
        struct transform t;
        memcpy(t.xf,points[i],7*sizeof(float));
        startingPoints.push_back(t);
    }
    list<struct transform> realStarts;
    float3 fitlb,fitub,reflb,refub;
    boundingBox(fitmol,fitlb,fitub);
    boundingBox(refmol,reflb,refub);
    //fprintf(stderr,"Fitmol bounding box: [%f,%f,%f] - [%f,%f,%f]\n",fitlb.x,fitlb.y,fitlb.z,fitub.x,fitub.y,fitub.z);
    //fprintf(stderr,"Refmol bounding box: [%f,%f,%f] - [%f,%f,%f]\n",reflb.x,reflb.y,reflb.z,refub.x,refub.y,refub.z);
    for (list<struct transform>::iterator iter = startingPoints.begin(); iter != startingPoints.end(); iter++) {
        struct transform t = *iter;
        realStarts.push_back(t);
        float x,y,z;

        x = reflb.x-fitlb.x;
        y = reflb.y-fitlb.y;
        z = reflb.z-fitlb.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = reflb.x-fitlb.x;
        y = reflb.y-fitlb.y;
        z = refub.z-fitub.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = reflb.x-fitlb.x;
        y = refub.y-fitub.y;
        z = reflb.z-fitlb.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = reflb.x-fitlb.x;
        y = refub.y-fitub.y;
        z = refub.z-fitub.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = refub.x-fitub.x;
        y = reflb.y-fitlb.y;
        z = reflb.z-fitlb.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = refub.x-fitub.x;
        y = reflb.y-fitlb.y;
        z = refub.z-fitub.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = refub.x-fitub.x;
        y = refub.y-fitub.y;
        z = reflb.z-fitlb.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = refub.x-fitub.x;
        y = refub.y-fitub.y;
        z = refub.z-fitub.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);

        x = reflb.x;
        y = reflb.y;
        z = reflb.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = reflb.x;
        y = reflb.y;
        z = refub.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = reflb.x;
        y = refub.y;
        z = reflb.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = reflb.x;
        y = refub.y;
        z = refub.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = refub.x;
        y = reflb.y;
        z = reflb.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = refub.x;
        y = reflb.y;
        z = refub.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = refub.x;
        y = refub.y;
        z = reflb.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = refub.x;
        y = refub.y;
        z = refub.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
    }
    return realStarts;
    }
    // 12 rotational starts x 17 translational
    #elif INIT_MODE == 4
    {
    const float halfrt2 = 0.707106781f;
    float points[12][7] = { {0,0,0,1,0,0,0}, // null
                         {0,0,0,0,1,0,0}, // 180 deg
                         {0,0,0,0,0,1,0}, // 180 deg
                         {0,0,0,0,0,0,1}, // 180 deg
                         {0,0,0,halfrt2,halfrt2,0,0}, // 90 deg
                         {0,0,0,halfrt2,0,halfrt2,0}, // 90 deg
                         {0,0,0,halfrt2,0,0,halfrt2}, // 90 deg
                         {0,0,0,-0.5,0.5,0.5,-0.5},   // 90+90
                         {0,0,0,0.5,-0.5,0.5,-0.5},   // 90+90
                         {0,0,0,0.5,0.5,0.5,-0.5},    // 90+90
                         {0,0,0,0.5,-0.5,-0.5,-0.5},  // 90+90
                         {0,0,0,0.5,0.5,-0.5,-0.5}    // 90+90
                       };
    for (int i = 0; i < 12; i++) {
        struct transform t;
        memcpy(t.xf,points[i],7*sizeof(float));
        startingPoints.push_back(t);
    }
    list<struct transform> realStarts;
    float3 fitlb,fitub,reflb,refub;
    boundingBox(fitmol,fitlb,fitub);
    boundingBox(refmol,reflb,refub);
    for (list<struct transform>::iterator iter = startingPoints.begin(); iter != startingPoints.end(); iter++) {
        struct transform t = *iter;
        realStarts.push_back(t);
        float x,y,z;

        x = reflb.x-fitlb.x;
        y = reflb.y-fitlb.y;
        z = reflb.z-fitlb.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = reflb.x-fitlb.x;
        y = reflb.y-fitlb.y;
        z = refub.z-fitub.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = reflb.x-fitlb.x;
        y = refub.y-fitub.y;
        z = reflb.z-fitlb.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = reflb.x-fitlb.x;
        y = refub.y-fitub.y;
        z = refub.z-fitub.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = refub.x-fitub.x;
        y = reflb.y-fitlb.y;
        z = reflb.z-fitlb.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = refub.x-fitub.x;
        y = reflb.y-fitlb.y;
        z = refub.z-fitub.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = refub.x-fitub.x;
        y = refub.y-fitub.y;
        z = reflb.z-fitlb.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = refub.x-fitub.x;
        y = refub.y-fitub.y;
        z = refub.z-fitub.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);

        x = reflb.x;
        y = reflb.y;
        z = reflb.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = reflb.x;
        y = reflb.y;
        z = refub.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = reflb.x;
        y = refub.y;
        z = reflb.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = reflb.x;
        y = refub.y;
        z = refub.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = refub.x;
        y = reflb.y;
        z = reflb.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = refub.x;
        y = reflb.y;
        z = refub.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = refub.x;
        y = refub.y;
        z = reflb.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
        
        x = refub.x;
        y = refub.y;
        z = refub.z;
        t.xf[0] = x; t.xf[1] = y; t.xf[2] = z;
        realStarts.push_back(t);
    }
    return realStarts;
    }
    // 30 random starts
    #elif INIT_MODE == 5
    {
    float3 fitlb,fitub,reflb,refub;
    boundingBox(fitmol,fitlb,fitub);
    boundingBox(refmol,reflb,refub);
    list<struct transform> realStarts;
    const int numStarts = 30;
    float nullpt[] = {0,0,0,1,0,0,0};
    struct transform t;
    memcpy(t.xf,nullpt,7*sizeof(float));
    realStarts.push_back(t);
    srand(10);
    for (int i = 0; i < numStarts; i++) {
        float temp  = (rand()/((float)RAND_MAX)) - 0.5;
        t.xf[0] = temp*refub.x;
        temp  = (rand()/((float)RAND_MAX)) - 0.5;
        t.xf[1] = temp*refub.y;
        temp  = (rand()/((float)RAND_MAX)) - 0.5;
        t.xf[2] = temp*refub.z;
        temp  = (rand()/((float)RAND_MAX))*3.141592654f;
        float temp2  = (rand()/((float)RAND_MAX))*3.141592654f;
        float temp3  = (rand()/((float)RAND_MAX))*3.141592654f*2;
        t.xf[3] = cosf(temp);
        t.xf[4] = sinf(temp)*cosf(temp2);
        t.xf[5] = sinf(temp)*sinf(temp2)*cosf(temp3);
        t.xf[6] = sinf(temp)*sinf(temp2)*sinf(temp3);
        realStarts.push_back(t);
    }
    return realStarts;
    }
    // 12 random starts
    #elif INIT_MODE == 6
    {
    float3 fitlb,fitub,reflb,refub;
    boundingBox(fitmol,fitlb,fitub);
    boundingBox(refmol,reflb,refub);
    list<struct transform> realStarts;
    const int numStarts = 12;
    float nullpt[] = {0,0,0,1,0,0,0};
    struct transform t;
    memcpy(t.xf,nullpt,7*sizeof(float));
    realStarts.push_back(t);
    srand(10);
    for (uint i = 0; i < numStarts; i++) {
        float temp  = (rand()/((float)RAND_MAX)) - 0.5;
        t.xf[0] = temp*refub.x;
        temp  = (rand()/((float)RAND_MAX)) - 0.5;
        t.xf[1] = temp*refub.y;
        temp  = (rand()/((float)RAND_MAX)) - 0.5;
        t.xf[2] = temp*refub.z;
        temp  = (rand()/((float)RAND_MAX))*3.141592654f;
        float temp2  = (rand()/((float)RAND_MAX))*3.141592654f;
        float temp3  = (rand()/((float)RAND_MAX))*3.141592654f*2;
        t.xf[3] = cosf(temp);
        t.xf[4] = sinf(temp)*cosf(temp2);
        t.xf[5] = sinf(temp)*sinf(temp2)*cosf(temp3);
        t.xf[6] = sinf(temp)*sinf(temp2)*sinf(temp3);
        realStarts.push_back(t);
    }
    return realStarts;
    }
    // 4 rotational starts x translation onto ring centroids
    #elif INIT_MODE == 7
    {
    float points[4][7] = { {0,0,0,1,0,0,0},
                         {0,0,0,0,1,0,0},
                         {0,0,0,0,0,1,0},
                         {0,0,0,0,0,0,1}
                       };
    for (int i = 0; i < 4; i++) {
        struct transform t;
        memcpy(t.xf,points[i],7*sizeof(float));
        startingPoints.push_back(t);
    }
    list<struct transform> realStarts;
    for (list<struct transform>::iterator rotiter = startingPoints.begin(); rotiter != startingPoints.end(); rotiter++) {
        realStarts.push_back(*rotiter);
        struct transform t = *rotiter;
        for (list<float3>::const_iterator fitcent = fitmolRingCentroids.begin(); fitcent != fitmolRingCentroids.end(); fitcent++) {
            // Find the coordinates of the fitmol ring centroid under the starting transformation
            float3 transformedCenter = transformSinglePoint(rotiter->xf,*fitcent);
            for (list<float3>::const_iterator refcent = refmolRingCentroids.begin(); refcent != refmolRingCentroids.end(); refcent++) {
               float3 offset = *refcent - transformedCenter;
               t.xf[0] = offset.x;
               t.xf[1] = offset.y;
               t.xf[2] = offset.z;
               realStarts.push_back(t); 
            }
        }
    }
    return realStarts;
    }
    // 12 rotational starts x translation onto ring centroids
    #elif INIT_MODE == 8
    {
    const float halfrt2 = 0.707106781f;
    float points[12][7] = { {0,0,0,1,0,0,0}, // null
                         {0,0,0,0,1,0,0}, // 180 deg
                         {0,0,0,0,0,1,0}, // 180 deg
                         {0,0,0,0,0,0,1}, // 180 deg
                         {0,0,0,halfrt2,halfrt2,0,0}, // 90 deg
                         {0,0,0,halfrt2,0,halfrt2,0}, // 90 deg
                         {0,0,0,halfrt2,0,0,halfrt2}, // 90 deg
                         {0,0,0,-0.5,0.5,0.5,-0.5},   // 90+90
                         {0,0,0,0.5,-0.5,0.5,-0.5},   // 90+90
                         {0,0,0,0.5,0.5,0.5,-0.5},    // 90+90
                         {0,0,0,0.5,-0.5,-0.5,-0.5},  // 90+90
                         {0,0,0,0.5,0.5,-0.5,-0.5}    // 90+90
                       };
    for (int i = 0; i < 12; i++) {
        struct transform t;
        memcpy(t.xf,points[i],7*sizeof(float));
        startingPoints.push_back(t);
    }
    list<struct transform> realStarts;
    for (list<struct transform>::iterator rotiter = startingPoints.begin(); rotiter != startingPoints.end(); rotiter++) {
        realStarts.push_back(*rotiter);
        struct transform t = *rotiter;
        for (list<float3>::const_iterator fitcent = fitmolRingCentroids.begin(); fitcent != fitmolRingCentroids.end(); fitcent++) {
            // Find the coordinates of the fitmol ring centroid under the starting transformation
            float3 transformedCenter = transformSinglePoint(rotiter->xf,*fitcent);
            for (list<float3>::const_iterator refcent = refmolRingCentroids.begin(); refcent != refmolRingCentroids.end(); refcent++) {
               float3 offset = *refcent - transformedCenter;
               t.xf[0] = offset.x;
               t.xf[1] = offset.y;
               t.xf[2] = offset.z;
               realStarts.push_back(t); 
            }
        }
    }
    return realStarts;
    }
    #endif


} //}}}

extern "C" void loadMoleculesRDKit(int num_mols,list<RDKit::ROMol*>& molecules,
                                   CUDAmol** fitmols,CUDAmol& refmol,uint** molids,float** transforms,size_t& transform_pitch,
                                   dCUDAMultimol& hostFitMM,dCUDAMultimol& devFitMM,
                                   dCUDAMultimol& hostRefMM,dCUDAMultimol& devRefMM,
                                   float3& com_ref,float3** com_fit,
                                   uint& totalMols,uint& distinctMols)
{
    if (num_mols < 2) {
        std::cerr << "Error: not enough molecules are specified." << std::endl;
        return;
    }
    RDKit::ROMol* refrdmol;
    list<RDKit::ROMol*> fitrdmols;
    refrdmol = molecules.front();
    list<RDKit::ROMol*>::iterator it = molecules.begin();
    std::advance(it, 1);
    for (; it != molecules.end(); ++it)
    {
        fitrdmols.push_back(*it);
    }
    // Here we have the reference molecule in refrdmol
    // And the fit molecules in fitrdmols
    if (fitrdmols.empty()) {
        printf("Error - no fit molecules specified!\n");
        exit(2);
    }
    list<molAndTransform*> molxfList;

    // This section will need to be changed for multiconformer references
    list<CUDAmol> cmolConformers;
    list<dCUDAmol> dcmolConformers;
    list<list<float3> > ringCentroids;
    molFromRDKit(refrdmol,cmolConformers,dcmolConformers,ringCentroids);

    refmol = cmolConformers.front();
    cmolConformers.pop_front();
    com_ref = centerOfMass(refmol);
    remove_com(refmol,com_ref);

    dCUDAmol refdmol = dcmolConformers.front();
    dcmolConformers.pop_front();
    remove_com(refdmol,com_ref);

    dCUDAmolToMultimol(refdmol,hostRefMM,devRefMM);
    delete[] refdmol.x; delete[] refdmol.y; delete[] refdmol.z; delete[] refdmol.a;
    list<float3> refmolRingCentroids;
    // Add a com_ref-compensated centroid to the list of ring centroids for each one found
    for (list<float3>::iterator i = ringCentroids.front().begin(); i != ringCentroids.front().end(); i++) {
        refmolRingCentroids.push_back(*i - com_ref);
    }
    ringCentroids.pop_front();
    // Done setting up the reference molecules

    totalMols = 0;
    distinctMols = fitrdmols.size();
    uint molid=0;
    for (list<RDKit::ROMol*>::iterator iter = fitrdmols.begin(); iter != fitrdmols.end(); iter++,molid++) {
        list<molAndTransform*> molStarts = rdmolToMolAndStarts(*iter,molid,refmol,refmolRingCentroids);
        totalMols += molStarts.size();
        molxfList.splice(molxfList.end(),molStarts);
    }

    // Load up the centers of mass and molecule ids for the host
    (*com_fit) =  (float3*)malloc(totalMols*sizeof(float3));
    (*molids)  =  (uint*)malloc(totalMols*sizeof(uint));
    uint i = 0;
    for (list<molAndTransform*>::iterator iter = molxfList.begin(); iter != molxfList.end(); iter++,i++) {
        (*com_fit)[i] = (*iter)->com;
        (*molids)[i] = (*iter)->molid;
    }

    molxfsTodcMMs(molxfList,hostFitMM,devFitMM);
    transform_pitch = hostFitMM.transform_pitch;
    (*transforms) = (float*)malloc(totalMols*transform_pitch*sizeof(float));
    memcpy(*transforms,hostFitMM.transforms,totalMols*transform_pitch*sizeof(float));
    molxfsTocMMs(molxfList,fitmols);


    // Clean up the elements in the list
    for (list<molAndTransform*>::iterator iter = molxfList.begin(); iter != molxfList.end();iter++) {
        delete (*iter);
    }

}

