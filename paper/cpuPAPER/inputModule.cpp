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
    }
    CUDAmol cmol;
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
    for (int i = 0; i < mol.natoms; i++)
        mol.atoms[i].w = 1.70f;
    #endif
    fclose(f);   
    return mol;
} //}}}

CUDAmol dupCUDAmol(const CUDAmol& smol) {
    CUDAmol dmol;
    dmol.natoms = smol.natoms;
    //dmol.atoms = (float4*)malloc(dmol.natoms * sizeof(float4));
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
        mol.atoms[i].x = atom.x - com.x;
        mol.atoms[i].y = atom.y - com.y;
        mol.atoms[i].z = atom.z - com.z;
        mol.atoms[i].w = atom.w;
    }
} //}}}

// This function needs to change for multiconformer fit molecules
// This function also needs to change for per-molecule starting positions
list<molAndTransform*> xyzwFileToMolAndStarts(string filename,uint molid,const CUDAmol& refmol) {
    CUDAmol cmol = xyzwFileToCUDAmol(filename.c_str());
    float3 com_fit = centerOfMass(cmol);
    remove_com(cmol,com_fit);

    list<struct transform> startingPoints = initializeStartingPoints(cmol,refmol);
    list<molAndTransform*> molStarts;
    for (list<struct transform>::iterator iter = startingPoints.begin(); iter!= startingPoints.end(); iter++) {
        molAndTransform* mat = new molAndTransform;
        mat->cmol = dupCUDAmol(cmol);
        mat->molid = molid;
        mat->com = com_fit;
        memcpy(mat->transform,iter->xf,7*sizeof(float));
        molStarts.push_back(mat);
    }

    return molStarts;
}

list<molAndTransform*> molFileToMolAndStarts(string filename,uint molid,const CUDAmol& refmol,const list<float3>& refmolRingCentroids) {
    list<CUDAmol> cmolConformers;
    list<list<float3> > ringCentroids;
    molFromFile(filename,cmolConformers,ringCentroids);

    CUDAmol cmol = cmolConformers.front();
    cmolConformers.pop_front();
    float3 com_fit = centerOfMass(cmol);
    
    remove_com( cmol,com_fit);

    list<float3> fitmolRingCentroids;
    // Add a com_fit-compensated centroid to the list of ring centroids for each one found
    for (list<float3>::iterator i = ringCentroids.front().begin(); i != ringCentroids.front().end(); i++) {
        fitmolRingCentroids.push_back(*i - com_fit);
    }
    ringCentroids.pop_front();

    list<struct transform> startingPoints = initializeStartingPoints(cmol,refmol,refmolRingCentroids,fitmolRingCentroids);
    list<molAndTransform*> molStarts;
    for (list<struct transform>::iterator iter = startingPoints.begin(); iter!= startingPoints.end(); iter++) {
        molAndTransform* mat = new molAndTransform;
        mat->cmol = dupCUDAmol(cmol);
        mat->molid = molid;
        mat->com = com_fit;
        memcpy(mat->transform,iter->xf,7*sizeof(float));
        molStarts.push_back(mat);
    }

    return molStarts;
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

extern "C" void loadMolecules(int argc,char** argv,
                              CUDAmol** fitmols,CUDAmol& refmol,uint** molids,float** transforms,
                              float3& com_ref,float3** com_fit,
                              uint& totalMols,uint& distinctMols) 
{
    string refmolfile;
    list<string> fitmolfiles;
    if (argc > 2) {
        refmolfile = string(argv[1]);
        // Molecules were listed on the command line
        for (int i = 2; i < argc; i++) 
            fitmolfiles.push_back(string(argv[i]));
    } else {
        // Command line just had a file parameter; molecules listed in there.
        bool isref = true;
        FILE* listfile = fopen(argv[1],"rt");
        if (!listfile) {
            printf("Error: could not open file %s\n",argv[1]);
            exit(1);
        }
        char buf[1025];
        while (!feof(listfile)) {
            // Add some test code for eof
            int ch = fgetc(listfile);
            if (ch != EOF) {
                ungetc(ch,listfile);
            } else break;

            fgets(buf,1024,listfile);
            // Nuke the newline
            char* nl = strchr(buf,'\n');
            if (nl)
                *nl = '\0';
            if (isref) {
                refmolfile = string(buf);
                isref = false;
            } else {
                //printf("Adding fit molecule file %s\n",buf);
                fitmolfiles.push_back(string(buf));
            }
        }
        fclose(listfile);
    }
    
    // Here we have the reference molecule file name in refmolfile
    // And the fit molecules in fitmolfiles
    if (fitmolfiles.empty()) {
        printf("Error - no fit molecules specified!\n");
        exit(2);
    }

    list<molAndTransform*> molxfList;

    // This section will need to be changed for multiconformer references
    list<CUDAmol> cmolConformers;
    list<list<float3> > ringCentroids;
    molFromFile(refmolfile,cmolConformers,ringCentroids);

    refmol = cmolConformers.front();
    cmolConformers.pop_front();
    com_ref = centerOfMass(refmol);
    remove_com(refmol,com_ref);
    
    list<float3> refmolRingCentroids;
    // Add a com_ref-compensated centroid to the list of ring centroids for each one found
    for (list<float3>::iterator i = ringCentroids.front().begin(); i != ringCentroids.front().end(); i++) {
        refmolRingCentroids.push_back(*i - com_ref);
    }
    ringCentroids.pop_front();
    // Done setting up the reference molecules
    
    totalMols = 0;
    distinctMols = fitmolfiles.size();
    uint molid=0;
    for (list<string>::iterator iter = fitmolfiles.begin(); iter != fitmolfiles.end(); iter++,molid++) {
        list<molAndTransform*> molStarts = molFileToMolAndStarts(*iter,molid,refmol,refmolRingCentroids);
        totalMols += molStarts.size();
        molxfList.splice(molxfList.end(),molStarts);
    }

    // Load up the centers of mass, transforms, and molecule ids for the host
    (*com_fit) =  (float3*)malloc(totalMols*sizeof(float3));
    (*molids)  =  (uint*)malloc(totalMols*sizeof(uint));
    (*transforms) = (float*)malloc(totalMols*7*sizeof(float));
    uint i = 0;
    for (list<molAndTransform*>::iterator iter = molxfList.begin(); iter != molxfList.end(); iter++,i++) {
        (*com_fit)[i] = (*iter)->com;
        (*molids)[i] = (*iter)->molid;
        memcpy(*transforms+7*i,(*iter)->transform,7*sizeof(float));
    }

    molxfsTocMMs(molxfList,fitmols);

    
    // Clean up the elements in the list
    for (list<molAndTransform*>::iterator iter = molxfList.begin(); iter != molxfList.end();iter++) {
        delete (*iter);
    }

}
