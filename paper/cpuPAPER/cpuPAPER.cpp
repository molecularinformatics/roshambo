/*
 * cpuPAPER/cpuPAPER.cpp
 * Main program file for cpuPAPER
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
#include "inputModule.h"
#include "hostVolume.h"
//#define HYPERSPHERE_WALK
#include "nrOptimization.cpp"
//#include "oeInterface.cpp"
#include <iostream>
#include <cstring>
#include <cstdio>
#include <sys/time.h>
using namespace std;

CUDAmol nrfitmol,nrrefmol;
Grid nrrefgrid,nrfitgrid;
float nrmargin,nrrefvol,nrfitvol;

#include "config.h"
double getustime(void) { // {{{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    double t = tv.tv_sec*1e6 + tv.tv_usec;
    return t;
} //}}}
/*
CUDAmol xyzwFileToCUDAmol(char* fn) {
    FILE* f=fopen(fn,"rt");
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
*/
inline float4 operator-(const float4& a,const float3& b) {
    float4 temp;
    temp.x = a.x-b.x;
    temp.y = a.y-b.y;
    temp.z = a.z-b.z;
    temp.w = a.w;
    return temp;
}
inline float3 operator-(const float3& a,const float3& b) {
	return make_float3(a.x-b.x,a.y-b.y,a.z-b.z);
}
inline float3 operator+(const float3& a,const float3& b) {
	return make_float3(a.x+b.x,a.y+b.y,a.z+b.z);
}

void nrAnalyticGradient(float* transform,float* gradient);
float nrAnalyticObjective(float* transform);

void affineQuatToMatrix(const float* quat,float* matrix);
void printTransformMatrix(const float* matrix);

void inertialOverlay(CUDAmol& refmol,CUDAmol& fitmol,float3& com_ref,float3& com_fit);
float3 centerOfMass(const CUDAmol& mol);

void composeTransform(float* a,const float* b);

void optimize(CUDAmol& refmol,CUDAmol* fitmols,float* transforms,float* overlaps,uint totalMols);

int main(int argc,char** argv) {
    if (argc < 2) {
        cout << "cpuPAPER [reference file] [fit files]\n";
        cout << "or\n";
        cout << "cpuPAPER [listing file]\n";
        return 1;
    }
    CUDAmol refmol,*fitmols;
    uint* molids;
    float* transforms;
    float3 com_ref;
    float3* com_fits;
    uint totalMols,distinctMols;
    loadMolecules(argc,argv,&fitmols,refmol,&molids,&transforms,com_ref,&com_fits,totalMols,distinctMols);
    char outfn[256];
    FILE* outf;
    #if 0
    for (uint i = 0; i < totalMols; i++) {
        printf("Molstart %u:\n",i);
        CUDAmol finalmol;
        transformCUDAmolToHost(fitmols[i],finalmol,transforms+i*7,0);
        sprintf(outfn,"diag/fit_cpu_start_%u.xyzw",i);
        outf = fopen(outfn,"wt");
        fprintf(outf,"%d\n",finalmol.natoms);
        printf("finalmol.natoms = %u, fitmol.natoms = %u\n",finalmol.natoms,fitmols[i].natoms);
        for (uint j = 0; j < finalmol.natoms; j++) {
            fprintf(outf,"%f %f %f %f\n",finalmol.atoms[j].x,finalmol.atoms[j].y,finalmol.atoms[j].z,finalmol.atoms[j].w);
            printf("\t%u (%f %f %f %f) -> ",j,fitmols[i].atoms[j].x,fitmols[i].atoms[j].y,fitmols[i].atoms[j].z,fitmols[i].atoms[j].w);
            printf("(%f %f %f %f)\n",finalmol.atoms[j].x,finalmol.atoms[j].y,finalmol.atoms[j].z,finalmol.atoms[j].w);
        }
        fclose(outf);
        delete[] finalmol.atoms;
    }
    #endif
    float* overlaps = new float[totalMols];

    bool benchmark = true; 
    if (benchmark) { //{{{
        uint bench_runs = 10;
        double start = getustime();
        // Benchmarking iteration: run optimize kernel
        // TODO: copy molecules over too
        for (uint i = 0; i < bench_runs; i++) {
            optimize(refmol,fitmols,transforms,overlaps,totalMols);
        }
        double end = getustime();
        double runtime = ((end-start)/1000)/bench_runs;
        printf("Benchmark results over %d iterations on %d molecules (%d mol/starts): %f ms/batch optimization, %f ms/molecule, %f ms/position\n",bench_runs,distinctMols,totalMols,runtime,runtime/distinctMols,runtime/totalMols);
    } //}}}

    #if 0
    // Optimize from starting positions
    optimize(refmol,fitmols,transforms,overlaps,totalMols);
    // Max-reduce over each molecule
    uint* bestidx = new uint[distinctMols];
    float* bestOvl = new float[distinctMols];
    for (uint i = 0; i < distinctMols; i++)
        bestOvl[i] = -1e9f;
    for (uint i = 0; i < totalMols; i++) {
        printf("Molstart %d: Final overlap %f, Final transform: [%f,%f,%f , %f,%f,%f,%f]\n",i,overlaps[i],transforms[i*7+0],transforms[i*7+1],transforms[i*7+2],transforms[i*7+3],transforms[i*7+4],transforms[i*7+5],transforms[i*7+6]);
        uint id = molids[i];
        if (overlaps[i] > bestOvl[id]) {
            bestidx[id] = i;
            bestOvl[id] = overlaps[i];
        }
    }

    // Derive and print transformation matrix for each molecule
    for (uint id = 0; id < distinctMols; id++) {
        float matrix[16];
        uint idx = bestidx[id];
	    affineQuatToMatrix(transforms+7*idx,matrix);
    
        // Modify this transformation matrix so that center-of-mass shifts are invisible to the user
        // Initialize translation matrix to the identity
        float translation[16];
        memset(translation,0,16*sizeof(float));
        translation[0*4+0] = 1;
        translation[1*4+1] = 1;
        translation[2*4+2] = 1;
        translation[3*4+3] = 1;
        // Compose the transforms as follows:
        // <----------
        // move to ref_com * optimized transform * move to fit_com
        
        // Set translation to -com_fit
        translation[0*4+3] = -com_fits[idx].x;
        translation[1*4+3] = -com_fits[idx].y;
        translation[2*4+3] = -com_fits[idx].z;
        
        composeTransform(matrix,translation);
        
        // Set translation to com_ref
        translation[0*4+3] = com_ref.x;
        translation[1*4+3] = com_ref.y;
        translation[2*4+3] = com_ref.z;
        
        composeTransform(translation,matrix);
        
        memcpy(matrix,translation,16*sizeof(float));
        printTransformMatrix(matrix);
    }


	if (true) {
    #if 1 
        char outfn[256];
        FILE* outf;
        for (uint i = 0; i < totalMols; i++) {
            CUDAmol finalmol;
            transformCUDAmolToHost(fitmols[i],finalmol,transforms+i*7,0);
            sprintf(outfn,"diag/fit_cpu_%u.xyzw",i);
            outf = fopen(outfn,"wt");
            fprintf(outf,"%d\n",finalmol.natoms);
            for (uint i = 0; i < finalmol.natoms; i++) {
                fprintf(outf,"%f %f %f %f\n",finalmol.atoms[i].x,finalmol.atoms[i].y,finalmol.atoms[i].z,finalmol.atoms[i].w);
            }
            fclose(outf);
            delete[] finalmol.atoms;
        }
        
        sprintf(outfn,"diag/ref_cpu.xyzw");
        outf = fopen(outfn,"wt");
        fprintf(outf,"%d\n",refmol.natoms);
        for (uint i = 0; i < refmol.natoms; i++) {
            fprintf(outf,"%f %f %f %f\n",refmol.atoms[i].x,refmol.atoms[i].y,refmol.atoms[i].z,refmol.atoms[i].w);
        }
        fclose(outf);
    #endif
    }
    #endif
    /*if (refmol.natoms == finalmol.natoms) {
        float rmsd = 0.0;
        for (unsigned int i = 0; i < refmol.natoms; i++) {
            float4 r = refmol.atoms[i];
            float4 f = finalmol.atoms[i];
            float sd = (r.x-f.x)*(r.x-f.x) + (r.y-f.y)*(r.y-f.y) + (r.z-f.z)*(r.z-f.z);
            rmsd += sd;
    #ifdef DEBUG_OUTPUT
            cout << "SD for atom "<<i<<": "<<sd<<"( ref: ["<<r.x<<","<<r.y<<","<<r.z<<"], fit: ["<<f.x<<","<<f.y<<","<<f.z<<"])"<<endl;
            cout << "Deviation vector: ["<<(r.x-f.x)<<","<<(r.y-f.y)<<","<<(r.z-f.z)<<"]"<<endl<<endl;
    #endif
        }
        rmsd /= refmol.natoms;
        rmsd = sqrtf(rmsd);
        //cout << "RMSD: "<<rmsd<<" A\n";
    }*/
    
    return 0;
    
}

void optimize(CUDAmol& refmol,CUDAmol* fitmols,float* transforms,float* overlaps,uint totalMols) {    
    for (uint i = 0; i < totalMols; i++) {
        nrfitmol = fitmols[i];
        nrrefmol = refmol;
        int nIters;

        float objective;
        float* transform = transforms + 7*i;
        
        /*float matrix[16];
        affineQuatToMatrix(transform,matrix);
        printTransformMatrix(matrix);*/
		
		// Ending numerical parms : STPMX, ITMAX, EPS, TOLX
        //printf("Molstart %d: Starting transform: [%f,%f,%f , %f,%f,%f,%f]\n",i,transforms[i*7+0],transforms[i*7+1],transforms[i*7+2],transforms[i*7+3],transforms[i*7+4],transforms[i*7+5],transforms[i*7+6]);

        // dfpmin is a BFGS-type optimizer
        // Parameters:
        //      - transform: starting position for optimization (and final coordinates)
        //      - 0.01: convergence tolerance for zeroing the gradient
        //      - nIters: returns number of optimization iterations performed
        //      - objective: returns value achieved of objective function
        //      - nrAnalyticObjective: function defined below to evaluate the objective function (negative of overlap volume)
        //      - nrAnalyticGradient: function defined below to evaluate gradient of objective function
		dfpmin<7>(transform,0.01,nIters,objective,nrAnalyticObjective,nrAnalyticGradient);
        overlaps[i] = -objective;
        
        //CUDAmol finalmol;
        //transformCUDAmolToHost(fitmols[i],finalmol,transform,0);
        //Grid nullgrid;
        //overlaps[i]=getOverlapVolume(refmol,finalmol,nullgrid);
        //printf("Molstart %d: ending overlap: %f, ending transform: [%f,%f,%f , %f,%f,%f,%f]\n",i,overlaps[i],transforms[i*7+0],transforms[i*7+1],transforms[i*7+2],transforms[i*7+3],transforms[i*7+4],transforms[i*7+5],transforms[i*7+6]);
        //delete[] finalmol.atoms;
    }
    return;
}
        
    
// Performs a=a*b for 4x4 matrices
void composeTransform(float* a,const float* b) {
    float m[4][4];
    for (int i = 0; i < 4; i++) {
        for ( int j = 0; j < 4; j++) {
            m[i][j] = 0;
            for ( int k = 0; k < 4; k++) {
                m[i][j] += a[i*4+k]*b[k*4+j];
            }
        }
    }
    memcpy(a,m,16*sizeof(float));
    return;
}


void affineQuatToMatrix(const float* quat,float* matrix) {
	// Elements of quaternion are x,e,n,s
    float e2 = quat[4]*quat[4];
    float n2 = quat[5]*quat[5];
    float s2 = quat[6]*quat[6];
    float x2 = quat[3]*quat[3];
	float invmagsqr = 1.0f/(x2+e2+n2+s2);
    float xe = quat[3]*quat[4];
    float ne = quat[5]*quat[4];
    float xs = quat[3]*quat[6];
    float es = quat[4]*quat[6];
    float nx = quat[3]*quat[5];
    float ns = quat[5]*quat[6];

    float scaledrot[3][3] = { e2-n2-s2+x2,  2*(ne+xs),    2*(es-nx),
                      2*(ne-xs),    -e2+n2-s2+x2, 2*(ns+xe),
                      2*(es+nx),    2*(ns-xe),    -e2-n2+s2+x2};
	
	// Copy scaled rotation matrix to transformation matrix, scaling for quat length
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			matrix[i*4 + j] = scaledrot[i][j]*invmagsqr;
		}
	}
    #ifdef DEBUG_OUTPUT
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            //if (i == j) continue;
            float dot = 0;
            for (int k = 0 ; k < 3; k++) {
                //printf("\t component: %f * %f=%f\n",matrix[k*4+j],matrix[k*4+i],matrix[k*4+j]*matrix[k*4+i]);

                dot += matrix[k*4+j]*matrix[k*4+i];
            }
            printf("Dot product of rows %d, %d = %f\n",j,i,dot);
        }
    }
    #endif
	// Copy the translation
	for (int i = 0; i < 3; i++) {
        //printf("in aQTM: writing to %d\n",i*3+3);
		matrix[i*4 + 3] = quat[i];
	}
	
	// Set projective part of matrix to the identity
	matrix[12] = matrix[13] = matrix[14] = 0;
	matrix[15] = 1;
	
	return;
}
	
void printTransformMatrix(const float* matrix) {
	printf("[[ %9.6f %9.6f %9.6f %9.6f\n   %9.6f %9.6f %9.6f %9.6f\n   %9.6f %9.6f %9.6f %9.6f\n   %9.6f %9.6f %9.6f %9.6f]]\n",
		   matrix[0],matrix[1],matrix[2],matrix[3],matrix[4],matrix[5],matrix[6],matrix[7],matrix[8],matrix[9],
		   matrix[10],matrix[11],matrix[12],matrix[13],matrix[14],matrix[15]);
}


float nrAnalyticObjective(float* transform) {
    CUDAmol newmol;
    transformCUDAmolToHost(nrfitmol,newmol,transform,0);
        
	// debug output block
    #ifdef DEBUG_OUTPUT
	static int callctr = 0;
	if (true) {
		char outfn[256];
		sprintf(outfn,"diag/fit_%d.xyzw",callctr);
		FILE* outf = fopen(outfn,"wt");
		fprintf(outf,"%d\n",newmol.natoms);
		for (int i = 0; i < newmol.natoms; i++) {
			fprintf(outf,"%f %f %f %f\n",newmol.atoms[i].x,newmol.atoms[i].y,newmol.atoms[i].z,newmol.atoms[i].w);
		}
		fclose(outf);
		callctr++;
	}
    #endif

    // TODO: remove this grid
    Grid nullgrid;
    
    // Negate overlap volume - more overlap is better
    float objective = -1 * getOverlapVolume(nrrefmol,newmol,nullgrid);

    // Quaternion penalty
    float magp = sqrtf(transform[3]*transform[3]+transform[4]*transform[4]+transform[5]*transform[5]+transform[6]*transform[6]);
    const float gamma = PENALTY_GAMMA;
    float penalty = 0.5*gamma*(magp-1)*(magp-1);
    objective+=penalty;
    #ifdef DEBUG_OUTPUT
    cerr << "In nrObjective: transform=["<<transform[0]<<","<<transform[1]<<","<<transform[2]<<"],["<<transform[3]<<","<<transform[4]<<","<<transform[5]<<","<<transform[6]<<"], objective="<<objective<<", penalty = "<<penalty<<endl;
    #endif
    delete[] newmol.atoms;
    return objective;
}

void nrAnalyticGradient(float* transform, float* gradient) {
    CUDAmol newmol;
    transformCUDAmolToHost(nrfitmol,newmol,transform,0);

    // TODO: remove this grid
    Grid nullgrid;
    
    getGradient(nrrefmol,newmol,nullgrid,transform,gradient);
    #ifdef DEBUG_OUTPUT
    cout << endl;
    cout << "In nrGradient: transform=["<<transform[0]<<","<<transform[1]<<","<<transform[2]<<"],["<<transform[3]<<","<<transform[4]<<","<<transform[5]<<","<<transform[6]<<"]"<<endl;
    cout << "In nrGradient: gradient=["<<gradient[0]<<","<<gradient[1]<<","<<gradient[2]<<"],["<<gradient[3]<<","<<gradient[4]<<","<<gradient[5]<<","<<gradient[6]<<"]"<<endl;
    cout << endl;
    #endif
    delete[] newmol.atoms;
    return;
}
