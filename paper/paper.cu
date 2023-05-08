/*
 * paper.cu
 * Main program file for PAPER
 *   - loads files from disk using inputModule
 *   - optimizes overlap with deviceOverlay
 *   - outputs transformation matrices for max overlap
 *
 * Author: Imran Haque, 2010
 * Copyright 2009-2010, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */
#include <stdio.h>
#include <list>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include "inputModule.h"
#include "cudaVolumeTypes.h"
#include "hostAnalyticVolume.h"
#include "deviceAnalyticVolume.h"
#include "deviceOverlay.h"
#include "transformTools.h"
#include <sys/time.h>
#include <cuda_runtime.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <iostream>

using namespace std;

#define ELTS7(x) x[0],x[1],x[2],x[3],x[4],x[5],x[6]

double getustime(void) { // {{{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    double t = tv.tv_sec*1e6 + tv.tv_usec;
    return t;
} //}}}

extern "C" float** paper(int gpuID, list<RDKit::ROMol*>& molecules) {
    int num_mols = 0;
    num_mols = molecules.size();

    cudaSetDevice(gpuID);
    fprintf(stderr,"# Executing PAPER on GPU %d\n",gpuID);

    // Load reference and fit molecules from disk into CUDAmols and dCUDAMultimols {{{
    CUDAmol refmol;
    CUDAmol* fitmols;
    float3 com_ref,*com_fit;
    dCUDAMultimol hostRefMM,hostFitMM,devRefMM,devFitMM;
    uint* molids;
    uint totalMols,distinctMols;
    float* transforms;
    size_t transform_pitch;
    loadMoleculesRDKit(num_mols, molecules,
                       &fitmols,refmol,&molids,&transforms,transform_pitch,
                       hostFitMM,devFitMM,
                       hostRefMM,devRefMM,
                       com_ref,&com_fit,
                       totalMols,distinctMols);
    uint nfitmols = totalMols;
    //printf("Loaded %d distinct fit molecules, with %d total fit molecules\n",distinctMols,totalMols);
    //printf("Ref molecule: %d atoms\n",refmol.natoms);
    //uint lastid=totalMols+1;
    //for (uint i = 0; i < totalMols; i++) {
    //if (molids[i] != lastid)
    //    printf("Fit molecule %d (id = %d): %d atoms\n",i,molids[i],fitmols[i].natoms);
    //printf("\tTransform = [%f,%f,%f,%f,%f,%f,%f]\n",ELTS7((hostFitMM.transforms+i*hostFitMM.transform_pitch)));
    //float* matrix = transformToCompensatedMatrix(hostFitMM.transforms+i*hostFitMM.transform_pitch,com_ref,com_fit[i]);
    //printTransformMatrix(matrix);
    //free(matrix);
    //lastid=molids[i];
    //}

    /*for (uint i = 0; i < hostFitMM.maxatoms; i++) {
        printf("atom[%d] = [%f,%f,%f,%f]\n",i,hostFitMM.mols[i],hostFitMM.mols[i+hostFitMM.pitch],hostFitMM.mols[i+2*hostFitMM.pitch],hostFitMM.mols[i+3*hostFitMM.pitch]);
    }*/
    //}}}


    float* hostDeviceOverlaps = (float*)malloc(nfitmols*sizeof(float));

    // Allocate space for numTimers on-GPU timers per molecule
    const int numTimers = 8;
    bool timingActivated = false;
    clock_t* hostTimings = (clock_t*)malloc(numTimers*sizeof(clock_t));

    float* hostDeviceTransforms = (float*)malloc(nfitmols*devFitMM.transform_pitch*sizeof(float));


    // optimize the overlaps
    double optstart = getustime();
    const int itercount = optimize_sepkernels(devFitMM,devRefMM,hostDeviceOverlaps,hostTimings,numTimers,com_ref,com_fit);
    double optend = getustime();
    fprintf(stderr,"# Shape overlay optimization used %d iterations of BFGS\n",itercount);

    // Get the transforms back
    cudaMemcpy(hostDeviceTransforms,devFitMM.transforms,nfitmols*transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);
    /*printf("\nAfter optimization:\n");
    for (int i = 0 ; i < nfitmols; i++) {
        float *xf = hostDeviceTransforms+i*transform_pitch;
        float *matrix = transformToCompensatedMatrix(hostDeviceTransforms+i*transform_pitch,com_ref,com_fit[i]);
        printf("[ %.2f %.2f %.2f; %.2f %.2f %.2f %.2f]\n",xf[0],xf[1],xf[2],xf[3],xf[4],xf[5],xf[6]);

        printf("%f\n",hostDeviceOverlaps[i]);
        printTransformMatrix(matrix,stdout);
        free(matrix);
    }*/
    //printf("\n");

    // Max-reduce over starting positions for each molecule
    float* bestOverlaps = new float[distinctMols];
    float* bestTransforms = new float[distinctMols*7];
    memset(bestOverlaps,0,distinctMols*sizeof(float));
    for (uint i = 0; i < totalMols; i++) {
        uint molid = molids[i];
        if (hostDeviceOverlaps[i] > bestOverlaps[molid]) {
            bestOverlaps[molid] = hostDeviceOverlaps[i];
            memcpy(bestTransforms+molid*7,hostDeviceTransforms+i*transform_pitch,7*sizeof(float));
        }
    }
    if (timingActivated) {
        printf("Size of clock_t on host side is %d\n",(int)sizeof(clock_t));
        for (int i = 0; i < numTimers; i++) {
            printf("Timer %d: %lld\n",i,(long long)(hostTimings[i]));
        }
        printf("Average clocks per operation:\n");
        printf("   Line-search: %f\n",(double)(hostTimings[0])/hostTimings[1]);
        printf("   Objective: %f * %f = %f\n",(double)(hostTimings[2])/hostTimings[3],(double)(hostTimings[3])/hostTimings[1],(double)(hostTimings[2])/hostTimings[1]);
        printf("   BFGS update: %f\n",(double)(hostTimings[4])/hostTimings[5]);
        printf("   Gradient: %f\n",(double)(hostTimings[6])/hostTimings[7]);
    }


    float** matrices = new float*[distinctMols];

    bool showresults = true;
    bool benchmark   = false;
    if (showresults) { //{{{
        for (uint i = 0; i < nfitmols; i++) {
            /*CUDAmol finalmol;
            transformCUDAmolToHost(fitmols[i],finalmol,hostDeviceTransforms+i*transform_pitch);
            float hostOvl = hostOverlapVolume(refmol,finalmol);
            delete[] finalmol.atoms;
            printf("Final transform: [%f,%f,%f , %f,%f,%f,%f]\n",hostDeviceTransforms[i*transform_pitch],hostDeviceTransforms[i*transform_pitch+1],hostDeviceTransforms[i*transform_pitch+2],hostDeviceTransforms[i*transform_pitch+3],hostDeviceTransforms[i*transform_pitch+4],hostDeviceTransforms[i*transform_pitch+5],hostDeviceTransforms[i*transform_pitch+6]);
            printf("Fit molecule %d: starting overlap: %f, (putative) ending overlap: %f, ending device overlap: %f, ending host overlap: %f\n\n",i,hostDeviceStartOvl[i],hostDeviceOverlaps[i],hostDeviceEndOvl[i],hostOvl);*/
        }
        for (uint i = 0; i < distinctMols; i++) {
            //printf("Molecule id #%d: optimal overlap value = %f\n",i,bestOverlaps[i]);
            //float *xf = bestTransforms+i*7;
            //printf("[ %.2f %.2f %.2f; %.2f %.2f %.2f %.2f]\n",xf[0],xf[1],xf[2],xf[3],xf[4],xf[5],xf[6]);
            float* matrix = transformToCompensatedMatrix(bestTransforms+i*7,com_ref,com_fit[i]);
            float* newArray = new float[16];
            for (int j = 0; j < 16; j++) {
                newArray[j] = matrix[j];
            }
            matrices[i] = newArray;
            //printTransformMatrix(matrix,stdout);
            free(matrix);
        }
        //printf("Optimization kernel took %f ms (%f ms/mol)\n",(optend-optstart)/1000.0,(optend-optstart)/(1000.0*distinctMols));
    } //}}}
    if (benchmark) { //{{{
        uint bench_runs = 10;
        double start = getustime();
        // Benchmarking iteration: copy transforms to device, execute kernel, bring back transforms and overlaps
        for (uint i = 0; i < bench_runs; i++) {
            // Copy refmol over
            cudaMemcpy(devRefMM.mols,hostRefMM.mols,4*hostRefMM.nmols*devRefMM.pitch*sizeof(float),cudaMemcpyHostToDevice);
            cudaMemcpy(devRefMM.atomcounts,hostRefMM.atomcounts,1*sizeof(uint),cudaMemcpyHostToDevice);

            // Copy fitmols and transforms over
            cudaMemcpy(devFitMM.mols,hostFitMM.mols,4*hostFitMM.nmols*devFitMM.pitch*sizeof(float),cudaMemcpyHostToDevice);
            cudaMemcpy(devFitMM.atomcounts,hostFitMM.atomcounts,hostFitMM.nmols*sizeof(uint),cudaMemcpyHostToDevice);
            cudaMemcpy(devFitMM.molids,hostFitMM.molids,hostFitMM.nmols*sizeof(uint),cudaMemcpyHostToDevice);
            cudaMemcpy(devFitMM.transforms,hostFitMM.transforms,nfitmols*transform_pitch*sizeof(float),cudaMemcpyHostToDevice);
            // Run optimization
            const int itercount = optimize_sepkernels(devFitMM,devRefMM,hostDeviceOverlaps,hostTimings,numTimers,com_ref,com_fit);

            // Copy results back
            cudaMemcpy(hostDeviceTransforms,devFitMM.transforms,nfitmols*transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            //cudaMemcpy(hostDeviceOverlaps,deviceOverlaps,nfitmols*sizeof(float),cudaMemcpyDeviceToHost);
            cudaThreadSynchronize();
            memset(bestOverlaps,0,distinctMols*sizeof(float));
            for (uint i = 0; i < totalMols; i++) {
                uint molid = molids[i];
                if (hostDeviceOverlaps[i] > bestOverlaps[molid]) {
                    bestOverlaps[molid] = hostDeviceOverlaps[i];
                    memcpy(bestTransforms+molid*7,hostDeviceTransforms+i*transform_pitch,7*sizeof(float));
                }
            }
        }
        double end = getustime();
        double runtime = ((end-start)/1000)/bench_runs;
        printf("Benchmark results over %d iterations on %d molecules (%d mol/starts): %f ms/batch optimization, %f ms/molecule, %f ms/position\n",bench_runs,distinctMols,totalMols,runtime,runtime/distinctMols,runtime/totalMols);

    } //}}}

    delete[] bestOverlaps;
    delete[] bestTransforms;
    return matrices;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: paper [GPU ID] [mol_file1] [mol_file2] [mol_file3] ...\n";
        cerr << "       paper [GPU ID] [runfile]\n";
        return 1;
    }

    int gpuID = atoi(argv[1]);
    list<RDKit::ROMol*> molecules;

    if (argc == 3) { // only one argument provided, assume it's a file with mol files
        string listing_filename = argv[2];
        ifstream listing_file(listing_filename);
        if (!listing_file) {
            cerr << "Error: Could not open file " << listing_filename << endl;
            return 1;
        }
        string mol_filename;
        while (getline(listing_file, mol_filename)) {
            auto mol = RDKit::MolFileToMol(mol_filename, false, false, true);
            if (!mol) {
                cerr << "Error: Could not parse molecule from file " << mol_filename << endl;
                return 1;
            }
            molecules.push_back(mol);
        }
    } else { // multiple arguments provided, assume each argument is a mol file
        for (int i = 2; i < argc; i++) {
            string filename = argv[i];
            auto mol = RDKit::MolFileToMol(filename, false, false, true);
            if (!mol) {
                cerr << "Error: Could not parse molecule from file " << filename << endl;
                return 1;
            }
            molecules.push_back(mol);
        }
    }

   float** result = paper(gpuID, molecules);
   for (int i = 0; i < 1; i++) {
       for (int j = 0; j < 16; j++) {
           std::cout << result[i][j] << " ";
       }
       std::cout << std::endl;
   }

    for (auto mol : molecules) {
        delete mol;
    }

    return 0;
}

