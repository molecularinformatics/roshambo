/*
 * deviceAnalyticVolume.h
 * Prototypes for GPU-accelerated volume calculation functions
 *
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */

__device__ void transformToRotationMatrix(float* t,float* matrix);
__device__ void transformMolecule(float* x,float* y,float* z,uint natoms,float* matrix,float* translation);
__device__ void getQuatGradients(float* x,float* y,float* z,uint ourfitcount,float* transform,float* dxdq,float* dxdr,float* dydr,float* dxdu);

__global__ void deviceOverlapVolume(float* fitmols,uint* fitatomcount,size_t fitmol_pitch,
                                    float* refmol ,uint  refatomcount,size_t refmol_pitch,
                                    float* transforms,size_t transform_pitch,
                                    float* finalOverlaps);
__global__ void deviceOverlapAndGradient(float* fitmols,uint* fitatomcount,size_t fitmol_pitch,
                                         float* refmol ,uint  refatomcount,size_t refmol_pitch,
                                         float* transforms,size_t transform_pitch,
                                         float* finalOverlapsAndGradients,size_t gradient_pitch);
