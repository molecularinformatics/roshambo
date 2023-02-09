/*
 * cudaVolumeTypes.h
 * Data structure definitions for PAPER
 *
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */

#ifndef _CUDAVOLUMETYPES_CU_
#define _CUDAVOLUMETYPES_CU_
typedef unsigned int uint;

#include <stdbool.h>

// Use this section if we're compiling under g++ instead of nvcc
#ifdef GPP
#include <unistd.h>
typedef struct _float4 {
    float x,y,z,w;
} float4;
typedef struct _float3 {
    float x,y,z;
} float3;
#endif

typedef struct _hCUDAmol {
    float4* atoms;
    uint natoms;
} CUDAmol;
typedef struct _dCUDAmol {
    float* x;
    float* y;
    float* z;
    float* a;
    uint natoms;
} dCUDAmol;
typedef struct _dCUDAMultimol {
    float* mols;
    uint* atomcounts;
    uint* molids;
    uint maxatoms;
    size_t pitch;
    uint nmols;
    float* transforms;
    size_t transform_pitch;
    bool isDeviceMM;
} dCUDAMultimol;
#endif
