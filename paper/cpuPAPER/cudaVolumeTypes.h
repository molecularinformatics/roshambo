/*
 * cpuPAPER/cudaVolumeTypes.h
 * Data structure definitions for cpuPAPER.
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
typedef struct _float4 {
	float x,y,z,w;
} float4;
typedef struct _float3 {
	float x,y,z;
} float3;
inline float3 make_float3(float a,float b,float c) {
	float3 tmp;
	tmp.x=a;
	tmp.y=b;
	tmp.z=c;
	return tmp;
}
typedef unsigned int uint;
typedef struct _uint3 {
	uint x,y,z;
} uint3;

typedef struct _CUDAmol {
    float4* atoms;
    uint natoms;
} CUDAmol;
typedef struct _Grid {
    float3 lb;
    float res;
    uint3 extent;
    float* points;
} Grid;
typedef struct _PolyGrid {
    float3 lb;
    float res;
    uint3 extent;
    float** points;
} PolyGrid;
#endif
