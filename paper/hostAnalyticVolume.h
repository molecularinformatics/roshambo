/*
 * hostAnalyticVolume.h
 * Prototypes for Gaussian analytic volume calculation on the CPU
 *  
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */

#ifndef hostvolumeh
#define hostvolumeh
#include "cudaVolumeTypes.h"
inline float dot3(float3 a,float3 b);
inline float3 operator-(const float3& a,const float3& b);
inline float distsqr(const float4& a,const float4& b);
inline float min(float a,float b);
inline float max(float a,float b);

float hostOverlapVolume(const CUDAmol& ref,const CUDAmol& fit);
float hostOverlapVolumeAnalytic2(const CUDAmol& ref,const CUDAmol& fit);
float hostOverlapVolumeHS(const CUDAmol& ref,const CUDAmol& fit);

void hostQuatGradients(const CUDAmol& fit,float* transform,float3* qDers,float3* rDers,float3* sDers,float3* uDers);
void hostGradient(const CUDAmol& ref,const CUDAmol& fit,float* transform,float* gradient);

void transformCUDAmolToHost(CUDAmol& cmol,CUDAmol& newmol,float* transform);

#endif
