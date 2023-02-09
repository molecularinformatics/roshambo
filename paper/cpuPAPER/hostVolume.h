/*
 * cpuPAPER/hostVolume.h
 * Prototypes for CPU volume calculation functions.
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
#include "config.h"
inline float dot3(float3 a,float3 b);
inline float3 operator-(const float3& a,const float3& b);
inline float distsqr(const float3& a,const float4& b);
inline float distsqr(const float4& a,const float4& b);
inline float min(float a,float b);
inline float max(float a,float b);
void boundingBox(CUDAmol& mol,float margin,float3& lb,float3& ub);
inline float3 gridSpaceToRealSpace(float res,const float3& lb,uint x,uint y,uint z);
inline float rho(float4 atom,float3 gc);
float getVolume(const CUDAmol& ref,Grid& grid);
float getOverlapVolume(const CUDAmol& ref,const CUDAmol& fit,Grid& grid);
float getOverlapVolumeVerbose(const CUDAmol& ref,const CUDAmol& fit,Grid& grid);
void getQuatGradients(const CUDAmol& fit,float* transform,float3* qDers,float3* rDers,float3* sDers,float3* uDers);
void getGradient(const CUDAmol& ref,const CUDAmol& fit,Grid& grid,float* transform,float* gradient);
float getObjectiveAndGradient(const CUDAmol& ref,const CUDAmol& fit,Grid& grid,float* transform,float* gradient);
Grid getHostGridFromBox(float3 lb,float3 ub,float res);
Grid getHostGrid(CUDAmol& mol,float res,float margin);
void transformCUDAmolToHost(CUDAmol& cmol,CUDAmol& newmol,float* transform,uint startatom);
#endif
