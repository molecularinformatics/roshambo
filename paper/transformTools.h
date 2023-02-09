/*
 * transformTools.h
 * Prototypes of utility functions to handle 4x4 matrix/quaternion transformations
 *  
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */
#ifndef _TRANSFORMTOOLS_H_
#define _TRANSFORMTOOLS_H_
#include "cudaVolumeTypes.h"
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>

extern "C" {
void affineQuatToMatrix(const float* quat,float* matrix);
void printTransformMatrix(const float* matrix,FILE* f);

void composeTransform(float* a,const float* b);
float* transformToCompensatedMatrix(float* transform,float3 com_ref,float3 com_fit);
float3 transformSinglePoint(const float* quat,const float3 point);
}
#endif
