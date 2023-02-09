/*
 * deviceOverlay_sep.h
 * Prototypes for GPU-accelerated volume overlap optimization with multiple kernel calls
 *  
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */

#ifndef _DEVICEOVERLAYSEP_H_
#define _DEVICEOVERLAYSEP_H_
#include "cudaVolumeTypes.h"
 __host__ int optimize_sepkernels(dCUDAMultimol d_fitmols, dCUDAMultimol d_refMultimol, float* finalScores, clock_t *h_timers, const uint numTimers,const float3 com_ref,const float3* com_fit);
#endif

