/*
 * inputModule.h
 * Prototypes for molecule-file frontend to PAPER
 *  
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */
#ifndef _INPUTMODULE_H_
#define _INPUTMODULE_H_

#include "cudaVolumeTypes.h"

extern "C" void loadMolecules(int argc,char** argv,
                              CUDAmol** fitmols,CUDAmol& refmol,uint** molids,float** transforms,size_t& transform_pitch,
                              dCUDAMultimol& hostFitMM,dCUDAMultimol& devFitMM,
                              dCUDAMultimol& hostRefMM,dCUDAMultimol& devRefMM,
                              float3& com_ref,float3** com_fit,
                              uint& totalMols,uint& distinctMols);

#endif
