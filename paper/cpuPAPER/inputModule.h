#ifndef _INPUTMODULE_H_
#define _INPUTMODULE_H_

#include "cudaVolumeTypes.h"

extern "C" void loadMolecules(int argc,char** argv,
                              CUDAmol** fitmols,CUDAmol& refmol,uint** molids,float** transforms,
                              float3& com_ref,float3** com_fit,
                              uint& totalMols,uint& distinctMols);

#endif
