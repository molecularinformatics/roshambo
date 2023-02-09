/*
 * cpuPAPER/inputFileReader.h
 * Prototypes for loading molecules from disk using OpenBabel
 *
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */

#ifndef _INPUTFILEREADER_H_
#define _INPUTFILEREADER_H_

#include "cudaVolumeTypes.h"
#include "config_input.h"
#include <list>
#include <set>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdlib>
#include <openbabel/mol.h>

using namespace std;
using namespace OpenBabel;

CUDAmol obMolToCUDAmol(OBMol& obmol);


float3 atomset_centroid(OBMol& mol,const set<int>& atoms);

extern "C" void molFromFile(string filename,list<CUDAmol>& cmols,list<list<float3> >& ring_centroids);

#endif
