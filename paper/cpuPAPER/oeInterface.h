/* 
 * cpuPAPER/oeInterface.h
 * Prototypes for OpenEye molecule loader functions
 *
 *
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */
extern "C" CUDAmol molfileToCUDAmol(char* filename);
CUDAmol oeMolToCUDAmol(OEMol& mol);
extern "C" void CUDAmolToFile(char* model_file,char* out_file,CUDAmol& cmol);
extern "C" void combinedCUDAmolToFile(char* model_file1,char* model_file2,char* outfile1,char* outfile2, CUDAmol& cmol,uint natoms1);
