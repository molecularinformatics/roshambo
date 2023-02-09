/*
 * deviceAnalyticVolume.cu
 * Implements GPU-accelerated Gaussian analytic volume calculations
 *
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */

#define PI 3.14159265358f
#define PIRTPI 5.56832799683f
#define nThreads blockDim.x
#define fitmolIdx blockIdx.x
#define tidx threadIdx.x

#define DEBUG_VERBOSE 1
#define DEBUG_TRANSFORMATION 2 
#define DEBUG_OVERLAP 4
#define DEBUG_CHAINRULE 8
#define DEBUG_GRADIENT 16

//#define EMU_DEBUG (DEBUG_VERBOSE | DEBUG_CHAINRULE | DEBUG_GRADIENT)
#define EMU_DEBUG 0

#if EMU_DEBUG
#include <stdio.h>
#endif

__device__ void transformToRotationMatrix(float* t,float* matrix);
__device__ void transformMolecule(float* x,float* y,float* z,uint natoms,float* matrix,float* translation);
__device__ void getQuatGradients(float* x,float* y,float* z,uint ourfitcount,float* transform,float* dxdq,float* dxdr,float* dydr,float* dxdu);

/* Molecule layout in memory:
 *
 *  m0a0.x m0a1.x m0a2.x ...
 *  m0a0.y m0a1.y m0a2.y ...
 *  m0a0.z m0a1.z m0a2.z ...
 *  m0a0.w m0a1.w m0a2.w ...
 *  m1a0.x m1a1.x m1a2.x
 *  ...
 *
 *  4*nmolecules rows, xxxmol_pitch floats per row (but we only load xxxatomcount[i] of those)
 *
 * Requires 16*max(#fit atoms) + 16*(# ref atoms) + 4*nthreads bytes of dynamic shared memory
 * Parallel reduction assumes nThreads is a power of 2
 */
__global__ void deviceOverlapVolume(float* fitmols,uint* fitatomcount,size_t fitmol_pitch,
                                    float* refmol ,uint  refatomcount,size_t refmol_pitch,
                                    float* transforms,size_t transform_pitch,
                                    float* finalOverlaps)
{ // {{{
    #if EMU_DEBUG
        if (threadIdx.x == 0) {
            printf("\nEntering block %d: fitatomcount=%lx\n--------------\n",blockIdx.x,(long)fitatomcount);
        }
    #endif
    // Set up our shared memory usage scheme {{{
    
    // The number of elements in the refxfit atom matrix 
    extern __shared__ float shmem[];

    uint ourfitcount;
    uint limit;
    __shared__ float rotationMatrix[9];
    __shared__ float transform[7];
    //uint shmem_base;
    //uint temp;
    //shmem_base = 0;
    ourfitcount = fitatomcount[fitmolIdx];
    limit = ourfitcount * refatomcount;

    float* fitmol_x = &shmem[0];
    float* fitmol_y = &shmem[ourfitcount];
    float* fitmol_z = &shmem[2*ourfitcount];
    float* fitmol_a = &shmem[3*ourfitcount];

    float* refmol_x = &shmem[4*ourfitcount+0];
    float* refmol_y = &shmem[4*ourfitcount+refatomcount];
    float* refmol_z = &shmem[4*ourfitcount+2*refatomcount];
    float* refmol_a = &shmem[4*ourfitcount+3*refatomcount];
    
    // We have nThreads * sizeof(float) partial Vijs
    float* Vijs = &shmem[4*ourfitcount+4*refatomcount];

    // --- END SHMEM ALLOCATION }}}

    // Initialize shared memory {{{
    // Preload fit molecule into shared memory
    // i contains the offset of the atom that thread 0 will load on the current iteration
    for (uint i = 0; i < ourfitcount; i+=nThreads) {
        uint atomidx = i + threadIdx.x;
        if (atomidx < ourfitcount) {
            // Molecule k starts at 4*fitmol_pitch*k (4 rows per molecule) for x row
            fitmol_x[atomidx] = fitmols[(fitmolIdx*fitmol_pitch*4) + atomidx];
            fitmol_y[atomidx] = fitmols[(fitmolIdx*fitmol_pitch*4) + (fitmol_pitch)   + atomidx];
            fitmol_z[atomidx] = fitmols[(fitmolIdx*fitmol_pitch*4) + (2*fitmol_pitch) + atomidx];
            fitmol_a[atomidx] = fitmols[(fitmolIdx*fitmol_pitch*4) + (3*fitmol_pitch) + atomidx];

        }
    }

    // Preload ref molecule into shared memory
    // i contains the offset of the atom that thread 0 will load on the current iteration
    for (uint i = 0; i < refatomcount; i+=nThreads) {
        uint atomidx = i + threadIdx.x;
        if (atomidx < refatomcount) {
            // Molecule k starts at 4*fitmol_pitch*k (4 rows per molecule) for x row
            refmol_x[atomidx] = refmol[atomidx];
            refmol_y[atomidx] = refmol[(refmol_pitch)   + atomidx];
            refmol_z[atomidx] = refmol[(2*refmol_pitch) + atomidx];
            refmol_a[atomidx] = refmol[(3*refmol_pitch) + atomidx];
        }
    }

    // Preload transform into shared memory
    if (threadIdx.x < 7) {
        transform[threadIdx.x] = transforms[fitmolIdx*transform_pitch + threadIdx.x];
    }

    // Initialize partial Vij count
    Vijs[threadIdx.x] = 0.0f;
    
    __syncthreads();

    // -------------- END SHARED MEMORY INITIALIZATION ---------------------- }}}

    // Transform the local copy of the fit molecule {{{
    #if (EMU_DEBUG & DEBUG_TRANSFORMATION) //{{{
    if (threadIdx.x == 0) {
        printf("Transform in block %d: [%f,%f,%f,%f,%f,%f,%f]\n",blockIdx.x,transform[0],transform[1],transform[2],transform[3],transform[4],transform[5],transform[6]);
        #if ((EMU_DEBUG & DEBUG_TRANSFORMATION) & (EMU_DEBUG & DEBUG_VERBOSE))
        for (uint i = 0; i < refatomcount; i++) {
            printf("\t Ref Atom %d: [%f,%f,%f], alpha = %f\n",i,refmol_x[i],refmol_y[i],refmol_z[i],refmol_a[i]);
        }
        printf("\n");
        printf("Pre-transform fit coordinates:\n");
        for (uint i = 0; i < ourfitcount; i++) {
            printf("\t Fit Atom %d: [%f,%f,%f], alpha = %f\n",i,fitmol_x[i],fitmol_y[i],fitmol_z[i],fitmol_a[i]);
        }
        printf("\n");
        #endif
    }

    #endif //}}}
    transformToRotationMatrix(transform,rotationMatrix);
    __syncthreads();
    #if (EMU_DEBUG & DEBUG_TRANSFORMATION) //{{{
    if (tidx == 0) {
        printf("Rotation matrix in block %d:\n",blockIdx.x);
        printf("[[ %f %f %f\n   %f %f %f\n   %f %f %f ]]\n",rotationMatrix[0],rotationMatrix[1],rotationMatrix[2],rotationMatrix[3],rotationMatrix[4],rotationMatrix[5],rotationMatrix[6],rotationMatrix[7],rotationMatrix[8]);
        printf("\n");
        #if ((EMU_DEBUG & DEBUG_TRANSFORMATION) && (EMU_DEBUG & DEBUG_VERBOSE))
        for (uint i = 0; i < ourfitcount; i++) {
            printf("Atom %d started at [%f,%f,%f]\n",i,fitmol_x[i],fitmol_y[i],fitmol_z[i]);
        }
        #endif
    }
    #endif //}}}
    transformMolecule(fitmol_x,fitmol_y,fitmol_z,ourfitcount,rotationMatrix,transform);
    __syncthreads();
    #if ((EMU_DEBUG & DEBUG_TRANSFORMATION) && (EMU_DEBUG & DEBUG_VERBOSE)) //{{{
    if (tidx == 0) {
        for (uint i = 0; i < ourfitcount; i++) {
            printf("Atom %d moved to [%f,%f,%f]\n",i,fitmol_x[i],fitmol_y[i],fitmol_z[i]);
        }
    }
    #endif //}}}
        
    #if ((EMU_DEBUG & DEBUG_TRANSFORMATION) && (EMU_DEBUG & DEBUG_VERBOSE)) //{{{
    if (threadIdx.x == 0) {
        printf("Post-transform fit coordinates:\n");
        for (uint i = 0; i < ourfitcount; i++) {
            printf("\t Atom %d: [%f,%f,%f], alpha = %f\n",i,fitmol_x[i],fitmol_y[i],fitmol_z[i],fitmol_a[i]);
        }
        printf("\n");
    }

    #endif //}}}
    // --- END TRANSFORMATION SETUP }}}

    // Calculate partial overlap volume per thread {{{
    for (uint base = 0; base < limit; base += nThreads) {
        uint myidx = base + threadIdx.x;
        if (myidx < limit) {
            uint refatom = myidx/ourfitcount;
            // Replace modulo with a MAD
            uint fitatom = myidx - __umul24(refatom,ourfitcount);
            #if ((EMU_DEBUG & DEBUG_OVERLAP) & (EMU_DEBUG & DEBUG_VERBOSE)) //{{{
            //printf("Thread %d: myidx = %d, refatom = %d, fitatom = %d, ourfitcount= %d\n",tidx,myidx,refatom,fitatom,refatomcount);
            #endif //}}}
        
            float temp,Rij2=0.0f;
            temp = fitmol_x[fitatom] - refmol_x[refatom]; 
            Rij2 += temp*temp;
            temp = fitmol_y[fitatom] - refmol_y[refatom]; 
            Rij2 += temp*temp;
            temp = fitmol_z[fitatom] - refmol_z[refatom]; 
            Rij2 += temp*temp;

            // TODO: cutoff?

            temp = __expf(-(refmol_a[refatom])*fitmol_a[fitatom]*Rij2/(refmol_a[refatom]+fitmol_a[fitatom]));
            #if ((EMU_DEBUG & DEBUG_OVERLAP) & (EMU_DEBUG & DEBUG_VERBOSE)) //{{{
            float Kij = temp;
            float suma = refmol_a[refatom]+fitmol_a[fitatom];
            float rtsum = sqrtf(suma);
            float denom = suma*suma;
            #endif //}}}

            temp *= 8 * __powf(PI/(refmol_a[refatom]+fitmol_a[fitatom]),1.5f);
            //temp *= 8 * PIRTPI * sqrtf(refmol_a[refatom]+fitmol_a[fitatom])/( (refmol_a[refatom]+fitmol_a[fitatom])*(refmol_a[refatom]+fitmol_a[fitatom]));
            #if ((EMU_DEBUG & DEBUG_OVERLAP) & (EMU_DEBUG & DEBUG_VERBOSE)) //{{{
            printf("Thread %d calculating overlap r[%d] f[%d] = %f\n",tidx,refatom,fitatom,temp);
            //printf("\tr[%d]= (%f,%f,%f,%f)\n\tf[%d]=(%f,%f,%f,%f)\n",refatom,refmol_x[refatom],refmol_y[refatom],refmol_z[refatom],refmol_a[refatom],fitatom,fitmol_x[fitatom],fitmol_y[fitatom],fitmol_z[fitatom],fitmol_a[fitatom]);
            //printf("\tRij2 = %f, Kij = %f, Kij scaling = 8*%f\n",Rij2,Kij,__powf(PI/(refmol_a[refatom]+fitmol_a[fitatom]),1.5f));
            //printf("\tRij2 = %f, Kij = %f, Kij scaling = 8 * %f * (%f)/(%f)\n",Rij2,Kij,PIRTPI,rtsum,denom);
            //printf("\n");
            #endif //}}}
            Vijs[threadIdx.x] += temp;
        }
    }
    __syncthreads();
    // -- END MAIN CALCULATION LOOP }}}

    // Parallel-reduce overlap volume {{{
    // Modified from (buggy) lecture slides of Hendrik Lensch
    // http://www.mpi-inf.mpg.de/departments/d4/teaching/ws2007-2008/cg/slides/CG20-Cuda2.pdf
    for (uint stride = nThreads>>1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (threadIdx.x < stride)
            Vijs[threadIdx.x] += Vijs[threadIdx.x + stride];
    }
    __syncthreads();
    // -- END PARALLEL REDUCE }}}

    // Store overlap volume into global memory {{{
    if (threadIdx.x == 0)
        finalOverlaps[fitmolIdx] = Vijs[0];
    // -- END COMMIT }}}
    return;
} // }}}
/*
 * Requires 32*max(#fit atoms) + 16*(# ref atoms) + 8*4*nthreads of dynamic shared memory
 * Parallel reduction assumes nThreads is a power of 2
 */
__global__ void deviceOverlapAndGradient(float* fitmols,uint* fitatomcount,size_t fitmol_pitch,
                                         float* refmol ,uint  refatomcount,size_t refmol_pitch,
                                         float* transforms,size_t transform_pitch,
                                         float* finalOverlapsAndGradients,size_t gradient_pitch) 
{ // {{{

    #if EMU_DEBUG
        if (threadIdx.x == 0) {
            //printf("\nEntering block %d: fitmolIdx = %d, fitatomcount=%lx\n--------------\n",blockIdx.x,fitmolIdx,(long)fitatomcount);
        }
    #endif

    // Set up our shared memory usage scheme {{{
    
    // The number of elements in the refxfit atom matrix 
    extern __shared__ float shmem[];

    uint ourfitcount;
    uint limit;
    __shared__ float rotationMatrix[9];
    __shared__ float transform[7];
    //uint shmem_base;
    //uint temp;
    //shmem_base = 0;
    ourfitcount = fitatomcount[fitmolIdx];
    #if ((EMU_DEBUG & DEBUG_GRADIENT) & (EMU_DEBUG & DEBUG_VERBOSE)) //{{{
        printf("Thread %d: loaded fitcount\n",tidx); fflush(stdout);
    #endif //}}}
    limit = ourfitcount * refatomcount;

    float* fitmol_x = &shmem[0];
    float* fitmol_y = &shmem[ourfitcount];
    float* fitmol_z = &shmem[2*ourfitcount];
    float* fitmol_a = &shmem[3*ourfitcount];

    float* dxdq = &shmem[4*ourfitcount];
    float* dxdr = &shmem[5*ourfitcount];
    float* dydr = &shmem[6*ourfitcount];
    float* dxdu = &shmem[7*ourfitcount];

    float* refmol_x = &shmem[8*ourfitcount+0];
    float* refmol_y = &shmem[8*ourfitcount+  refatomcount];
    float* refmol_z = &shmem[8*ourfitcount+2*refatomcount];
    float* refmol_a = &shmem[8*ourfitcount+3*refatomcount];
    
    // We have nThreads * sizeof(float) partial Vijs, and same for each gradient cord
    float* Vijs = &shmem[8*ourfitcount+4*refatomcount];
    float* ddxs = &shmem[8*ourfitcount+4*refatomcount+  nThreads];
    float* ddys = &shmem[8*ourfitcount+4*refatomcount+2*nThreads];
    float* ddzs = &shmem[8*ourfitcount+4*refatomcount+3*nThreads];
    float* ddqs = &shmem[8*ourfitcount+4*refatomcount+4*nThreads];
    float* ddrs = &shmem[8*ourfitcount+4*refatomcount+5*nThreads];
    float* ddss = &shmem[8*ourfitcount+4*refatomcount+6*nThreads];
    float* ddus = &shmem[8*ourfitcount+4*refatomcount+7*nThreads];

    // -- END SHMEM ALLOCATION }}}

    #if ((EMU_DEBUG & DEBUG_GRADIENT) & (EMU_DEBUG & DEBUG_VERBOSE)) //{{{
        printf("Thread %d: set up shmem\n",tidx); fflush(stdout);
    #endif //}}}
    // Initialize shared memory {{{
    // Preload fit molecule into shared memory
    // i contains the offset of the atom that thread 0 will load on the current iteration
    for (uint i = 0; i < ourfitcount; i+=nThreads) {
        uint atomidx = i + threadIdx.x;
        if (atomidx < ourfitcount) {
            // Molecule k starts at 4*fitmol_pitch*k (4 rows per molecule) for x row
            fitmol_x[atomidx] = fitmols[(fitmolIdx*fitmol_pitch*4) + atomidx];
            fitmol_y[atomidx] = fitmols[(fitmolIdx*fitmol_pitch*4) + (fitmol_pitch)   + atomidx];
            fitmol_z[atomidx] = fitmols[(fitmolIdx*fitmol_pitch*4) + (2*fitmol_pitch) + atomidx];
            fitmol_a[atomidx] = fitmols[(fitmolIdx*fitmol_pitch*4) + (3*fitmol_pitch) + atomidx];

        }
    }

    // Preload ref molecule into shared memory
    // i contains the offset of the atom that thread 0 will load on the current iteration
    for (uint i = 0; i < refatomcount; i+=nThreads) {
        uint atomidx = i + threadIdx.x;
        if (atomidx < refatomcount) {
            // Molecule k starts at 4*fitmol_pitch*k (4 rows per molecule) for x row
            refmol_x[atomidx] = refmol[atomidx];
            refmol_y[atomidx] = refmol[(refmol_pitch)   + atomidx];
            refmol_z[atomidx] = refmol[(2*refmol_pitch) + atomidx];
            refmol_a[atomidx] = refmol[(3*refmol_pitch) + atomidx];
        }
    }

    // Preload transform into shared memory
    if (threadIdx.x < 7) {
        transform[threadIdx.x] = transforms[fitmolIdx*transform_pitch + threadIdx.x];
    }

    // Initialize partial counts
    Vijs[threadIdx.x] = 0.0f;
    ddxs[threadIdx.x] = 0.0f;
    ddys[tidx] = 0.0f;
    ddzs[tidx] = 0.0f;
    ddqs[tidx] = 0.0f;
    ddrs[tidx] = 0.0f;
    ddss[tidx] = 0.0f;
    ddus[tidx] = 0.0f;
    
    __syncthreads();

    // -------------- END SHARED MEMORY INITIALIZATION ---------------------- }}}

    #if ((EMU_DEBUG & DEBUG_GRADIENT) & (EMU_DEBUG & DEBUG_VERBOSE)) //{{{
        printf("Thread %d: inited shmem\n",tidx);
    #endif //}}}
    // Transform the local copy of the fit molecule {{{
    transformToRotationMatrix(transform,rotationMatrix);
    __syncthreads();
    transformMolecule(fitmol_x,fitmol_y,fitmol_z,ourfitcount,rotationMatrix,transform);
    __syncthreads();
    // --- END TRANSFORMATION SETUP }}}
    
    #if ((EMU_DEBUG & DEBUG_GRADIENT) & (EMU_DEBUG & DEBUG_VERBOSE)) //{{{
        printf("Thread %d: transformed mol\n",tidx);
    #endif //}}}
    // Precalculate four quaternion gradients (others are equal to +-1 times these) {{{
    getQuatGradients(fitmol_x,fitmol_y,fitmol_z,ourfitcount,transform,dxdq,dxdr,dydr,dxdu);
    __syncthreads();
    // -- END QUATERNION CHAIN RULE SETUP }}}
    #if ((EMU_DEBUG & DEBUG_GRADIENT) & (EMU_DEBUG & DEBUG_VERBOSE)) //{{{
        printf("Thread %d: calculated quat ders\n",tidx);
    #endif //}}}

    // Calculate partial overlap volume and gradient per thread {{{
    for (uint base = 0; base < limit; base += nThreads) {
        uint myidx = base + threadIdx.x;
        if (myidx < limit) {
            uint refatom = myidx/ourfitcount;
            // Replace modulo with a MAD
            uint fitatom = myidx - __umul24(refatom,ourfitcount);
            #if ((EMU_DEBUG & DEBUG_GRADIENT) & (EMU_DEBUG & DEBUG_VERBOSE)) //{{{
            printf("Thread %d: refatom %d, fitatom %d\n",tidx,refatom,fitatom);
            #endif //}}}
        
            float temp,Rij2=0.0f;
            temp = fitmol_x[fitatom] - refmol_x[refatom]; 
            Rij2 += temp*temp;
            temp = fitmol_y[fitatom] - refmol_y[refatom]; 
            Rij2 += temp*temp;
            temp = fitmol_z[fitatom] - refmol_z[refatom]; 
            Rij2 += temp*temp;

            // TODO: cutoff?

            // temp is the Vij
            temp = __expf(-(refmol_a[refatom])*fitmol_a[fitatom]*Rij2/(refmol_a[refatom]+fitmol_a[fitatom]));
            temp *= 8 * __powf(PI/(refmol_a[refatom]+fitmol_a[fitatom]),1.5f);
            Vijs[threadIdx.x] += temp;

            // now transform temp into the scalar coefficient on the gradient calc
            temp *= (-2)*refmol_a[refatom]*fitmol_a[fitatom]/(refmol_a[refatom]+fitmol_a[fitatom]);
            
            // Use preprocessor to define the other quaternion grads in terms of 4 we know {{{
            #define dxdq (dxdq[fitatom])
            #define dxdr (dxdr[fitatom])
            #define dxdu (dxdu[fitatom])
            #define dydr (dydr[fitatom])
            #define dzds (dxdq)
            #define dydu (-dxdq)
            #define dyds (dxdr)
            #define dzdu (dxdr)
            #define dxds (-dydr)
            #define dzdq (dydr)
            #define dydq (dxdu)
            #define dzdr (-dxdu)
            // }}}
            // Do x,y,z separately to minimize # regs
            #if ((EMU_DEBUG & DEBUG_CHAINRULE) & (EMU_DEBUG & DEBUG_VERBOSE)) //{{{
                if (refatom == 0) {
                    printf("Fitatom %d: dxdq=%f, dxdr=%f, dxdu=%f, dydr=%f\n",fitatom,dxdq,dxdr,dxdu,dydr);
                }
            #endif //}}}
            
            // Derivs WRT x {{{
            // here delta is delta-prime of x
            float delta = refmol_x[refatom] - fitmol_x[fitatom];
            delta *= temp;
            ddxs[tidx] += delta;
            ddqs[tidx] += delta*dxdq;
            ddrs[tidx] += delta*dxdr;
            ddss[tidx] += delta*dxds;
            ddus[tidx] += delta*dxdu;
            //}}}

            // Derivs WRT y {{{
            // here delta is delta-prime of y
            delta = refmol_y[refatom] - fitmol_y[fitatom];
            delta *= temp;
            ddys[tidx] += delta;
            ddqs[tidx] += delta*dydq;
            ddrs[tidx] += delta*dydr;
            ddss[tidx] += delta*dyds;
            ddus[tidx] += delta*dydu;
            // }}}

            // Derivs WRT z {{{
            // here delta is delta-prime of z
            delta = refmol_z[refatom] - fitmol_z[fitatom];
            delta *= temp;
            ddzs[tidx] += delta;
            ddqs[tidx] += delta*dzdq;
            ddrs[tidx] += delta*dzdr;
            ddss[tidx] += delta*dzds;
            ddus[tidx] += delta*dzdu;
            // }}}

            // Clean up preprocessor trickery {{{
            #undef dxdq
            #undef dxdr
            #undef dydr
            #undef dxdu
            #undef dzds 
            #undef dydu 
            #undef dyds 
            #undef dzdu 
            #undef dxds 
            #undef dzdq 
            #undef dydq 
            #undef dzdr 
            // }}}
        }
    }
    __syncthreads();
    // -- END MAIN CALCULATION LOOP }}}

    #if ((EMU_DEBUG & DEBUG_GRADIENT) & (EMU_DEBUG & DEBUG_VERBOSE)) //{{{
        printf("Thread %d: calculated overlap/grad vals\n",tidx);
    #endif //}}}
    // Parallel-reduce overlap and gradients {{{
    // Modified from (buggy) lecture slides of Hendrik Lensch
    // http://www.mpi-inf.mpg.de/departments/d4/teaching/ws2007-2008/cg/slides/CG20-Cuda2.pdf
    for (uint stride = nThreads>>1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (threadIdx.x < stride) {
            Vijs[threadIdx.x] += Vijs[threadIdx.x + stride];
            ddxs[threadIdx.x] += ddxs[threadIdx.x + stride];
            ddys[threadIdx.x] += ddys[threadIdx.x + stride];
            ddzs[threadIdx.x] += ddzs[threadIdx.x + stride];
            ddqs[threadIdx.x] += ddqs[threadIdx.x + stride];
            ddrs[threadIdx.x] += ddrs[threadIdx.x + stride];
            ddss[threadIdx.x] += ddss[threadIdx.x + stride];
            ddus[threadIdx.x] += ddus[threadIdx.x + stride];
        }
    }
    __syncthreads();
    // -- END PARALLEL REDUCTION }}}
    #if ((EMU_DEBUG & DEBUG_GRADIENT) & (EMU_DEBUG & DEBUG_VERBOSE)) //{{{
        printf("Thread %d: parallel reduced\n",tidx);
    #endif //}}}

//TODO: need to add in the penalty term!

    // Write results back to global memory {{{

    // Transpose overlap and gradients into shared memory for coalesced write to gmem
    // Reuse the space we had for the rotation matrix
    switch (tidx) {
        case 0: rotationMatrix[0] = Vijs[0]; break;
        case 1: rotationMatrix[1] = ddxs[0]; break;
        case 2: rotationMatrix[2] = ddys[0]; break;
        case 3: rotationMatrix[3] = ddzs[0]; break;
        case 4: rotationMatrix[4] = ddqs[0]; break;
        case 5: rotationMatrix[5] = ddrs[0]; break;
        case 6: rotationMatrix[6] = ddss[0]; break;
        case 7: rotationMatrix[7] = ddus[0]; break;
        default: break;
    }
    __syncthreads();

    #if ((EMU_DEBUG & DEBUG_GRADIENT) & (EMU_DEBUG & DEBUG_VERBOSE)) //{{{
        printf("Thread %d: transposed results\n",tidx);
    #endif //}}}
    // Store overlap and gradients into global memory
    if (threadIdx.x < 8) {
        finalOverlapsAndGradients[fitmolIdx*gradient_pitch + tidx] = rotationMatrix[tidx];
    }
    #if ((EMU_DEBUG & DEBUG_GRADIENT) & (EMU_DEBUG & DEBUG_VERBOSE)) //{{{
        printf("Thread %d: committed\n",tidx);
    #endif //}}}
    // -- END RESULTS COMMIT }}}

    return;

} //}}}

__device__ void transformToRotationMatrix(float* t,float* matrix) 
{ // {{{
    #define q t[3]
    #define r t[4]
    #define s t[5]
    #define u t[6]
    #define q2 q*q
    #define r2 r*r
    #define s2 s*s
    #define u2 u*u
    #define qr q*r
    #define sr s*r
    #define qu q*u
    #define ru r*u
    #define sq s*q
    #define su s*u
    #define invmag2 (1.0f/(q*q+r*r+s*s+u*u))
    // Assumes nThreads >= 9
    switch (threadIdx.x) {
        case 0: matrix[threadIdx.x] = invmag2 * (q2+r2-s2-u2); break;
        case 1: matrix[threadIdx.x] = invmag2 * 2 * (sr+qu); break;
        case 2: matrix[threadIdx.x] = invmag2 * 2 * (ru-sq); break;
        case 3: matrix[threadIdx.x] = invmag2 * 2 * (sr-qu); break;
        case 4: matrix[threadIdx.x] = invmag2 * (q2-r2+s2-u2); break;
        case 5: matrix[threadIdx.x] = invmag2 * 2 * (su+qr); break;
        case 6: matrix[threadIdx.x] = invmag2 * 2 * (ru+sq); break;
        case 7: matrix[threadIdx.x] = invmag2 * 2 * (su-qr); break;
        case 8: matrix[threadIdx.x] = invmag2 * (q2-r2-s2+u2); break;
        default: break;
    }
    return;
    #undef q
    #undef r
    #undef s
    #undef u
    #undef q2
    #undef r2
    #undef s2
    #undef u2
    #undef qr
    #undef sr
    #undef qu
    #undef ru
    #undef sq
    #undef su
    #undef invmag2
} // }}}

__device__ void transformMolecule(float* x,float* y,float* z,uint natoms,float* matrix,float* translation) 
{ // {{{
    // TODO: can get rid of this loop if we guarantee nThreads >= natoms
    // TODO: replace mul/div with shifts
    uint niter = (natoms + nThreads - 1)/nThreads;
    for (uint iter = 0; iter < niter; iter++) {
        uint atom = iter*nThreads + threadIdx.x;
        if (atom < natoms) {
            float oldx,oldy,oldz;
            oldx = x[atom];
            oldy = y[atom];
            oldz = z[atom];
            x[atom] = oldx*matrix[0*3+0] + oldy*matrix[0*3+1] + oldz*matrix[0*3+2] + translation[0];
            y[atom] = oldx*matrix[1*3+0] + oldy*matrix[1*3+1] + oldz*matrix[1*3+2] + translation[1];
            z[atom] = oldx*matrix[2*3+0] + oldy*matrix[2*3+1] + oldz*matrix[2*3+2] + translation[2];
        }
    }

    return;
} // }}}

__device__ void getQuatGradients(float* x,float* y,float* z,uint natoms,float* transform,float* dxdq,float* dxdr,float* dydr,float* dxdu) 
{ // {{{ 
    uint niter = (natoms + nThreads - 1)/nThreads;
    float invmag2 = 1.0f/(transform[3]*transform[3] + transform[4]*transform[4] + transform[5]*transform[5] + transform[6]*transform[6]);
    #define q transform[3]
    #define r transform[4]
    #define s transform[5]
    #define u transform[6]
    for (uint iter = 0; iter < niter; iter++) {
        uint atom = iter*nThreads + threadIdx.x;
        if (atom < natoms) {
            dxdq[atom] = (invmag2*2.0f*( q*x[atom] + u*y[atom] - s*z[atom]));
            dxdr[atom] = (invmag2*2.0f*( r*x[atom] + s*y[atom] + u*z[atom]));
            dydr[atom] = (invmag2*2.0f*( s*x[atom] - r*y[atom] + q*z[atom]));
            dxdu[atom] = (invmag2*2.0f*(-u*x[atom] + q*y[atom] + r*z[atom]));
        }
    }
    #undef q
    #undef r
    #undef s
    #undef u
    return;

} // }}}
