/*
 * deviceOverlay.cu
 * Implements GPU-accelerated Gaussian volume overlap optimization.
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

#include "cudaVolumeTypes.h"
#include "config_algo.h"
#include "config_input.h"
#include "transformTools.h"
#include <stdint.h>

struct PAPER_data { //{{{
    // Mol/coord data
    dCUDAMultimol fitmols;
    dCUDAMultimol refmol;
    float *gradient;
    float *search_direction;
    float *invH;
    float *score;
    int *completed;
    float *ls_maxstep;
    unsigned hessian_pitch;

    // Fixed parameters
    float bfgs_tol_gradient;
    uint  bfgs_max_iters;
    float bfgs_STPMX;
    float bfgs_epsilon;
    float bfgs_tol_coord;
    float ls_alpha;
    float ls_tol;
    uint  ls_max_iters;

    // Timers
    clock_t *timers;
    unsigned num_timers;
    unsigned timer_pitch;

    // Miscellaneous parameters
    bool cpu_bfgs_update;
}; //}}}

__device__ __constant__ PAPER_data d_optparms;

__device__ __shared__ uint row_per_block,col_per_block;

static __device__ void transformToRotationMatrix(float* t,float* matrix);
static __device__ void transformMolecule(float* x,float* y,float* z,uint natoms,float* matrix,float* translation);
static __device__ void getQuatGradients(float* x,float* y,float* z,uint ourfitcount,float* transform,float* dxdq,float* dxdr,float* dydr,float* dxdu);

static __device__ void transformToRotationMatrix(float* t,float* matrix) 
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
    /*switch (threadIdx.x) {
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
    }*/
    if (threadIdx.x == 0) {
        matrix[0] = invmag2 * (q2+r2-s2-u2); 
        matrix[1] = invmag2 * 2 * (sr+qu); 
        matrix[2] = invmag2 * 2 * (ru-sq); 
        matrix[3] = invmag2 * 2 * (sr-qu); 
        matrix[4] = invmag2 * (q2-r2+s2-u2); 
        matrix[5] = invmag2 * 2 * (su+qr); 
        matrix[6] = invmag2 * 2 * (ru+sq); 
        matrix[7] = invmag2 * 2 * (su-qr); 
        matrix[8] = invmag2 * (q2-r2-s2+u2); 
    }
    
    // cross-matrix - seems to not give the same answer as above?
    /*__shared__ float cm[16];
    if (threadIdx.x < 16) {
        int row = threadIdx.x >> 2;
        int col = threadIdx.x & 0x3;
        cm[threadIdx.x] = 2.0f*invmag2*t[3+row]*t[3+col];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        //float imag2 = 1.0f/(cm[0] + cm[5] + cm[10] + cm[15]);
        float imag2 = invmag2;
        matrix[0] = 0.5f * (cm[0] + cm[5] - cm[10] - cm[15]);
        matrix[4] = 0.5f * (cm[0] - cm[5] + cm[10] - cm[15]);
        matrix[8] = 0.5f * (cm[0] - cm[5] - cm[10] + cm[15]);
        matrix[1] = (cm[9]  + cm[3]);
        matrix[3] = (cm[9]  - cm[3]);
        matrix[2] = (cm[7]  - cm[8]);
        matrix[6] = (cm[7]  + cm[8]);
        matrix[5] = (cm[11] + cm[1]);
        matrix[7] = (cm[11] - cm[1]);
    }*/

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

static __device__ void transformMolecule(float* x,float* y,float* z,uint natoms,float* matrix,float* translation) 
{ // {{{
    for (uint base = 0; base < natoms; base += blockDim.x) {
        uint atom = base + threadIdx.x;
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

static __device__ void getQuatGradients(float* x,float* y,float* z,uint natoms,float* transform,float* dxdq,float* dxdr,float* dydr,float* dxdu) 
{ // {{{ 
    float invmag2 = 1.0f/(transform[3]*transform[3] + transform[4]*transform[4] + transform[5]*transform[5] + transform[6]*transform[6]);
    #define q transform[3]
    #define r transform[4]
    #define s transform[5]
    #define u transform[6]
    for (uint base = 0; base < natoms; base += blockDim.x) {
        uint atom = base + threadIdx.x;
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

/* devIOV description and prereqs {{{ 
 * Requires 9+nThreads floats (36+4*nThreads bytes) of temp space
 * 
 * Assumes it is given the *untransformed* fit molecule
 *
 * Parallel reduction assumes nThreads is a power of 2
 }}} */
__device__ float devInternalOverlapVolume(float* fitmol_x,float* fitmol_y,float* fitmol_z,float* fitmol_a,uint ourfitcount,
                                        float* refmol_x,float* refmol_y,float* refmol_z,float* refmol_a,uint refatomcount,
                                        float* transform,float* shmem_temp,clock_t* shTimers)
{ // {{{
    
    // Set up our temporary memory usage scheme {{{
    
    float* rotationMatrix = shmem_temp;
    float* Vijs = shmem_temp+9;
    #ifndef NO_DIV_ADDRESS
    uint limit = ourfitcount * refatomcount;
    #endif

    // --- END SHMEM ALLOCATION }}}

    // Initialize partial Vij count
    Vijs[threadIdx.x] = 0.0f;
    __syncthreads();

    // Transform the local copy of the fit molecule {{{
    transformToRotationMatrix(transform,rotationMatrix);
    __syncthreads();
    transformMolecule(fitmol_x,fitmol_y,fitmol_z,ourfitcount,rotationMatrix,transform);
    __syncthreads();
    
    // --- END TRANSFORMATION SETUP }}}

    // Calculate partial overlap volume per thread {{{
    #ifndef NO_DIV_ADDRESS
    for (uint base = 0; base < limit; base += nThreads) {
        uint myidx = base + threadIdx.x;
        if (myidx < limit) {
            uint refatom = myidx/ourfitcount;
            // Replace modulo with a MAD
            uint fitatom = myidx - __umul24(refatom,ourfitcount);
    #else
    const uint startrow = threadIdx.x / ourfitcount;
    const uint startcol = threadIdx.x - __umul24(ourfitcount,startrow);
    //const uint startcol = threadIdx.x % ourfitcount;
    uint refatom = startrow, fitatom = startcol;
    while (refatom < refatomcount) {
        if (true) {
    #endif
        
            float temp,Rij2=0.0f;
            temp = fitmol_x[fitatom] - refmol_x[refatom]; 
            Rij2 += temp*temp;
            temp = fitmol_y[fitatom] - refmol_y[refatom]; 
            Rij2 += temp*temp;
            temp = fitmol_z[fitatom] - refmol_z[refatom]; 
            Rij2 += temp*temp;

            // TODO: cutoff?

            #ifndef FAST_OVERLAP
                temp = __expf(-(refmol_a[refatom])*fitmol_a[fitatom]*Rij2/(refmol_a[refatom]+fitmol_a[fitatom]));
                //temp = __expf(__fdividef(-(refmol_a[refatom])*fitmol_a[fitatom]*Rij2,(refmol_a[refatom]+fitmol_a[fitatom])));

                temp *= 8 * __powf(PI/(refmol_a[refatom]+fitmol_a[fitatom]),1.5f);
                //temp *= 8 * __powf(__fdividef(PI,(refmol_a[refatom]+fitmol_a[fitatom])),1.5f);
                //temp *= 8 * PIRTPI * sqrtf(refmol_a[refatom]+fitmol_a[fitatom])/( (refmol_a[refatom]+fitmol_a[fitatom])*(refmol_a[refatom]+fitmol_a[fitatom]));
            #else
                #ifndef ALL_CARBON_RADII
                    // This code takes fewer clocks but returns numerically different results from the above
                    float sum = refmol_a[refatom]+fitmol_a[fitatom];
                    float inv = 1.0f/sum;
                    float rsqi = rsqrtf(sum);
                    float Kij = __expf(-refmol_a[refatom]*fitmol_a[fitatom]*Rij2*inv);
                    float Vij = 8 * PIRTPI * rsqi * inv * Kij;
                    temp = Vij;
                #else
                    // radius = 1.70, a = 2.41798793102f/(rad*rad)
                    //const float a = 0.836674025f; //2.41798793102f/(1.70f*1.70f);
                    //const float sum = 1.67334805f; // a+a
                    //const float inv = 0.597604306f; // 1/sum
                    //const float prod = 0.700023424f; // a*a
                    //const float rsq = 0.773048709f; // rsqrt(sum)
                    const float npinv = -0.418337012f; // -prod*inv
                    const float Vij_factor = 20.5795263f; // 8 * PIRTPI * rsq * inv
                    float Kij = __expf(npinv*Rij2);
                    float Vij = Vij_factor * Kij;
                    temp = Vij;
                #endif
            #endif

            Vijs[threadIdx.x] += temp;
        }
        #ifdef NO_DIV_ADDRESS
        refatom += row_per_block;
        fitatom += col_per_block;
        if (fitatom >= ourfitcount) {
            fitatom -= ourfitcount;
            refatom++;
        }
        #endif

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

    // Return reduced volume
    return Vijs[0];
} // }}}

/* devIOG description and prereqs {{{
 * Requires 9 + 4*fitcount + 7*nthreads floats of temp space (36 + 16*fitcount + 28*nthreads bytes)
 *
 * Assumes that fitmol has already been transformed by devInternalOverlapVolume.
 * 
 * Parallel reduction assumes nThreads is a power of 2
 }}}*/
__device__ void devInternalOverlapGradient(float* fitmol_x,float* fitmol_y,float* fitmol_z,float* fitmol_a,uint ourfitcount,
                                    float* refmol_x,float* refmol_y,float* refmol_z,float* refmol_a,uint refatomcount,
                                    float* transform,float* gradient,float* shmem_temp,clock_t* shTimers)
{ // {{{


    // Set up our temporary memory usage scheme {{{
    
    #ifndef NO_DIV_ADDRESS
    uint limit = ourfitcount * refatomcount;
    #endif

    float* dxdq = shmem_temp;
    float* dxdr = shmem_temp + ourfitcount;
    float* dydr = shmem_temp + 2*ourfitcount;
    float* dxdu = shmem_temp + 3*ourfitcount;

    // We have nThreads * sizeof(float) partial Vijs, and same for each gradient cord
    float* ddxs = shmem_temp + 4*ourfitcount;
    float* ddys = shmem_temp + 4*ourfitcount + nThreads;
    float* ddzs = shmem_temp + 4*ourfitcount + 2*nThreads;
    float* ddqs = shmem_temp + 4*ourfitcount + 3*nThreads;
    float* ddrs = shmem_temp + 4*ourfitcount + 4*nThreads;
    float* ddss = shmem_temp + 4*ourfitcount + 5*nThreads;
    float* ddus = shmem_temp + 4*ourfitcount + 6*nThreads;

    // -- END SHMEM ALLOCATION }}}

    // Initialize shared memory {{{
    // Initialize partial counts
    ddxs[tidx] = 0.0f;
    ddys[tidx] = 0.0f;
    ddzs[tidx] = 0.0f;
    ddqs[tidx] = 0.0f;
    ddrs[tidx] = 0.0f;
    ddss[tidx] = 0.0f;
    ddus[tidx] = 0.0f;
    
    __syncthreads();

    // -------------- END SHARED MEMORY INITIALIZATION ---------------------- }}}

    // Precalculate four quaternion gradients (others are equal to +-1 times these) {{{
    getQuatGradients(fitmol_x,fitmol_y,fitmol_z,ourfitcount,transform,dxdq,dxdr,dydr,dxdu);
    __syncthreads();
    // -- END QUATERNION CHAIN RULE SETUP }}}

    // Calculate partial overlap volume and gradient per thread {{{
    #ifndef NO_DIV_ADDRESS
    for (uint base = 0; base < limit; base += nThreads) {
        uint myidx = base + threadIdx.x;
        if (myidx < limit) {
            uint refatom = myidx/ourfitcount;
            // Replace modulo with a MAD
            uint fitatom = myidx - __umul24(refatom,ourfitcount);
    #else
    const uint startrow = threadIdx.x / ourfitcount;
    const uint startcol = threadIdx.x - __umul24(ourfitcount,startrow);
    //const uint startcol = threadIdx.x % ourfitcount;
    uint refatom = startrow, fitatom = startcol;
    while (refatom < refatomcount) {
        if (true) {
    #endif
        
            float temp,Rij2=0.0f;
            temp = fitmol_x[fitatom] - refmol_x[refatom]; 
            Rij2 += temp*temp;
            temp = fitmol_y[fitatom] - refmol_y[refatom]; 
            Rij2 += temp*temp;
            temp = fitmol_z[fitatom] - refmol_z[refatom]; 
            Rij2 += temp*temp;

            // TODO: cutoff?

            // temp is the Vij
            #ifndef FAST_OVERLAP
                temp = __expf(-(refmol_a[refatom])*fitmol_a[fitatom]*Rij2/(refmol_a[refatom]+fitmol_a[fitatom]));
                temp *= 8 * __powf(PI/(refmol_a[refatom]+fitmol_a[fitatom]),1.5f);
                //temp = __expf(__fdividef(-(refmol_a[refatom])*fitmol_a[fitatom]*Rij2,(refmol_a[refatom]+fitmol_a[fitatom])));
                //temp *= 8 * __powf(__fdividef(PI,(refmol_a[refatom]+fitmol_a[fitatom])),1.5f);

                // now transform temp into the scalar coefficient on the gradient calc
                temp *= (-2)*refmol_a[refatom]*fitmol_a[fitatom]/(refmol_a[refatom]+fitmol_a[fitatom]);
                //temp *= __fdividef((-2)*refmol_a[refatom]*fitmol_a[fitatom],(refmol_a[refatom]+fitmol_a[fitatom]));
            #else
                #ifndef ALL_CARBON_RADII    
                    // This code takes fewer clocks but returns numerically different results from the above
                    float sum = refmol_a[refatom]+fitmol_a[fitatom];
                    float inv = 1.0f/sum;
                    float prodinv = refmol_a[refatom]*fitmol_a[fitatom]*inv;
                    float rsqi = rsqrtf(sum);
                    float Kij = __expf(-Rij2*prodinv);
                    float Vij = 8 * PIRTPI * rsqi * inv * Kij;
                    temp = -2.0f * Vij * prodinv;
                #else
                    // radius = 1.70, a = 2.41798793102f/(rad*rad)
                    //const float a = 0.836674025f; //2.41798793102f/(1.70f*1.70f);
                    //const float sum = 1.67334805f; // a+a
                    //const float inv = 0.597604306f; // 1/sum
                    //const float prod = 0.700023424f; // a*a
                    //const float rsq = 0.773048709f; // rsqrt(sum)
                    const float npinv = -0.418337012f; // -prod*inv
                    //const float npinv2 = -0.836674024f; // -prod*inv*2
                    //const float Vij_factor = 20.5795263f; // 8 * PIRTPI * rsq * inv
                    const float dVij_factor = -17.2183551f; // npinv2 * Vij_factor
                    float Kij = __expf(npinv*Rij2);
                    //float Vij = Vij_factor * Kij;
                    temp = dVij_factor*Kij;
                #endif

            #endif
            
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
        #ifdef NO_DIV_ADDRESS
        refatom += row_per_block;
        fitatom += col_per_block;
        if (fitatom >= ourfitcount) {
            fitatom -= ourfitcount;
            refatom++;
        }
        #endif
    }
    __syncthreads();
    // -- END MAIN CALCULATION LOOP }}}

    // Parallel-reduce overlap and gradients {{{
    // Modified from (buggy) lecture slides of Hendrik Lensch
    // http://www.mpi-inf.mpg.de/departments/d4/teaching/ws2007-2008/cg/slides/CG20-Cuda2.pdf
    for (uint stride = nThreads>>1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (threadIdx.x < stride) {
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

    // Write results back  {{{

    /*switch (tidx) {
        case 0: gradient[0] = ddxs[0]; break;
        case 1: gradient[1] = ddys[0]; break;
        case 2: gradient[2] = ddzs[0]; break;
        case 3: gradient[3] = ddqs[0]; break;
        case 4: gradient[4] = ddrs[0]; break;
        case 5: gradient[5] = ddss[0]; break;
        case 6: gradient[6] = ddus[0]; break;
        default: break;
    }*/
    if (threadIdx.x == 0) {
        gradient[0] = ddxs[0];
        gradient[1] = ddys[0];
        gradient[2] = ddzs[0];
        gradient[3] = ddqs[0];
        gradient[4] = ddrs[0];
        gradient[5] = ddss[0];
        gradient[6] = ddus[0];
    }
    __syncthreads();

    // -- END RESULTS COMMIT }}}

    return;

} //}}}

/* devIO description and prereqs {{{
 * Requires 9+nThreads floats (36+4*nThreads bytes) of temp space
 * 
 * Assumes it is given the *untransformed* fit molecule
 *
 * Parallel reduction assumes nThreads is a power of 2
 }}}*/
__device__ float devInternalObjective(dCUDAmol fitmol_orig,dCUDAmol fitmol,dCUDAmol refmol,float* transform,float* shmem_temp,clock_t* shTimers, bool addPenalty = true)
{ //{{{
    clock_t start = clock();
    // Copy the original molecule back into fitmol before running overlap function {{{ 
    for (uint base = 0; base < fitmol.natoms; base += nThreads) {
        uint atom = base + tidx;
        if (atom < fitmol.natoms) {
            fitmol.x[atom] = fitmol_orig.x[atom];
            fitmol.y[atom] = fitmol_orig.y[atom];
            fitmol.z[atom] = fitmol_orig.z[atom];
        }
    }
    __syncthreads(); // }}}

    float objective = -1 * devInternalOverlapVolume(fitmol.x,fitmol.y,fitmol.z,fitmol.a,fitmol.natoms,
                                                    refmol.x,refmol.y,refmol.z,refmol.a,refmol.natoms,
                                                    transform,shmem_temp,shTimers);

    // Add the penalty term
    if (addPenalty) {
        float magp = sqrtf(transform[3]*transform[3] + transform[4]*transform[4] + transform[5]*transform[5] + transform[6]*transform[6]);
        float penalty = 0.5f*PENALTY_GAMMA*(magp-1)*(magp-1);
        objective += penalty;
    }
    
    clock_t elapsed = clock() - start;
    if (d_optparms.num_timers >= 4 && threadIdx.x == 0) {
        shTimers[2] += elapsed;
        shTimers[3]++;
    }

    return objective;

} //}}}

/* devIG description and prereqs {{{
 * Requires 9 + 4*fitcount + 7*nthreads floats of temp space (36 + 16*fitcount + 28*nthreads bytes)
 *
 * Assumes that fitmol has already been transformed by devInternalOverlapVolume.
 * 
 * Parallel reduction assumes nThreads is a power of 2
 }}}*/
__device__ void devInternalGradient(dCUDAmol fitmol,dCUDAmol refmol,float* transform,float* gradient,float* shmem_temp,clock_t* shTimers)
{ //{{{
    clock_t start = clock();
    devInternalOverlapGradient(fitmol.x,fitmol.y,fitmol.z,fitmol.a,fitmol.natoms,
                               refmol.x,refmol.y,refmol.z,refmol.a,refmol.natoms,
                               transform,gradient,shmem_temp,shTimers);

    
    float invmagp = rsqrtf(transform[3]*transform[3] + transform[4]*transform[4] + transform[5]*transform[5] + transform[6]*transform[6]);
    if (tidx >= 3 && tidx < 7) {
        gradient[tidx] += PENALTY_GAMMA*(1-invmagp)*transform[tidx];
    }
    __syncthreads();
    clock_t elapsed = clock() - start;
    if (d_optparms.num_timers >= 8 && threadIdx.x == 0) {
        shTimers[6] += elapsed;
        shTimers[7]++;
    }
    return;

} //}}}

__device__ float dot7(float* a,float* b,float* t) { //{{{
    if (tidx < 7) {
        t[tidx] = a[tidx]*b[tidx];
    }
    __syncthreads();
    // TODO: see if this can be profitably parallelized
    float retval = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6];
    // We need to barrier here because otherwise the next call may start overwriting the temp space we used in this call
    __syncthreads();
    return retval;
} //}}}

__device__ float max7(float* a,float* t) { //{{{
    //TODO: see if this is worth parallelizing
    float retval = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(a[6],a[5]),a[4]),a[3]),a[2]),a[1]),a[0]);
    __syncthreads();
    return retval;
} //}}}

__device__ void copy7(float* dest,float* src) { //{{{
    if (tidx < 7)
        dest[tidx] = src[tidx];
    __syncthreads();
} //}}}

__device__ void scal7(float k,float* v) { //{{{
    if (tidx < 7)
        v[tidx] = k*v[tidx];
    __syncthreads();
} //}}}

__device__ void add7(float* dest,float* a,float k2,float* b) { //{{{
    if (tidx < 7)
        dest[tidx] = a[tidx] + k2*b[tidx];
    __syncthreads();
} //}}}

__device__ void add7(float* dest,float k1,float* a,float k2,float* b) { //{{{
    if (tidx < 7) {
        dest[tidx] = k1*a[tidx] + k2*b[tidx];
    }
    __syncthreads();
} //}}}

__device__ void sgemv7(float* dest,float* m,float* v,float* temp) { //{{{
    for (uint i = 0; i < 7; i++) {
        if (tidx < 7) {
            temp[tidx] = m[i*7+tidx]*v[tidx];
        }
        __syncthreads();
        if (tidx == 0)
            dest[i] = temp[0]+temp[1]+temp[2]+temp[3]+temp[4]+temp[5]+temp[6];
        __syncthreads();
    }
} //}}}

__host__ void sgemv7(float* dest,const float* m,const float* v) { //{{{
    for (uint i = 0; i < 7; i++) {
        dest[i] = 0;
        for (int j = 0; j < 7; j++) {
            dest[i] += m[i*7+j]*v[j];
        }
    }
    return;
} //}}}

__device__ void sub7(float* dest,float* a,float* b) { //{{{
    if (tidx < 7)
        dest[tidx] = a[tidx] - b[tidx];
    __syncthreads();
} //}}}

/*
 * Requires max(7,objective temp) floats of temp space
 */
__device__ float lineSearch(dCUDAmol fitmol_orig,dCUDAmol fitmol,dCUDAmol refmol,
                            float* xorig,float forig,float* xnew,float* gradient,float* direction,
                            float maxstep,float* temp,clock_t* shTimers,
                            const float alpha = 1e-4f, const float tolerance = 1e-5f, const uint iter_max=100)

{ //{{{
    // Initialize line search {{{
    // lambdas represent step sizes
    float lambda,lambda_new,lambda_old,lambda_cvg;
    lambda = 1.0f;
    
    if (false) {
        // Re-project the rotational component of the search direction so that
        // it's tangent to the hypersphere surface (remove radial component)
        if (tidx == 0) {
            float dot = direction[3]*xorig[3] + direction[4]*xorig[4] + direction[5]*xorig[5] + direction[6]*xorig[6];
            float mag2 = xorig[3]*xorig[3] + xorig[4]*xorig[4] + xorig[5]*xorig[5] + xorig[6]*xorig[6];
            for (int i = 3; i < 6; i++)
                direction[i] = direction[i] - (dot/mag2)*xorig[i];
        }
        __syncthreads();
    }

    // If the magnitude of the search vector is large enough such that the Newton step exceeds our max step, scale it down
    float magdir2 = direction[0]*direction[0] + direction[1]*direction[1] + direction[2]*direction[2] + direction[3]*direction[3] + direction[4]*direction[4] + direction[5]*direction[5] + direction[6]*direction[6];
    __syncthreads();
    if (magdir2 > maxstep*maxstep) {
        magdir2 = rsqrtf(magdir2);
        if (tidx < 7) {
            direction[tidx] *= maxstep*magdir2;
        }
        __syncthreads();
    }
    float slope = dot7(gradient,direction,temp);
    // If the function is increasing in the search direction, someone screwed up. Roundoff error?
    // TODO: is this what we really want?
    if (slope > 0) {
        copy7(xnew,xorig);
        return forig;
    }

    float fold = forig;
    float f = forig;

    // Here just using xnew as some extra temp space
    if (tidx < 7) {
        xnew[tidx] = fabsf(direction[tidx])/fmaxf(1.0f,fabsf(xorig[tidx]));
    }
    __syncthreads();
    lambda_cvg = tolerance/max7(xnew,temp);

    // end initialization }}}

    // Iterate until we find a point of sufficient decrease, converge, or run into roundoff errors //{{{
    for(uint iter = 0;;iter++) {
        __syncthreads();
        // Return if we've converged on the x values
        if (lambda < lambda_cvg || iter > iter_max) {
            //copy7(xnew,xorig);
            //return forig;
            return f;
        }

        fold = f;
        // x = xorig+lambda*direction
        add7(xnew,xorig,lambda,direction);

        //float iqmag = rsqrtf(xnew[3]*xnew[3] + xnew[4]*xnew[4] + xnew[5]*xnew[5] + xnew[6]*xnew[6]);
        //if (threadIdx.x < 4) xnew[3+threadIdx.x] *= iqmag;
        //__syncthreads();
        
        f = devInternalObjective(fitmol_orig,fitmol,refmol,xnew,temp,shTimers);

        // Return if we've achieved sufficient decrease
        if ( f <= (forig + alpha*lambda*slope)) {
            return f;
        }
        // Did not converge or achieve sufficient decrease, so backtrack

        if (lambda == 1.0f) {
            // This was the first step, so use a quadratic approximation to the function
            lambda_new = -slope / (2*(f-fold-slope));
        } else {
            // This was at least the second step; use a cubic approximation
            float k1 = f - lambda*slope - forig;
            float k2 = fold - lambda_old*slope - forig;
            float a = (1/(lambda-lambda_old)) * ( (1/(lambda*lambda)) * k1 - (1/(lambda_old*lambda_old)) * k2);
            float b = (1/(lambda-lambda_old)) * ( -(lambda_old/(lambda*lambda)) * k1 + (lambda/(lambda_old*lambda_old)) * k2);
            float discriminant = b*b -3*a*slope;
            if (a == 0) {
                // The cubic coefficient is zero, so find the max of the corresponding quadratic
                lambda_new = -slope/(2*b);
            } else if (discriminant < 0) {
                // Roundoff error issues here
                // TODO: how to deal with this?
                //copy7(xnew,xorig);
                //return forig;
                lambda_new = lambda;
            } else {
                // We can actually use the cubic approx
                lambda_new = (-b + sqrtf(discriminant))/(3*a);
            }
        }
      
      
        lambda_new = fmaxf(0.1f*lambda,lambda_new);
        lambda_new = fminf(lambda_new,0.5f*lambda);
        lambda_old = lambda;
        lambda = lambda_new;
    } //}}}
} //}}}

__device__ float* load_molecules(const dCUDAMultimol fitmols, const dCUDAMultimol refMultimol,
                                 dCUDAmol& fitmol_orig, dCUDAmol& fitmol, dCUDAmol& refmol, float* shmem, bool orig_global=false)
{ //{{{ 
    float* temp;
    fitmol.x = shmem;
    fitmol.y = shmem + fitmols.maxatoms;
    fitmol.z = shmem + 2*fitmols.maxatoms;
    fitmol.a = shmem + 3*fitmols.maxatoms;
    
    refmol.x = shmem + 4*fitmols.maxatoms;
    refmol.y = shmem + 4*fitmols.maxatoms + refMultimol.maxatoms;
    refmol.z = shmem + 4*fitmols.maxatoms + 2*refMultimol.maxatoms;
    refmol.a = shmem + 4*fitmols.maxatoms + 3*refMultimol.maxatoms;

    #ifdef ORIG_GLOBAL
    orig_global = true;
    #endif


    if (!orig_global) {
        fitmol_orig.x = shmem + 4*fitmols.maxatoms + 4*refMultimol.maxatoms; 
        fitmol_orig.y = shmem + 4*fitmols.maxatoms + 4*refMultimol.maxatoms + fitmols.maxatoms; 
        fitmol_orig.z = shmem + 4*fitmols.maxatoms + 4*refMultimol.maxatoms + 2*fitmols.maxatoms; 
        fitmol_orig.a = shmem + 4*fitmols.maxatoms + 4*refMultimol.maxatoms + 3*fitmols.maxatoms; 
        temp          = shmem + 4*fitmols.maxatoms + 4*refMultimol.maxatoms + 4*fitmols.maxatoms; 
    } else {
        fitmol_orig.x = fitmols.mols + fitmolIdx*fitmols.pitch*4;
        fitmol_orig.y = fitmols.mols + fitmolIdx*fitmols.pitch*4 + fitmols.pitch;
        fitmol_orig.z = fitmols.mols + fitmolIdx*fitmols.pitch*4 + 2*fitmols.pitch;
        fitmol_orig.a = fitmols.mols + fitmolIdx*fitmols.pitch*4 + 3*fitmols.pitch;
        temp          = shmem + 4*fitmols.maxatoms + 4*refMultimol.maxatoms; 
    }

    
    // Load number of atoms in each molecule
    fitmol.natoms = fitmols.atomcounts[fitmolIdx];
    refmol.natoms = refMultimol.maxatoms;
    fitmol_orig.natoms = fitmol.natoms;

    // Initialize elements used for overlap/gradient loop addressing
    if (threadIdx.x == 0) {
        row_per_block = blockDim.x / fitmol.natoms;
        //col_per_block = blockDim.x % fitmol.natoms;
        col_per_block = blockDim.x - fitmol.natoms * row_per_block;
    }

    // Load reference molecule
    for (int base = 0; base < refmol.natoms; base += blockDim.x) {
        int atomidx = base + threadIdx.x;
        if (atomidx < refmol.natoms) {
            // Molecule k starts at 4*refmol_pitch*k (4 rows per molecule) for x row
            refmol.x[atomidx] = refMultimol.mols[atomidx];
            refmol.y[atomidx] = refMultimol.mols[(refMultimol.pitch)   + atomidx];
            refmol.z[atomidx] = refMultimol.mols[(2*refMultimol.pitch) + atomidx];
            refmol.a[atomidx] = refMultimol.mols[(3*refMultimol.pitch) + atomidx];
        }
    }

    // Load original version of fit molecule
    if (!orig_global) {
        for (int base = 0; base < fitmol.natoms; base+=blockDim.x) {
            int atomidx = base+tidx;
            if (atomidx < fitmol_orig.natoms) {
                // Molecule k starts at 4*fitmol_pitch*k (4 rows per molecule) for x row
                fitmol_orig.x[atomidx] = fitmols.mols[(fitmolIdx*fitmols.pitch*4) + atomidx];
                fitmol_orig.y[atomidx] = fitmols.mols[(fitmolIdx*fitmols.pitch*4) + (fitmols.pitch)   + atomidx];
                fitmol_orig.z[atomidx] = fitmols.mols[(fitmolIdx*fitmols.pitch*4) + (2*fitmols.pitch) + atomidx];
                fitmol_orig.a[atomidx] = fitmols.mols[(fitmolIdx*fitmols.pitch*4) + (3*fitmols.pitch) + atomidx];
            }
        }   
    }

    // Load fit alphas, since we do not copy those later
    for (int base = 0; base < fitmol.natoms; base+=blockDim.x) {
        int atomidx = base+tidx;
        if (atomidx < fitmol_orig.natoms) fitmol.a[atomidx] = fitmol_orig.a[atomidx];
    }

    __syncthreads();

    return temp;
} //}}}

__global__ void init_dev_BFGS(void)
{ //{{{
    // Initialize BFGS
    extern __shared__ float shmem[];
    __shared__ float gradient[7];
    __shared__ float transform[7];
    // TODO: this should not be statically defined!
    const int numTimers = 8;
    __shared__ clock_t shTimers[numTimers];
    dCUDAmol fitmol_orig, fitmol, refmol;

    if (threadIdx.x < numTimers) shTimers[threadIdx.x] = 0;

    // Initialize molecule pointers and shared temp memory
    float* temp = load_molecules(d_optparms.fitmols,d_optparms.refmol,fitmol_orig,fitmol,refmol,shmem,true);

    //      - init objective and gradient at starting point
    if (threadIdx.x < 7) transform[threadIdx.x] = d_optparms.fitmols.transforms[fitmolIdx*d_optparms.fitmols.transform_pitch + threadIdx.x];
    __syncthreads();
    // Needs 9+nThreads floats temp
    float f = devInternalObjective(fitmol_orig,fitmol,refmol,transform,temp,shTimers);
    //  9 + 4*fitcount + 7*nthreads floats of temp space
    devInternalGradient(fitmol,refmol,transform,gradient,temp,shTimers);

    if (threadIdx.x == 0) d_optparms.score[fitmolIdx] = f;
    if (threadIdx.x < 7)  d_optparms.gradient[fitmolIdx*d_optparms.fitmols.transform_pitch + threadIdx.x] = gradient[threadIdx.x];
    
    //      - init starting direction to -grad
    if (threadIdx.x < 7) d_optparms.search_direction[fitmolIdx*d_optparms.fitmols.transform_pitch + threadIdx.x] = -gradient[threadIdx.x];

    //      - init bfgs_maxstep
    float maxstep = d_optparms.bfgs_STPMX * fmaxf(7.0f,sqrtf(dot7(transform,transform,temp)));
    if (threadIdx.x == 0) d_optparms.ls_maxstep[fitmolIdx] = maxstep;

    //      - initialize completion flag to 0
    if (threadIdx.x == 0) d_optparms.completed[fitmolIdx] = 0;
    //      - init inverse Hessian to the identity matrix
    if (threadIdx.x < 49) shmem[threadIdx.x] = 0;
    __syncthreads();
    if (threadIdx.x < 7)  shmem[threadIdx.x*7+threadIdx.x] = 1;
    __syncthreads();
    if (threadIdx.x < 49) d_optparms.invH[d_optparms.hessian_pitch*fitmolIdx + threadIdx.x] = shmem[threadIdx.x];

    return;
} //}}}

__global__ void line_search_kernel(void)
{ //{{{
    // Do line search and check for x convergence
    extern __shared__ float shmem[];
    __shared__ float transform[7];
    __shared__ float direction[7];
    __shared__ float new_transform[7];
    __shared__ float gradient[7];
    // TODO: this should not be statically defined!
    const int numTimers = 8;
    __shared__ clock_t shTimers[numTimers];
    __shared__ float forig, maxstep;
    __shared__ int completed;

    // Terminate block immediately if we're done with this molecule
    if (threadIdx.x == 0) completed = d_optparms.completed[fitmolIdx];
    __syncthreads();
    if (completed) return;

    // Init timing variables
    clock_t* gTimers = d_optparms.timers + blockIdx.x*d_optparms.timer_pitch;
    if (threadIdx.x < d_optparms.num_timers) shTimers[threadIdx.x] = 0;
    if (d_optparms.num_timers > 0 && threadIdx.x == 0) shTimers[1] = 1;
    clock_t start = clock();

    dCUDAmol fitmol, fitmol_orig, refmol;
    float *temp = load_molecules(d_optparms.fitmols, d_optparms.refmol, fitmol_orig, fitmol, refmol, shmem,false);

    // Load starting coordinates, directions, etc.
    if (threadIdx.x < 7) {
        transform[threadIdx.x] = d_optparms.fitmols.transforms[fitmolIdx*d_optparms.fitmols.transform_pitch + threadIdx.x];
        direction[threadIdx.x] = d_optparms.search_direction[fitmolIdx*d_optparms.fitmols.transform_pitch + threadIdx.x];
        gradient[threadIdx.x]  = d_optparms.gradient[fitmolIdx*d_optparms.fitmols.transform_pitch + threadIdx.x];
    }
    if (threadIdx.x == 0) {
        forig = d_optparms.score[fitmolIdx];
        maxstep = d_optparms.ls_maxstep[fitmolIdx];
    }
    __syncthreads();

    float fnew = lineSearch(fitmol_orig,fitmol,refmol,transform,forig,new_transform,gradient,direction,maxstep,temp,shTimers,
                            d_optparms.ls_alpha,d_optparms.ls_tol,d_optparms.ls_max_iters);

    // Set the direction vector to the actual magnitude of the step we took and commit new coordinate
    if (threadIdx.x < 7) {
        d_optparms.fitmols.transforms[fitmolIdx*d_optparms.fitmols.transform_pitch + threadIdx.x] = new_transform[threadIdx.x];
        direction[threadIdx.x] = new_transform[threadIdx.x] - transform[threadIdx.x];
        temp[threadIdx.x] = fabsf(direction[threadIdx.x])/fmaxf(1.0f,fabsf(new_transform[threadIdx.x]));
        d_optparms.search_direction[fitmolIdx*d_optparms.fitmols.transform_pitch + threadIdx.x] = direction[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        d_optparms.score[fitmolIdx] = fnew;
    }
    __syncthreads();

    // Check for convergence on x values
    float test = max7(temp,temp+7);
    if (test < d_optparms.bfgs_tol_coord) {
        if (threadIdx.x == 0) d_optparms.completed[fitmolIdx] = 1;
    }
    // Commit timer values to global mem
    if (d_optparms.num_timers > 0 && threadIdx.x == 0) shTimers[0] = clock() - start;
    __syncthreads();
    if (threadIdx.x < d_optparms.num_timers) {
        gTimers[threadIdx.x] += shTimers[threadIdx.x];
    }
    return;
} //}}}

__global__ void gradient_kernel(void)
{ //{{{
    // evaluate gradient, check convergence, and update direction for next line search
    extern __shared__ float shmem[];
    __shared__ float transform[7];
    __shared__ float direction[7];
    __shared__ float gradient[7];
    __shared__ float old_gradient[7];
    __shared__ float invH[49];
    __shared__ float hdg[7];
    __shared__ float rotationMatrix[9];
    // TODO: this should not be statically defined!
    const int numTimers = 8;
    __shared__ clock_t shTimers[numTimers];
    __shared__ float forig;
    __shared__ int completed;

    float test = 0;
    float inv_alpha = 0;
    float magdg2 = 0;
    float magdir2 = 0;

    // Terminate block immediately if we're done with this molecule
    if (threadIdx.x == 0) completed = d_optparms.completed[fitmolIdx];
    __syncthreads();
    if (completed) return;

    // Init timing variables
    clock_t* gTimers = d_optparms.timers + blockIdx.x*d_optparms.timer_pitch;
    if (threadIdx.x < d_optparms.num_timers) shTimers[threadIdx.x] = 0;
    if (d_optparms.num_timers > 0 && threadIdx.x == 0) shTimers[5] = 1;
    clock_t start = clock();

    dCUDAmol fitmol, fitmol_orig, refmol;
    float *temp = load_molecules(d_optparms.fitmols, d_optparms.refmol, fitmol_orig, fitmol, refmol, shmem,true);

    // Load starting coordinates, directions, etc.
    if (threadIdx.x < 49) {
        invH[threadIdx.x] = d_optparms.invH[d_optparms.hessian_pitch*fitmolIdx + threadIdx.x];
    }
    if (threadIdx.x < 7) {
        transform[threadIdx.x] = d_optparms.fitmols.transforms[fitmolIdx*d_optparms.fitmols.transform_pitch + threadIdx.x];
        old_gradient[threadIdx.x] = d_optparms.gradient[fitmolIdx*d_optparms.fitmols.transform_pitch + threadIdx.x];
        direction[threadIdx.x] = d_optparms.search_direction[fitmolIdx*d_optparms.fitmols.transform_pitch + threadIdx.x];
    }
    if (threadIdx.x == 0) {
        forig = d_optparms.score[fitmolIdx];
    }
    __syncthreads();

    // Transform the original molecule (gradient expects *transformed* input)
    transformToRotationMatrix(transform,rotationMatrix);
    __syncthreads();
    for (int base = 0; base < fitmol.natoms; base += blockDim.x) {
        int atom = base + threadIdx.x;
        if (atom < fitmol.natoms) {
            float oldx = fitmol_orig.x[atom];
            float oldy = fitmol_orig.y[atom];
            float oldz = fitmol_orig.z[atom];
            fitmol.x[atom] = oldx*rotationMatrix[0*3+0] + oldy*rotationMatrix[0*3+1] + oldz*rotationMatrix[0*3+2] + transform[0];
            fitmol.y[atom] = oldx*rotationMatrix[1*3+0] + oldy*rotationMatrix[1*3+1] + oldz*rotationMatrix[1*3+2] + transform[1];
            fitmol.z[atom] = oldx*rotationMatrix[2*3+0] + oldy*rotationMatrix[2*3+1] + oldz*rotationMatrix[2*3+2] + transform[2];
        }
    }
    __syncthreads();

    // Evaluate the gradient at the new point
    devInternalGradient(fitmol,refmol,transform,gradient,temp,shTimers);

    if (d_optparms.cpu_bfgs_update) goto function_end;
        
    // Check for convergence on gradient=0 {{{
    if (threadIdx.x < 7) temp[tidx] = (fabsf(gradient[tidx])*fmaxf(1.0f,fabsf(transform[tidx])))/fmaxf(1.0f,forig);
    __syncthreads();
    test = max7(temp,temp+7);
    if (test < d_optparms.bfgs_tol_gradient) {
        if (threadIdx.x == 0) d_optparms.completed[fitmolIdx] = 2;
        // Use the goto to handle teardown (timers, etc)
        goto function_end;
    } //}}}

    // Update search direction for next step

    // Evaluate the change in gradient from our last step
    //float* dg = old_gradient;
    __shared__ float dg[7];
    sub7(dg,gradient,old_gradient);

    inv_alpha = dot7(dg,direction,temp);
    magdg2 = dot7(dg,dg,temp);
    magdir2 = dot7(direction,direction,temp);

    if (inv_alpha*inv_alpha > d_optparms.bfgs_epsilon*magdg2*magdir2) {
        // Update the inverse Hessian matrix
        sgemv7(hdg,invH,dg,temp);
            
        float inv_beta = dot7(dg,hdg,temp);
        float beta = 1/inv_beta;
        // ok, so inv_alpha is actually alpha now
        inv_alpha = 1/inv_alpha;

        // Reuse dg to store the BFGS update vector
        add7(dg,inv_alpha,direction,-beta,hdg);

        // Update the inverse Hessian with sum of scaled vector outer products
        for (uint i = 0; i < 7; i++) {
            if (tidx < 7) {
                invH[i*7+tidx] += inv_alpha*(direction[i]*direction[tidx]) - beta*(hdg[i]*hdg[tidx]) + inv_beta*(dg[i]*dg[tidx]);
            }
        }
        __syncthreads();
    }
    
    // Set the search direction for the next iteration
    sgemv7(direction,invH,gradient,temp);
    //copy7(direction,gradient);
    scal7(-1,direction);

    // Commit new direction, gradient, and inverse Hessian to gmem
    if (threadIdx.x < 49) {
        d_optparms.invH[d_optparms.hessian_pitch*fitmolIdx + threadIdx.x] = invH[threadIdx.x];
    }
    if (threadIdx.x < 7) {
        d_optparms.gradient[fitmolIdx*d_optparms.fitmols.transform_pitch + threadIdx.x] = gradient[threadIdx.x];
        d_optparms.search_direction[fitmolIdx*d_optparms.fitmols.transform_pitch + threadIdx.x] = direction[threadIdx.x];
    }

    function_end:
    // Commit timer values to global mem
    if (d_optparms.num_timers > 0 && threadIdx.x == 0) shTimers[4] = clock() - start;
    __syncthreads();
    if (threadIdx.x < d_optparms.num_timers) {
        gTimers[threadIdx.x] += (uint64_t) shTimers[threadIdx.x];
    }

    return;
} //}}}

__global__ void final_objective_kernel(void)
{ //{{{
    // Do line search and check for x convergence
    extern __shared__ float shmem[];
    __shared__ float transform[7];
    // TODO: this should not be statically defined!
    const int numTimers = 8;
    __shared__ clock_t shTimers[numTimers];

    if (threadIdx.x < numTimers) shTimers[threadIdx.x] = 0;

    dCUDAmol fitmol, fitmol_orig, refmol;
    float *temp = load_molecules(d_optparms.fitmols, d_optparms.refmol, fitmol_orig, fitmol, refmol, shmem,true);

    // Load starting coordinates, directions, etc.
    if (threadIdx.x < 7) {
        transform[threadIdx.x] = d_optparms.fitmols.transforms[fitmolIdx*d_optparms.fitmols.transform_pitch + threadIdx.x];
    }
    __syncthreads();

    // Normalize quaternion
    float imagq = rsqrtf(transform[3]*transform[3] + transform[4]*transform[4] + transform[5]*transform[5] + transform[6]*transform[6]);
    if (threadIdx.x < 4) transform[3+threadIdx.x] *= imagq;
    __syncthreads();

    // Calculate final objective
    float obj = -devInternalObjective(fitmol_orig, fitmol, refmol, transform, temp, shTimers, false);

    // Commit objective and transform
    if (threadIdx.x < 7) {
        d_optparms.fitmols.transforms[fitmolIdx*d_optparms.fitmols.transform_pitch + threadIdx.x] = transform[threadIdx.x];
    }
    if (threadIdx.x == 0) d_optparms.score[fitmolIdx] = obj;

    return;
} //}}}

__host__ int optimize_sepkernels(dCUDAMultimol d_fitmols, dCUDAMultimol d_refMultimol, float* finalScores, clock_t *h_timers, const uint numTimers,const float3 com_ref,const float3 *com_fit)
{ //{{{
    PAPER_data h_optparms;
    cudaError_t cudaError;
    const bool debug = false;
    // There's no reason for this to be true unless you're doing weird benchmarking
    const bool cpu_bfgs_update = false;
    const bool tuneParms = false;
    const bool enableTiming = true;

    // Set up constant structure {{{
    // Memory allocations //{{{
    //      Refmol
    h_optparms.refmol = d_refMultimol;
    //      Fitmol
    h_optparms.fitmols = d_fitmols;
    //      Timers
    if (enableTiming) { //{{{
        size_t timer_pitch;
        cudaMallocPitch((void**)&h_optparms.timers,&timer_pitch,numTimers*sizeof(clock_t),d_fitmols.nmols);
        h_optparms.timer_pitch = timer_pitch / sizeof(clock_t);
        h_optparms.num_timers = numTimers;
    } else {
        h_optparms.timers = NULL;
        h_optparms.timer_pitch = 0;
        h_optparms.num_timers = 0;
    } //}}}

    //      Gradients  (line search/BFGS)
    cudaMalloc((void**)&h_optparms.gradient,h_optparms.fitmols.transform_pitch * d_fitmols.nmols * sizeof(float));
    //      Directions (line search/BFGS)
    cudaMalloc((void**)&h_optparms.search_direction,h_optparms.fitmols.transform_pitch * d_fitmols.nmols * sizeof(float));
    //      Start scores
    cudaMalloc((void**)&h_optparms.score,d_fitmols.nmols * sizeof(float));
    //      Completion flags
    cudaMalloc((void**)&h_optparms.completed,d_fitmols.nmols * sizeof(int));
    //int *completed = new int[d_fitmols.nmols];
    int *completed;
    cudaMallocHost((void**)&completed,d_fitmols.nmols*sizeof(int));

    //      Maximum step size in line search
    cudaMalloc((void**)&h_optparms.ls_maxstep,d_fitmols.nmols * sizeof(float));
    //      Inverse Hessian (BFGS)
    size_t invHpitch;
    cudaMallocPitch((void**)&h_optparms.invH,&invHpitch,49*sizeof(float),d_fitmols.nmols);
    h_optparms.hessian_pitch = invHpitch/sizeof(float);
    // }}}

    // Set up optimization parameters {{{
    // BFGS gradient tolerance
    h_optparms.bfgs_tol_gradient = 0.01;
    // BFGS maximum iterations
    h_optparms.bfgs_max_iters = 10;
    // BFGS maximum step size factor
    h_optparms.bfgs_STPMX = 100;
    // BFGS precision
    h_optparms.bfgs_epsilon = 3.0e-06;
    // BFGS coordinate tolerance
    h_optparms.bfgs_tol_coord = 4*h_optparms.bfgs_epsilon;
    // LS step ratio
    h_optparms.ls_alpha = 1.0e-4;
    // LS convergence tolerance
    h_optparms.ls_tol = 1.0e-5;
    // LS maximum iterations
    h_optparms.ls_max_iters = 100;

    // Will the CPU handle BFGS invH/direction updates instead of the GPU?
    h_optparms.cpu_bfgs_update = cpu_bfgs_update;
    //}}}
    // }}}
    
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Error in structure setup: %s\n",cudaGetErrorString(cudaError));
        exit(3);
    }
    // Copy constant structure to the device
    cudaMemcpyToSymbol(d_optparms,&h_optparms,sizeof(PAPER_data),0,cudaMemcpyHostToDevice);
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Error copying constant structure: %s\n",cudaGetErrorString(cudaError));
        exit(3);
    }

    // Shared mem required to store molecule coordinates
    #ifdef ORIG_GLOBAL
    const int mol_shmem = sizeof(float)*(  4*d_fitmols.maxatoms + 4*d_refMultimol.maxatoms);
    #else
    const int mol_shmem = sizeof(float)*(2*4*d_fitmols.maxatoms + 4*d_refMultimol.maxatoms);
    #endif

    //  9 + 4*fitcount + 7*nthreads floats of temp space
    const int init_threads = 64;
    const int ls_threads = 64;
    const int grad_threads = 64;
    const int obj_threads = 64;
    const int BFGS_init_shmem = mol_shmem + sizeof(float)*(9 + 4*d_fitmols.maxatoms + 7*init_threads);
    const int LS_shmem        = mol_shmem + sizeof(float)*(9 + ls_threads);
    const int obj_shmem       = mol_shmem + sizeof(float)*(9 + obj_threads);
    const int grad_shmem      = mol_shmem + sizeof(float)*(9 + 4*d_fitmols.maxatoms + 7*grad_threads);
    if (debug) printf("Shmem for LS/objective: %d; init/grad: %d\n",LS_shmem,grad_shmem);

    // Initialize BFGS
    init_dev_BFGS<<<d_fitmols.nmols,init_threads,BFGS_init_shmem>>>();
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Error initializing BFGS: %s\n",cudaGetErrorString(cudaError));
        exit(3);
    }

    float* hTransforms = NULL;
    float* hGradients = NULL;
    float* hOldGradients = NULL;
    float* hDirections = NULL;
    float* hinvH = NULL;
    float* hScores = NULL;
    float* hMaxstep = NULL;
    
    if (debug || cpu_bfgs_update) {
	cudaMallocHost((void**)&hTransforms,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float));
	cudaMallocHost((void**)&hGradients,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float));
	cudaMallocHost((void**)&hOldGradients,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float));
	cudaMallocHost((void**)&hDirections,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float));
	cudaMallocHost((void**)&hinvH,d_fitmols.nmols*h_optparms.hessian_pitch*sizeof(float));
	cudaMallocHost((void**)&hScores,d_fitmols.nmols*sizeof(float));
	cudaMallocHost((void**)&hMaxstep,d_fitmols.nmols*sizeof(float));
    }

    if (cpu_bfgs_update) {
        cudaMemcpy(hGradients,h_optparms.gradient,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(hinvH,h_optparms.invH,d_fitmols.nmols*h_optparms.hessian_pitch*sizeof(float),cudaMemcpyDeviceToHost);
    }

    if (enableTiming) {
        // Initialize timers
        cudaMemset(h_optparms.timers,0,h_optparms.timer_pitch*d_fitmols.nmols*sizeof(clock_t));
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Error in timing init: %s\n",cudaGetErrorString(cudaError));
            exit(3);
        }
    }

    int iter = 0;
    bool allcompleted;
    do {
        if (debug) { //{{{
            printf("Starting iteration %d\n",iter);
            cudaMemcpy(hTransforms,d_fitmols.transforms,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hGradients,h_optparms.gradient,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hDirections,h_optparms.search_direction,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hScores,h_optparms.score,d_fitmols.nmols*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hMaxstep,h_optparms.ls_maxstep,d_fitmols.nmols*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hinvH,h_optparms.invH,d_fitmols.nmols*h_optparms.hessian_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess) {
                printf("Error retrieving transforms: %s\n",cudaGetErrorString(cudaError));
                exit(3);
            }
            for (uint i = 0; i < d_fitmols.nmols; i++) {
                //printf("Molecule id #%d: optimal overlap value = %f\n",i,bestOverlaps[i]);
                float *xf = hTransforms+i*d_fitmols.transform_pitch;
                printf("\nTransform:[ %.4f %.4f %.4f; %.4f %.4f %.4f %.4f]\n",xf[0],xf[1],xf[2],xf[3],xf[4],xf[5],xf[6]);
                xf = hGradients+i*d_fitmols.transform_pitch;
                printf("Gradient:[ %.4f %.4f %.4f; %.4f %.4f %.4f %.4f]\n",xf[0],xf[1],xf[2],xf[3],xf[4],xf[5],xf[6]);
                xf = hDirections+i*d_fitmols.transform_pitch;
                printf("Direction:[ %.4f %.4f %.4f; %.4f %.4f %.4f %.4f]\n",xf[0],xf[1],xf[2],xf[3],xf[4],xf[5],xf[6]);
                printf("Score: %f, Maxstep: %f\n",hScores[i],hMaxstep[i]);
                float* matrix = transformToCompensatedMatrix(xf,com_ref,com_fit[0]);
                //printTransformMatrix(matrix,stdout);
                //printf("Hessian^-1:\n");
                //for (int j = 0; j < 7; j++) {
                //    xf = hinvH + i*h_optparms.hessian_pitch + j*7;
                //    printf("  [ %.4f %.4f %.4f; %.4f %.4f %.4f %.4f]\n",xf[0],xf[1],xf[2],xf[3],xf[4],xf[5],xf[6]);
                //}
                //printf("\n");

                free(matrix);
            }
            fprintf(stdout,"--------------------------------------\n");
        } //}}}
    
        // Do line search and check x cvg
        line_search_kernel<<<d_fitmols.nmols,ls_threads,LS_shmem>>>();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Error in line search kernel: %s\n",cudaGetErrorString(cudaError));
            exit(3);
        }

        if (debug) { //{{{
            printf("After line search:\n");
            cudaMemcpy(hTransforms,d_fitmols.transforms,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hGradients,h_optparms.gradient,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hDirections,h_optparms.search_direction,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hinvH,h_optparms.invH,d_fitmols.nmols*h_optparms.hessian_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hScores,h_optparms.score,d_fitmols.nmols*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hMaxstep,h_optparms.ls_maxstep,d_fitmols.nmols*sizeof(float),cudaMemcpyDeviceToHost);
            cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess) {
                printf("Error retrieving transforms: %s\n",cudaGetErrorString(cudaError));
                exit(3);
            }
            for (uint i = 0; i < d_fitmols.nmols; i++) {
                //printf("Molecule id #%d: optimal overlap value = %f\n",i,bestOverlaps[i]);
                float *xf = hTransforms+i*d_fitmols.transform_pitch;
                printf("\nTransform:[ %.4f %.4f %.4f; %.4f %.4f %.4f %.4f]\n",xf[0],xf[1],xf[2],xf[3],xf[4],xf[5],xf[6]);
                xf = hGradients+i*d_fitmols.transform_pitch;
                printf("Gradient:[ %.4f %.4f %.4f; %.4f %.4f %.4f %.4f]\n",xf[0],xf[1],xf[2],xf[3],xf[4],xf[5],xf[6]);
                xf = hDirections+i*d_fitmols.transform_pitch;
                printf("Direction:[ %.4f %.4f %.4f; %.4f %.4f %.4f %.4f]\n",xf[0],xf[1],xf[2],xf[3],xf[4],xf[5],xf[6]);
                printf("Score: %f, Maxstep: %f\n",hScores[i],hMaxstep[i]);
                float* matrix = transformToCompensatedMatrix(xf,com_ref,com_fit[0]);
                //printTransformMatrix(matrix,stdout);
                //printf("Hessian^-1:\n");
                //for (int j = 0; j < 7; j++) {
                //    xf = hinvH + i*h_optparms.hessian_pitch + j*7;
                //    printf("  [ %.4f %.4f %.4f; %.4f %.4f %.4f %.4f]\n",xf[0],xf[1],xf[2],xf[3],xf[4],xf[5],xf[6]);
                //}
                //printf("\n");

                free(matrix);
            }
            fprintf(stdout,"--------------------------------------\n");
        } //}}}

        // evaluate gradient, check convergence, and update direction for next line search
        gradient_kernel<<<d_fitmols.nmols,grad_threads,grad_shmem>>>();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Error in gradient kernel: %s\n",cudaGetErrorString(cudaError));
            exit(3);
        }

        // For benchmarking for GPU GEMS
        if (cpu_bfgs_update) {
            // Swap gradient buffers before copying new result
            float* t = hOldGradients;
            hOldGradients = hGradients;
            hGradients = t;

            cudaMemcpy(hTransforms,d_fitmols.transforms,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hGradients,h_optparms.gradient,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hDirections,h_optparms.search_direction,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            //cudaMemcpy(hinvH,h_optparms.invH,d_fitmols.nmols*h_optparms.hessian_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hScores,h_optparms.score,d_fitmols.nmols*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(completed,h_optparms.completed,d_fitmols.nmols*sizeof(int),cudaMemcpyDeviceToHost);
            // Check convergence; update invH and direction for next iteration
            for (int mol = 0; mol < d_fitmols.nmols; mol++) { // {{{
                const float* oldGradient = hOldGradients + mol*d_fitmols.transform_pitch;
                const float* gradient    = hGradients    + mol*d_fitmols.transform_pitch;
                const float* transform   = hTransforms   + mol*d_fitmols.transform_pitch;
                float* direction         = hDirections   + mol*d_fitmols.transform_pitch;
                float* invH              = hinvH         + mol*h_optparms.hessian_pitch;
                if (completed[mol]) continue;

                // Check convergence on gradient
                float test =  fabsf(gradient[0])*fmaxf(1.0f,fabsf(transform[0]))/fmaxf(1.0f,hScores[mol]);
                for (int i = 1; i < 7; i++) {
                    test = fmaxf(test,fabsf(gradient[i])*fmaxf(1.0f,fabsf(transform[i]))/fmaxf(1.0f,hScores[mol]));
                }
                if (test < h_optparms.bfgs_tol_gradient) {
                    completed[mol] = 2;
                    continue;
                }
                
                if (completed[mol]) continue;
                // Update search direction
                float dg[7];
                for (int i = 0; i < 7; i++) dg[i] = gradient[i] - oldGradient[i];
                float inv_alpha = 0;
                for (int i = 0; i < 7; i++) inv_alpha += dg[i]*direction[i];
                float magdg2 = 0;
                for (int i = 0; i < 7; i++) magdg2 += dg[i]*dg[i];
                float magdir2 = 0;
                for (int i = 0; i < 7; i++) magdir2 += direction[i]*direction[i];

                if (inv_alpha * inv_alpha > h_optparms.bfgs_epsilon*magdg2*magdir2) { //{{{
                    // Update inverse Hessian
                    float hdg[7];
                    sgemv7(hdg,invH,dg);
                    float inv_beta = 0;
                    for (int i = 0; i < 7; i++) inv_beta += dg[i]*hdg[i];
                    float beta = 1/inv_beta;
                    // ok, so inv_alpha is actually alpha now
                    inv_alpha = 1/inv_alpha;
            
                    // Reuse dg to store the BFGS update vector
                    for (int i = 0; i < 7; i++) dg[i] = inv_alpha*direction[i] - beta*hdg[i];

                    // Update the inverse Hessian with sum of scaled vector outer products
                    for (uint i = 0; i < 7; i++) {
                        for (int j = 0; j < 7; j++) {
                            invH[i*7+j] += inv_alpha*(direction[i]*direction[j]) - beta*(hdg[i]*hdg[j]) + inv_beta*(dg[i]*dg[j]);
                        }
                    }
                } // }}}

                // Set the search direction for the next iteration
                sgemv7(direction,invH,gradient);
                for (int i = 0; i < 7; i++) direction[i] = -direction[i];

            } // }}}
            cudaMemcpy(h_optparms.search_direction,hDirections,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float),cudaMemcpyHostToDevice);
            //cudaMemcpy(h_optparms.invH,hinvH,d_fitmols.nmols*h_optparms.hessian_pitch*sizeof(float),cudaMemcpyHostToDevice);
        }
        if (debug) { //{{{
            printf("After gradient kernel:\n");
            cudaMemcpy(hTransforms,d_fitmols.transforms,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hGradients,h_optparms.gradient,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hDirections,h_optparms.search_direction,d_fitmols.nmols*d_fitmols.transform_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hinvH,h_optparms.invH,d_fitmols.nmols*h_optparms.hessian_pitch*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hScores,h_optparms.score,d_fitmols.nmols*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hMaxstep,h_optparms.ls_maxstep,d_fitmols.nmols*sizeof(float),cudaMemcpyDeviceToHost);
            cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess) {
                printf("Error retrieving transforms: %s\n",cudaGetErrorString(cudaError));
                exit(3);
            }
            for (uint i = 0; i < d_fitmols.nmols; i++) {
                //printf("Molecule id #%d: optimal overlap value = %f\n",i,bestOverlaps[i]);
                float *xf = hTransforms+i*d_fitmols.transform_pitch;
                printf("\nTransform:[ %.4f %.4f %.4f; %.4f %.4f %.4f %.4f]\n",xf[0],xf[1],xf[2],xf[3],xf[4],xf[5],xf[6]);
                xf = hGradients+i*d_fitmols.transform_pitch;
                printf("Gradient:[ %.4f %.4f %.4f; %.4f %.4f %.4f %.4f]\n",xf[0],xf[1],xf[2],xf[3],xf[4],xf[5],xf[6]);
                xf = hDirections+i*d_fitmols.transform_pitch;
                printf("Direction:[ %.4f %.4f %.4f; %.4f %.4f %.4f %.4f]\n",xf[0],xf[1],xf[2],xf[3],xf[4],xf[5],xf[6]);
                printf("Score: %f, Maxstep: %f\n",hScores[i],hMaxstep[i]);
                float* matrix = transformToCompensatedMatrix(xf,com_ref,com_fit[0]);
                //printTransformMatrix(matrix,stdout);
                //printf("Hessian^-1:\n");
                //for (int j = 0; j < 7; j++) {
                //    xf = hinvH + i*h_optparms.hessian_pitch + j*7;
                //    printf("  [ %.4f %.4f %.4f; %.4f %.4f %.4f %.4f]\n",xf[0],xf[1],xf[2],xf[3],xf[4],xf[5],xf[6]);
                //}
                //printf("\n");

                free(matrix);
            }
            fprintf(stdout,"--------------------------------------\n");
        } //}}}

        // Copy completion flags back to host and check for completion of all molecules
        if (!cpu_bfgs_update) cudaMemcpy(completed,h_optparms.completed,d_fitmols.nmols*sizeof(int),cudaMemcpyDeviceToHost);
        allcompleted = true;
        int ccount[4] = {0,0,0,0};
        if (debug) printf("Completion flags in iteration %d:",iter);
        for (int i = 0; i < d_fitmols.nmols; i++) {
            if (debug) printf(" %d",completed[i]);
            ccount[completed[i]]++;
            allcompleted = allcompleted && completed[i];
            if (!debug && !tuneParms) if (!allcompleted) break;
        }
        if (debug) printf("\n");
        if (tuneParms) printf("Completion status counts at iter %d: 0: %d 1: %d 2: %d 3: %d\n",iter,ccount[0],ccount[1],ccount[2],ccount[3]);
        iter++;
    } while (iter < h_optparms.bfgs_max_iters && !allcompleted);

    if (enableTiming) {
        // Reduce timer counts over blocks
        clock_t *t = new clock_t[h_optparms.timer_pitch*d_fitmols.nmols];
        cudaMemcpy(t,h_optparms.timers,h_optparms.timer_pitch*d_fitmols.nmols*sizeof(clock_t),cudaMemcpyDeviceToHost);
        memset(h_timers,0,numTimers*sizeof(clock_t));
        for (int i = 0; i < d_fitmols.nmols; i++) {
            clock_t* row = t+i*h_optparms.timer_pitch;
            bool anyNeg = false;
            for (int j = 0; j < numTimers; anyNeg = (row[j++] < 0) || anyNeg);
            for (int j = 0; !anyNeg && j < numTimers; j++) {
                h_timers[j] += (uint64_t) row[j];
                if (row[j] < 0) printf("Warning timer %d count < 0: %lld\n",j,(long long)row[j]);
            }
        }
        delete[] t;

    }

    // Final overlap cleanup - normalize transforms and calculate overlap
    final_objective_kernel<<<d_fitmols.nmols,obj_threads,obj_shmem>>>();
    cudaMemcpy(finalScores,h_optparms.score,d_fitmols.nmols*sizeof(float),cudaMemcpyDeviceToHost);

    // Clean up allocations
    cudaFree(h_optparms.gradient);
    cudaFree(h_optparms.search_direction);
    cudaFree(h_optparms.score);
    cudaFree(h_optparms.completed);
    //delete[] completed;
    cudaFreeHost(completed);
    if (enableTiming) {
        cudaFree(h_optparms.timers);
    }
    cudaFree(h_optparms.ls_maxstep);
    cudaFree(h_optparms.invH);

    if (debug || cpu_bfgs_update) {
        cudaFreeHost(hTransforms);
        cudaFreeHost(hGradients);
        cudaFreeHost(hOldGradients);
        cudaFreeHost(hDirections);
        cudaFreeHost(hinvH);
        cudaFreeHost(hScores);
        cudaFreeHost(hMaxstep);
    }

    return iter;

} //}}}
