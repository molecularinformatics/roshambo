/* 
 * cpuPAPER/nrOptimization.h
 * Prototypes for modified optimization routines from Numerical
 * Recipes in C, 2/e.
 * Routines modified to take dimensionality as template parameter.
 *
 * Source for routines is not included due to license restrictions;
 * these headers are as a reference for assistance in porting.
 *
 * Current: calculates cyclic decomposition of molecules.
 *
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */

#ifndef _NR_OPTIMIZATION_H_
#define _NR_OPTIMIZATION_H_
#include <math.h>
#include <stdio.h>
#include <string.h>

#define MOV3(a,b,c, d,e,f) (a)=(d); (b)=(e); (c)=(f);

/*
 * Optimization and other numerical methods from Numerical Recipes in C, 2nd ed.
 *
 */

/*
 * Given a function f and its derivative df and a bracketing triplet of abscissas ax,bx,cx,
 * this routine isolates the minimum to a fractional precision of about tol using a modification
 * of Brent's method that uses derivatives. The abscissa is returned as xmin and the minimum
 * function value is returned as the function's return value.
 *
 * tunable parameters: ITMAX = max iteration count
 *                     ZEPS
 *
 */
//float dbrent(float x,float bx,float cx, float (*f)(float),float (*df)(float),float tol,float& xmin,const int ITMAX=100,const float ZEPS=1.0e-10f);

template<int ndim> void amoeba(float p[][ndim],float y[],float ftol,float (*funk)(float*),int& nfunk,const int NMAX=1000);
template<int ndim> float amotry(float p[][ndim],float y[],float* psum,float (*funk)(float*),int ihi,float fac);



// Routines for conjugate gradients optimization
void mnbrak(float& ax, float& bx,float& cx, float& fa,float& fb,float& fc,float (*func)(float),const float GOLD=1.618034f ,const float GLIMIT=100.0,const float TINY=1.0e-20);
//void mnbrak(float& ax, float& bx,float& cx, float& fa,float& fb,float& fc,float (*func)(float),const float GOLD=1.618034f ,const float GLIMIT=2.0,const float TINY=1.0e-20);
float dbrent(float ax,float bx,float cx, float (*f)(float),float (*df)(float),float tol,float& xmin,const int ITMAX=100,const float ZEPS=1.0e-10f);
template<int n> void dlinmin(float* p,float* xi,float& fret,float(*func)(float*), void(*dfunc)(float*, float*),const float TOL=2.0e-4);
template<int n> float df1dim(float x);
template<int n> float f1dim(float x);
template<int n> void conjGrad(float* p,float ftol,int& iter,float& fret,float (*func)(float*),void (*dfunc)(float*, float*),const int ITMAX=200,const float EPS=1.0e-10);


template<int n> void lnsrch(float xold[], float fold, float g[], float p[], float x[],float& f, float stpmax, int& check, float (*func)(float []),const float TOLX=1.0e-7,const float ALF=1.0e-4);
template<int n> void dfpmin(float p[], float gtol, int& iter, float& fret, float(*func)(float []), void (*dfunc)(float [], float []),const float STPMX=100.0,const int ITMAX=200,const float EPS=3.0e-8,const float TOLX=12.0e-8);

// Inverts nxn matrix a by Gauss-Jordan elimination
template<int n> bool gaussj(float** a);

#endif
