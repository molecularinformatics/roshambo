/*
 * hostAnalyticVolume.cu
 * Implements Gaussian volume calculations on the CPU
 *  
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */

#include <math.h>
#include <stdio.h>
#include "hostAnalyticVolume.h"
#include "cudaVolumeTypes.h"


inline float dot3(float3 a,float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
inline float3 operator-(const float3& a,const float3& b) {
	return make_float3(a.x-b.x,a.y-b.y,a.z-b.z);
}
inline float distsqr(const float4& a,const float4& b) {
    float tmpx = b.x - a.x;
    float tmpy = b.y - a.y;
    float tmpz = b.z - a.z;
    return tmpx*tmpx + tmpy*tmpy + tmpz*tmpz;
}
inline float distsqr(const float x,const float y,const float z,const float4& b) {
    float tmpx = b.x - x;
    float tmpy = b.y - y;
    float tmpz = b.z - z;
    return tmpx*tmpx + tmpy*tmpy + tmpz*tmpz;
}


float hostOverlapVolume(const CUDAmol& ref,const CUDAmol& fit) {
    return hostOverlapVolumeAnalytic2(ref,fit);
}

// Calculates second-order approximate analytic overlap volume
float hostOverlapVolumeAnalytic2(const CUDAmol& ref,const CUDAmol& fit) {
	const float partialalpha = 2.41798793102f;
	const float pi = 3.14159265358f;
	float overlap = 0.0f;
    for (unsigned int xi = 0; xi < ref.natoms; xi++) {
        float4& i = ref.atoms[xi];
        float alphai = partialalpha/(i.w*i.w);
        for (unsigned int xj = 0; xj < fit.natoms; xj++) {
            float4& j = fit.atoms[xj];
            float Rij2 = distsqr(i,j);

            //if (Rij2 > 16) continue;

            float alphaj = partialalpha/(j.w*j.w);
            float Kij = expf(-(alphai*alphaj*Rij2)/(alphai+alphaj));
            float Vij = 8*Kij*powf((pi/(alphai+alphaj)),1.5);
            //printf("Vij from ref %d (w=%f), fit %d (w=%f): %f\n",xi,i.w,xj,j.w,Vij);
			overlap += Vij;
		}
	}
	return overlap;
}

// Calculates overlap volume on a 1/3 angstrom grid
float hostOverlapVolumeHS(const CUDAmol& ref,const CUDAmol& fit) {
    // Calculate bounding box
    float xmin,xmax,ymin,ymax,zmin,zmax;
    xmin= ref.atoms[0].x-ref.atoms[0].w;
    ymin= ref.atoms[0].y-ref.atoms[0].w;
    zmin= ref.atoms[0].z-ref.atoms[0].w;
    xmax= ref.atoms[0].x+ref.atoms[0].w;
    ymax= ref.atoms[0].y+ref.atoms[0].w;
    zmax= ref.atoms[0].z+ref.atoms[0].w;

    for (uint i = 1; i < ref.natoms; i++) {
        xmin = min(xmin,ref.atoms[i].x-ref.atoms[i].w);
        ymin = min(ymin,ref.atoms[i].y-ref.atoms[i].w);
        zmin = min(zmin,ref.atoms[i].z-ref.atoms[i].w);
        xmax = max(xmax,ref.atoms[i].x+ref.atoms[i].w);
        ymax = max(ymax,ref.atoms[i].y+ref.atoms[i].w);
        zmax = max(zmax,ref.atoms[i].z+ref.atoms[i].w);
    }
    for (uint i = 0; i < fit.natoms; i++) {
        xmin = min(xmin,fit.atoms[i].x-fit.atoms[i].w);
        ymin = min(ymin,fit.atoms[i].y-fit.atoms[i].w);
        zmin = min(zmin,fit.atoms[i].z-fit.atoms[i].w);
        xmax = max(xmax,fit.atoms[i].x+fit.atoms[i].w);
        ymax = max(ymax,fit.atoms[i].y+fit.atoms[i].w);
        zmax = max(zmax,fit.atoms[i].z+fit.atoms[i].w);
    }

    const float res = 0.33f;
    uint count = 0;
    for (float xcor = xmin; xcor < xmax; xcor += res) {
    for (float ycor = ymin; ycor < ymax; ycor += res) {
    for (float zcor = zmin; zcor < zmax; zcor += res) {
        bool refAtomFound = false;
        for (uint i = 0; i < ref.natoms && !refAtomFound; i++)
            refAtomFound = (ref.atoms[i].w * ref.atoms[i].w >= distsqr(xcor,ycor,zcor,ref.atoms[i]));
        if (!refAtomFound) continue;
        bool fitAtomFound = false;
        for (uint i = 0; i < fit.natoms && !fitAtomFound; i++)
            fitAtomFound = (fit.atoms[i].w * fit.atoms[i].w >= distsqr(xcor,ycor,zcor,fit.atoms[i]));
        // if we're here, there is a ref atom in range, so just check if there's a fit atom too
        if (fitAtomFound) count++;
    }}}

    return count*res*res*res;
    
}

void hostQuatGradients(const CUDAmol& fit,float* transform,float3* qDers,float3* rDers,float3* sDers,float3* uDers) {
    float q=transform[3];
    float r=transform[4];
    float s=transform[5];
    float u=transform[6];
    float invmag2 = 1.0f/(q*q+r*r+s*s+u*u);
    for (unsigned int i=0;i<fit.natoms;i++) {
        float xk=fit.atoms[i].x;
        float yk=fit.atoms[i].y;
        float zk=fit.atoms[i].z;
        float dxdq = invmag2*2.0f*( q*xk + u*yk - s*zk);
        float dxdr = invmag2*2.0f*( r*xk + s*yk + u*zk);
        float dydr = invmag2*2.0f*( s*xk - r*yk + q*zk);
        float dxdu = invmag2*2.0f*(-u*xk + q*yk + r*zk);
        float dzds = dxdq;
        float dydu = -dxdq;
        float dyds = dxdr;
        float dzdu = dxdr;
        float dxds = -dydr;
        float dzdq = dydr;
        float dydq = dxdu;
        float dzdr = -dxdu;
        qDers[i] = make_float3(dxdq,dydq,dzdq);   
        rDers[i] = make_float3(dxdr,dydr,dzdr);   
        sDers[i] = make_float3(dxds,dyds,dzds);   
        uDers[i] = make_float3(dxdu,dydu,dzdu);   
    }
    
    return;
}

void hostGradient(const CUDAmol& ref,const CUDAmol& fit,float* transform,float* gradient) {
	float3 *qDers = new float3[fit.natoms];
	float3 *rDers = new float3[fit.natoms];
	float3 *sDers = new float3[fit.natoms];
	float3 *uDers = new float3[fit.natoms];
	const float partialalpha = 2.41798793102f;
	const float pi = 3.14159265358f;
    hostQuatGradients(fit,transform,qDers,rDers,sDers,uDers);
    memset(gradient,0,28);
    for (unsigned int xi = 0; xi < ref.natoms; xi++) {
        float4& i = ref.atoms[xi];
        float alphai = partialalpha/(i.w*i.w);
        for (unsigned int xj = 0; xj < fit.natoms; xj++) {
            float4& j = fit.atoms[xj];
            float Rij2 = distsqr(i,j);

            //if (Rij2 > 16) continue;
            
            float alphaj = partialalpha/(j.w*j.w);

            float Kij = expf(-(alphai*alphaj*Rij2)/(alphai+alphaj));
            float Vij = 8*Kij*powf((pi/(alphai+alphaj)),1.5);
            float scalar = Vij*-2*alphai*alphaj/(alphai+alphaj);
            //cout << "alphai="<<alphai<<", alphaj="<<alphaj<<", Rij2="<<Rij2<<", Kij="<<Kij<<", Vij = "<<Vij<<endl;
            float3 delta;
            delta.x = i.x - j.x;
            delta.y = i.y - j.y;
            delta.z = i.z - j.z;
            //cout << "Delta = "<<delta.x<<","<<delta.y<<","<<delta.z<<"; scalar = "<<scalar<<endl;

            gradient[0] += scalar*delta.x;
            gradient[1] += scalar*delta.y;
            gradient[2] += scalar*delta.z;
			gradient[3] += scalar*dot3(delta,qDers[xj]);
			gradient[4] += scalar*dot3(delta,rDers[xj]);
			gradient[5] += scalar*dot3(delta,sDers[xj]);
			gradient[6] += scalar*dot3(delta,uDers[xj]);
        }
    }
    
	//float magp2 = transform[3]*transform[3]+transform[4]*transform[4]+transform[5]*transform[5]+transform[6]*transform[6];
    /* Remove radial component of rotation gradient */
    /*float proj = gradient[3]*transform[3]+gradient[4]*transform[4]+gradient[5]*transform[5]+gradient[6]*transform[6];
    gradient[3] -= (proj/magp2)*transform[3];
    gradient[4] -= (proj/magp2)*transform[4];
    gradient[5] -= (proj/magp2)*transform[5];
    gradient[6] -= (proj/magp2)*transform[6];*/
    
    /*float invmagp = 1/sqrtf(magp2);
	const float gamma = PENALTY_GAMMA;
	for (int i = 3; i < 7; i++)
		gradient[i] += gamma*(1-invmagp)*transform[i];
    */

    delete[] qDers;
    delete[] rDers;
    delete[] sDers;
    delete[] uDers;
    
    return;
}

void transformCUDAmolToHost(CUDAmol& cmol,CUDAmol& newmol,float* transform) {
    const uint startatom = 0;
    /*printf("\n********************************\n");
    printf("* Entering transformToHost     *\n");
    printf("********************************\n");
    printf("Starting with:\n----------------------\n");
    dumpCUDAmol(cmol);
    printf("\n");*/
    uint natoms = cmol.natoms - startatom;
    float4* newatoms = new float4[natoms];
    // Precalculate rotation matrix elements
    //                            3 4 5 6
    // Elements of quaternion are x,e,n,s
    float e2 = transform[4]*transform[4];
    float n2 = transform[5]*transform[5];
    float s2 = transform[6]*transform[6];
    float x2 = transform[3]*transform[3];
    float xe = transform[3]*transform[4];
    float ne = transform[5]*transform[4];
    float xs = transform[3]*transform[6];
    float es = transform[4]*transform[6];
    float nx = transform[3]*transform[5];
    float ns = transform[5]*transform[6];

    float a[3][3] = { e2-n2-s2+x2,  2*(ne+xs),    2*(es-nx),
                      2*(ne-xs),    -e2+n2-s2+x2, 2*(ns+xe),
                      2*(es+nx),    2*(ns-xe),    -e2-n2+s2+x2};


    // Get normalization factor for the quaternion part of the transform
    float invmag = 1.0f/(x2+e2+n2+s2);

    //printf("Rotation matrix in host:\n");
    //printf("[[ %f %f %f\n   %f %f %f\n   %f %f %f ]]\n",a[0][0]*invmag,a[0][1]*invmag,a[0][2]*invmag,a[1][0]*invmag,a[1][1]*invmag,a[1][2]*invmag,a[2][0]*invmag,a[2][1]*invmag,a[2][2]*invmag);
    
    // Normalize quaternion, just for kicks
    /*float normfac = sqrtf(x2+e2+n2+s2);
    for (int i = 3; i < 7; i++)
        transform[i]/=normfac;
    */
    for (unsigned int i = 0; i < natoms; i++) {
        float4 newatom;
        float4& oldatom = cmol.atoms[startatom+i];
        //printf("Rotating old atom (%d,%d) at [%f,%f,%f] to",i,startatom+i,oldatom.x,oldatom.y,oldatom.z);
        // Rotate
        newatom.x = oldatom.x*a[0][0] + oldatom.y*a[0][1] + oldatom.z*a[0][2];
        newatom.y = oldatom.x*a[1][0] + oldatom.y*a[1][1] + oldatom.z*a[1][2];
        newatom.z = oldatom.x*a[2][0] + oldatom.y*a[2][1] + oldatom.z*a[2][2];
        // Normalize for quaternion length
        newatom.x *= invmag;
        newatom.y *= invmag;
        newatom.z *= invmag;
        //printf(" [%f,%f,%f]; translating to",newatom.x,newatom.y,newatom.z);

        // Translate
        newatom.x += transform[0];
        newatom.y += transform[1];
        newatom.z += transform[2];
        //printf(" [%f,%f,%f]\n",newatom.x,newatom.y,newatom.z);
        
        // Copy radii
        newatom.w = oldatom.w;
        // Assign
        newatoms[i] = newatom;
        //printf("Atom %d moved from [%f,%f,%f] to [%f,%f,%f]\n",i,oldatom.x,oldatom.y,oldatom.z,newatom.x,newatom.y,newatom.z);
    }
    //printf("\n");
    newmol.natoms = natoms;
    newmol.atoms = newatoms;
    //printf("Inside transformToHost:\n-------------------\n");
    //dumpCUDAmol(newmol);
    return;
}
