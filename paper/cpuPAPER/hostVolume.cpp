/*
 * cpuPAPER/hostVolume.cpp
 * Implements Gaussian and hard-sphere volume calculations on the CPU.
 * Gradients unsupported for hard-sphere.
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */


#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <iostream>
using namespace std;
#include "hostVolume.h"
#include "cudaVolumeTypes.h"

//#define HARDSPHERE_VOLUME
#ifndef GRID_VOLUME
    #define ANALYTIC_GRADIENT
    #define ANALYTIC_VOLUME
#endif

#ifndef HARDSPHERE_VOLUME
#define GAUSSIAN_VOLUME
#else
#undef GAUSSIAN_VOLUME
#endif

inline float dot3(float3 a,float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
inline float3 operator-(const float3& a,const float3& b) {
	return make_float3(a.x-b.x,a.y-b.y,a.z-b.z);
}
inline float distsqr(const float3& a,const float4& b) {
    float tmpx = b.x - a.x;
    float tmpy = b.y - a.y;
    float tmpz = b.z - a.z;
    return tmpx*tmpx + tmpy*tmpy + tmpz*tmpz;
}
inline float distsqr(const float4& a,const float4& b) {
    float tmpx = b.x - a.x;
    float tmpy = b.y - a.y;
    float tmpz = b.z - a.z;
    return tmpx*tmpx + tmpy*tmpy + tmpz*tmpz;
}
inline float min(float a,float b) {
	return (a<b)?a:b;
}
inline float max(float a,float b) {
	return (a>b)?a:b;
}
void boundingBox(CUDAmol& mol,float margin,float3& lb,float3& ub) {
    float4& atom1 = mol.atoms[0];
    float rad = atom1.w;
    lb.x = atom1.x-rad; lb.y = atom1.y-rad; lb.z = atom1.z-rad;
    ub.x = atom1.x+rad; ub.y = atom1.y+rad; ub.z = atom1.z+rad;
    for (unsigned int i = 1; i< mol.natoms; i++) {
        float4& atomi = mol.atoms[i];
        rad = atomi.w;
        lb.x = min(lb.x,atomi.x-rad);
        lb.y = min(lb.y,atomi.y-rad);
        lb.z = min(lb.z,atomi.z-rad);
        ub.x = max(ub.x,atomi.x+rad);
        ub.y = max(ub.y,atomi.y+rad);
        ub.z = max(ub.z,atomi.z+rad);
    }
    lb.x -= margin; lb.y -= margin; lb.z -= margin;
    ub.x += margin; ub.y += margin; ub.z += margin;
    return;
}
inline float3 gridSpaceToRealSpace(float res,const float3& lb,uint x,uint y,uint z) {
    float3 rs;
    rs.x = lb.x + (x+0.5)*res;
    rs.y = lb.y + (y+0.5)*res;
    rs.z = lb.z + (z+0.5)*res;
    return rs;
}

inline float rho(float4 atom,float3 gc) {
	const float rt22 = 2.82842712475f;
	const float partialalpha = -2.41798793102f;
        float alpha = partialalpha/(atom.w * atom.w);
        float r2 = distsqr(gc,atom);
        return rt22*expf(alpha*r2);
}
	
#ifdef GAUSSIAN_VOLUME
//#if 0
float getVolume(const CUDAmol& ref,Grid& grid) {
	float volume = 0.0f;
	for (unsigned int x = 0; x < grid.extent.x; x++) {
	for (unsigned int y = 0; y < grid.extent.y; y++) {
	for (unsigned int z = 0; z < grid.extent.z; z++) {
		float gridval = 1.0f;
		float3 gc = gridSpaceToRealSpace(grid.res,grid.lb,x,y,z);
		for (unsigned int i = 0; i < ref.natoms; i++) {
			gridval *= (1-rho(ref.atoms[i],gc));
		}
		volume += (1-gridval);
	}
	}
	}
	return volume*grid.res*grid.res*grid.res;
			
}
#ifdef ANALYTIC_VOLUME
// Calculates second-order approximate analytic overlap volume
float getOverlapVolume(const CUDAmol& ref,const CUDAmol& fit,Grid& grid) {
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
			overlap += Vij;
		}
	}
	return overlap;
}
float getOverlapVolumeVerbose(const CUDAmol& ref,const CUDAmol& fit,Grid& grid) {
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
            printf("Vij from ref %d (w=%f), fit %d (w=%f) = %f\n",xi,i.w,xj,j.w,Vij);
			overlap += Vij;
		}
	}
	return overlap;
}
#else
float getOverlapVolume(const CUDAmol& ref,const CUDAmol& fit,Grid& grid) {
	float volume = 0.0f;
    #ifdef OPENMP
    #pragma omp parallel for reduction(+:volume)
    #endif
	for (int x = 0; x < grid.extent.x; x++) {
	for (int y = 0; y < grid.extent.y; y++) {
	for (int z = 0; z < grid.extent.z; z++) {
		float refgridval = 1.0f;
		float fitgridval = 1.0f;
		float3 gc = gridSpaceToRealSpace(grid.res,grid.lb,x,y,z);
		for (int i = 0; i < ref.natoms; i++) {
			refgridval *= (1-rho(ref.atoms[i],gc));
		}
		refgridval = 1-refgridval;
		for (int i = 0; i < fit.natoms; i++) {
			fitgridval *= (1-rho(fit.atoms[i],gc));
		}
		fitgridval = 1 - fitgridval;
		volume += refgridval*fitgridval;
	}
	}
	}
	return volume*grid.res*grid.res*grid.res;
			
}
#endif
#endif
#ifdef HARDSPHERE_VOLUME
//#ifdef GAUSSIAN_VOLUME
float getVolume(const CUDAmol& ref,Grid& grid) {
	int volume = 0;
	for (int x = 0; x < grid.extent.x; x++) {
	for (int y = 0; y < grid.extent.y; y++) {
	for (int z = 0; z < grid.extent.z; z++) {
		bool gridval = false;
		float3 gc = gridSpaceToRealSpace(grid.res,grid.lb,x,y,z);
		for (int i = 0; i < ref.natoms; i++) {
			gridval = gridval || (distsqr(gc,ref.atoms[i]) < ref.atoms[i].w*ref.atoms[i].w);
		}
		if (gridval)
			volume++;
	}
	}
	}
	return volume*grid.res*grid.res*grid.res;
			
}
float getOverlapVolume(const CUDAmol& ref,const CUDAmol& fit,Grid& grid) {
	int volume = 0;
	for (int x = 0; x < grid.extent.x; x++) {
	for (int y = 0; y < grid.extent.y; y++) {
	for (int z = 0; z < grid.extent.z; z++) {
		bool refgridval = false;
		vool fitgridval = false;
		float3 gc = gridSpaceToRealSpace(grid.res,grid.lb,x,y,z);
		for (int i = 0; i < ref.natoms; i++) {
			refgridval = refgridval || (distsqr(gc,ref.atoms[i]) < ref.atoms[i].w*ref.atoms[i].w);
		}
		for (int i = 0; i < fit.natoms; i++) {
			fitgridval = fitgridval || (distsqr(gc,fit.atoms[i]) < fit.atoms[i].w*fit.atoms[i].w);
		}
		if (refgridval && fitgridval)
			volume++;
	}
	}
	}
	return volume*grid.res*grid.res*grid.res;
			
}

#endif

void getQuatGradients(const CUDAmol& fit,float* transform,float3* qDers,float3* rDers,float3* sDers,float3* uDers) {
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
#ifdef GAUSSIAN_VOLUME
#ifdef ANALYTIC_GRADIENT
void getGradient(const CUDAmol& ref,const CUDAmol& fit,Grid& grid,float* transform,float* gradient) {
	float3 *qDers = new float3[fit.natoms];
	float3 *rDers = new float3[fit.natoms];
	float3 *sDers = new float3[fit.natoms];
	float3 *uDers = new float3[fit.natoms];
	const float partialalpha = 2.41798793102f;
	const float pi = 3.14159265358f;
    getQuatGradients(fit,transform,qDers,rDers,sDers,uDers);
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
    
	float magp2 = transform[3]*transform[3]+transform[4]*transform[4]+transform[5]*transform[5]+transform[6]*transform[6];
    /* Remove radial component of rotation gradient */
    /*float proj = gradient[3]*transform[3]+gradient[4]*transform[4]+gradient[5]*transform[5]+gradient[6]*transform[6];
    gradient[3] -= (proj/magp2)*transform[3];
    gradient[4] -= (proj/magp2)*transform[4];
    gradient[5] -= (proj/magp2)*transform[5];
    gradient[6] -= (proj/magp2)*transform[6];*/
    
    float invmagp = 1/sqrtf(magp2);
	const float gamma = PENALTY_GAMMA;
	for (int i = 3; i < 7; i++)
		gradient[i] += gamma*(1-invmagp)*transform[i];

    delete[] qDers;
    delete[] rDers;
    delete[] sDers;
    delete[] uDers;
    
    return;
}

#else
void getGradient(const CUDAmol& ref,const CUDAmol& fit,Grid& grid,float* transform,float* gradient) {
	float xder,yder,zder,qder,rder,sder,uder;
	xder=yder=zder=qder=rder=sder=uder=0.0f;
	float3 *qDers = new float3[fit.natoms];
	float3 *rDers = new float3[fit.natoms];
	float3 *sDers = new float3[fit.natoms];
	float3 *uDers = new float3[fit.natoms];
	const float partialalpha = -2.41798793102f;
	getQuatGradients(fit,transform,qDers,rDers,sDers,uDers);
	
	for (int x = 0; x < grid.extent.x; x++) {
	for (int y = 0; y < grid.extent.y; y++) {
	for (int z = 0; z < grid.extent.z; z++) {
		float rhoref = 1.0f;
		float rhofit = 1.0f;
		float3 gc = gridSpaceToRealSpace(grid.res,grid.lb,x,y,z);
		for (int i = 0; i < ref.natoms; i++) {
			rhoref *= (1-rho(ref.atoms[i],gc));
		}
		float rhobarref = 1-rhoref;
		for (int i = 0; i < fit.natoms; i++) {
			rhofit *= (1-rho(fit.atoms[i],gc));
		}
		
		//float Gr = rhoref*rhofit;
		for (int k = 0; k< fit.natoms; k++) {
			float4 atom = fit.atoms[k];
        	float alpha = partialalpha/(atom.w * atom.w);
			float rhok = rho(atom,gc);
			//float scalar = Gr*2*alpha*rhok/(1-rhok);
            // TODO: DANGER!!!! this is negative alpha!
			float scalar = rhobarref*rhofit*2*alpha*rhok/(1-rhok);
			float3 delta = gc - make_float3(atom.x,atom.y,atom.z);
			/*xder += (rhobarref - scalar*delta.x);
			yder += (rhobarref - scalar*delta.y);
			zder += (rhobarref - scalar*delta.z);
			qder += (rhobarref - scalar*dot3(delta,qDers[k]));
			rder += (rhobarref - scalar*dot3(delta,rDers[k]));
			sder += (rhobarref - scalar*dot3(delta,sDers[k]));
			uder += (rhobarref - scalar*dot3(delta,uDers[k]));
			*/
			xder += scalar*delta.x;
			yder += scalar*delta.y;
			zder += scalar*delta.z;
			qder += scalar*dot3(delta,qDers[k]);
			rder += scalar*dot3(delta,rDers[k]);
			sder += scalar*dot3(delta,sDers[k]);
			uder += scalar*dot3(delta,uDers[k]);
		}

	}
	}
	}
	// TODO: is there really a sign error?
	gradient[0] = grid.res*grid.res*grid.res*xder;	
	gradient[1] = grid.res*grid.res*grid.res*yder;	
	gradient[2] = grid.res*grid.res*grid.res*zder;	
	gradient[3] = grid.res*grid.res*grid.res*qder;	
	gradient[4] = grid.res*grid.res*grid.res*rder;	
	gradient[5] = grid.res*grid.res*grid.res*sder;	
	gradient[6] = grid.res*grid.res*grid.res*uder;

	float invmagp = 1/sqrtf(transform[3]*transform[3]+transform[4]*transform[4]+transform[5]*transform[5]+transform[6]*transform[6]);
	const float gamma = PENALTY_GAMMA;
	for (int i = 3; i < 7; i++)
		gradient[i] += gamma*(1-invmagp)*transform[i];
	
		
	delete[] qDers;
	delete[] rDers;
	delete[] sDers;
	delete[] uDers;
	
	return;
}
#endif
#endif

#ifdef GAUSSIAN_VOLUME
float getObjectiveAndGradient(const CUDAmol& ref,const CUDAmol& fit,Grid& grid,float* transform,float* gradient) {
	getGradient(ref,fit,grid,transform,gradient);
	float volume = getOverlapVolume(ref,fit,grid);
	float magp = sqrtf(transform[3]*transform[3]+transform[4]*transform[4]+transform[5]*transform[5]+transform[6]*transform[6]);
	const float gamma = PENALTY_GAMMA;
	float penalty = 0.5*gamma*(magp-1)*(magp-1);
	cout << ", penalty = "<<penalty;
	volume -= penalty;
	return volume;
}
#endif

Grid getHostGridFromBox(float3 lb,float3 ub,float res) {
	Grid retval;
	retval.lb=lb;
	uint3 extent;
    float3 dims = ub-lb;
    //printf("Box dimensions: %fx%fx%f\n",dims.x,dims.y,dims.z);
    extent.x = (uint)(ceil(dims.x/res));
    extent.y = (uint)(ceil(dims.y/res));
    extent.z = (uint)(ceil(dims.z/res));
    retval.res=res;
    retval.extent=extent;
    //printf("Box extent: %ux%ux%u\n",retval.extent.x,retval.extent.y,retval.extent.z);
    retval.points=new float[extent.x*extent.y*extent.z];
	return retval;
}

Grid getHostGrid(CUDAmol& mol,float res,float margin) {
    // Calculate bounding box
    float3 ub,lb;
    boundingBox(mol,margin,lb,ub);
	return getHostGridFromBox(lb,ub,res);
    //printf("Got bounding box: [%f,%f,%f] - [%f,%f,%f]\n",retval.lb.x,retval.lb.y,retval.lb.z,ub.x,ub.y,ub.z);
}

void transformCUDAmolToHost(CUDAmol& cmol,CUDAmol& newmol,float* transform,uint startatom) {
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
    }
    newmol.natoms = natoms;
    newmol.atoms = newatoms;
    //printf("Inside transformToHost:\n-------------------\n");
    //dumpCUDAmol(newmol);
    return;
}
