/*
 * transformTools.cu
 * Utility functions to handle 4x4 matrix/quaternion transformations
 *  
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the GPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */
#include "cudaVolumeTypes.h"
#include "transformTools.h"

float* transformToCompensatedMatrix(float* transform,float3 com_ref,float3 com_fit) {
    float* matrix = (float*)malloc(16*sizeof(float));
	
    affineQuatToMatrix(transform,matrix);

    float translation[16];
    memset(translation,0,16*sizeof(float));
    translation[0*4+0] = 1;
    translation[1*4+1] = 1;
    translation[2*4+2] = 1;
    translation[3*4+3] = 1;
    // Compose the transforms as follows:
    // <----------
    // move to ref_com * optimized transform * move to fit_com
    
    // Set translation to -com_fit
    translation[0*4+3] = -com_fit.x;
    translation[1*4+3] = -com_fit.y;
    translation[2*4+3] = -com_fit.z;
    
    composeTransform(matrix,translation);
    
    // Set translation to com_ref
    translation[0*4+3] = com_ref.x;
    translation[1*4+3] = com_ref.y;
    translation[2*4+3] = com_ref.z;
    
    composeTransform(translation,matrix);
    
    memcpy(matrix,translation,16*sizeof(float));

    return matrix;
}

// Performs a=a*b for 4x4 matrices
void composeTransform(float* a,const float* b) { //{{{
    float m[4][4];
    for (int i = 0; i < 4; i++) {
        for ( int j = 0; j < 4; j++) {
            m[i][j] = 0;
            for ( int k = 0; k < 4; k++) {
                m[i][j] += a[i*4+k]*b[k*4+j];
            }
        }
    }
    memcpy(a,m,16*sizeof(float));
    return;
} //}}}

void affineQuatToMatrix(const float* quat,float* matrix) { //{{{
	// Elements of quaternion are x,e,n,s
    float e2 = quat[4]*quat[4];
    float n2 = quat[5]*quat[5];
    float s2 = quat[6]*quat[6];
    float x2 = quat[3]*quat[3];
	float invmagsqr = 1.0f/(x2+e2+n2+s2);
    float xe = quat[3]*quat[4];
    float ne = quat[5]*quat[4];
    float xs = quat[3]*quat[6];
    float es = quat[4]*quat[6];
    float nx = quat[3]*quat[5];
    float ns = quat[5]*quat[6];

    float scaledrot[3][3] = { e2-n2-s2+x2,  2*(ne+xs),    2*(es-nx),
                      2*(ne-xs),    -e2+n2-s2+x2, 2*(ns+xe),
                      2*(es+nx),    2*(ns-xe),    -e2-n2+s2+x2};
	
	// Copy scaled rotation matrix to transformation matrix, scaling for quat length
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			matrix[i*4 + j] = scaledrot[i][j]*invmagsqr;
		}
	}
    #ifdef DEBUG_OUTPUT
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            //if (i == j) continue;
            float dot = 0;
            for (int k = 0 ; k < 3; k++) {
                //printf("\t component: %f * %f=%f\n",matrix[k*4+j],matrix[k*4+i],matrix[k*4+j]*matrix[k*4+i]);

                dot += matrix[k*4+j]*matrix[k*4+i];
            }
            printf("Dot product of rows %d, %d = %f\n",j,i,dot);
        }
    }
    #endif
	// Copy the translation
	for (int i = 0; i < 3; i++) {
        //printf("in aQTM: writing to %d\n",i*3+3);
		matrix[i*4 + 3] = quat[i];
	}
	
	// Set projective part of matrix to the identity
	matrix[12] = matrix[13] = matrix[14] = 0;
	matrix[15] = 1;
	
	return;
} //}}}
	
void printTransformMatrix(const float* matrix,FILE* f) { //{{{
	fprintf(f,"[[ %9.6f %9.6f %9.6f %9.6f\n   %9.6f %9.6f %9.6f %9.6f\n   %9.6f %9.6f %9.6f %9.6f\n   %9.6f %9.6f %9.6f %9.6f]]\n",
		   matrix[0],matrix[1],matrix[2],matrix[3],matrix[4],matrix[5],matrix[6],matrix[7],matrix[8],matrix[9],
		   matrix[10],matrix[11],matrix[12],matrix[13],matrix[14],matrix[15]);
} //}}}

float3 transformSinglePoint(const float* quat,const float3 point) {
    float matrix[16];
    affineQuatToMatrix(quat,matrix);
    float3 newpoint;
    newpoint.x = matrix[0]*point.x + matrix[1]*point.y + matrix[2]*point.z + matrix[3];
    newpoint.y = matrix[4]*point.x + matrix[5]*point.y + matrix[6]*point.z + matrix[7];
    newpoint.z = matrix[8]*point.x + matrix[9]*point.y + matrix[10]*point.z + matrix[11];
    return newpoint;
}
