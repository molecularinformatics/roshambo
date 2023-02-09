#include "cudaVolumeTypes.h"
#define GRID_VOLUME
#include "hostVolume.h"
#include "inputFileReader.h"
#include <iostream>
#include <cstdio>
#include <cstring>

using namespace std;

void matrix_transform(CUDAmol& cmol,float* rot,float* tran);
//Grid getHostGrid(CUDAmol& mol,float res,float margin);
//Grid getHostGridFromBox(float3 lb,float3 ub,float res);

int main(int argc,char** argv) {
    if (argc < 3) {
        cout << "Usage: evaluate [arg file] [result file]\n";
        return 1;
    }
    FILE* argfile,*resfile;
    argfile = fopen(argv[1],"r");
    resfile = fopen(argv[2],"r");
    if (!argfile || !resfile) {
        cout << "Error reading argument or result file\n";
        return 2;
    }
    char molfilename[1024],dummy[1024];
    fgets(molfilename,1024,argfile);
    *(char*)(memchr(molfilename,'\n',1024))='\0';

    list<CUDAmol> refmolconformers;
    list<list<float3> > ref_centroids;
    molFromFile(molfilename,refmolconformers,ref_centroids);
    CUDAmol refmol = refmolconformers.front();
    
    float gridres=0.4,gridmargin = 0.4;
    Grid grid = getHostGrid(refmol,gridres,gridmargin);
    // Using grid parameters (0.5,0.0) produces Tanimoto accurate to two decimal places
    // in informal testing vs (0.3,1.0). Good enough for visual resolution in plot.
    //Grid hrgrid = getHostGrid(refmol,0.3,gridmargin+1);
    float Oaa = getOverlapVolume(refmol,refmol,grid);
    //float hrOaa = getOverlapVolume(refmol,refmol,hrgrid);
    int refmolid;
    sscanf(molfilename,"/tmp/ramtmp/pcasdf/%d.sdf",&refmolid);
    //printf("# Refmol,Fitmol,Overlap, Tanimoto\n");
    do {
        fgets(molfilename,1024,argfile);
        if (feof(argfile))
            break;
        char* nl = (char*)(memchr(molfilename,'\n',1024));
        if (nl != NULL)
            *nl = '\0';

        list<CUDAmol> fitmolconformers;
        list<list<float3> > fit_centroids;
        molFromFile(molfilename,fitmolconformers,fit_centroids);
        CUDAmol fitmol = fitmolconformers.front();
        int fitmolid;
        sscanf(molfilename,"/tmp/ramtmp/pcasdf/%d.sdf",&fitmolid);
        float rot[9];
        float tran[3];
        fscanf(resfile,"[[ %f %f %f %f\n",&rot[0],&rot[1],&rot[2],&tran[0]);
        fscanf(resfile,"   %f %f %f %f\n",&rot[3],&rot[4],&rot[5],&tran[1]);
        fscanf(resfile,"   %f %f %f %f\n",&rot[6],&rot[7],&rot[8],&tran[2]);
        // Throw away the last row of the transformation matrix 
        fgets(dummy,1024,resfile);
        #if 0
        printf("Transformation matrix read in for file %s:\n",molfilename);
        printf("[[ %f %f %f %f\n",rot[0],rot[1],rot[2],tran[0]);
        printf("   %f %f %f %f\n",rot[3],rot[4],rot[5],tran[1]);
        printf("   %f %f %f %f\n",rot[6],rot[7],rot[8],tran[2]);
        printf("   0 0 0 1 ]]\n",rot[6],rot[7],rot[8],tran[2]);
        #endif

        // Apply transformation matrix
        matrix_transform(fitmol,rot,tran);
        // Calculate overlap with reference
        float Oab;
        //float hrOab;
        Grid fitgrid = getHostGrid(fitmol,gridres,gridmargin);
        //Grid hrfitgrid = getHostGrid(fitmol,gridres,gridmargin+1);
        if ((grid.extent.x * grid.extent.y * grid.extent.z) < (fitgrid.extent.x * fitgrid.extent.y * fitgrid.extent.z)) {
            Oab = getOverlapVolume(refmol,fitmol,grid);
            //hrOab = getOverlapVolume(refmol,fitmol,hrgrid);
        } else {
            Oab = getOverlapVolume(refmol,fitmol,fitgrid);
            //hrOab = getOverlapVolume(refmol,fitmol,hrfitgrid);
        }
        float Obb = getOverlapVolume(fitmol,fitmol,fitgrid);
        //float hrObb = getOverlapVolume(fitmol,fitmol,hrfitgrid);
        //printf("Tanimoto on refgrid = %f, tanimoto on fitgrid = %f\n",Oab/(Oaa+Obb-Oab),Oab_fit/(Oaa+Obb-Oab_fit));
        //printf("Tanimoto = %f, HR tanimoto = %f\n",Oab/(Oaa+Obb-Oab),hrOab/(hrOaa+hrObb-hrOab));
        //printf("Points on refgrid = %d, points on fitgrid = %d\n",grid.extent.x*grid.extent.y*grid.extent.z, fitgrid.extent.x*fitgrid.extent.y*fitgrid.extent.z);
        float tanimoto = Oab/(Oaa+Obb-Oab);
        //printf("%d,%f,%f\n",fitmolid,result.overlap,result.tanimoto);
        printf("%d,%d,%f,%f\n",refmolid,fitmolid,Oab,tanimoto);
        delete[] fitmol.atoms;
            
    } while (!feof(argfile));
    delete[] refmol.atoms;
    fclose(argfile);
    fclose(resfile);
    return 0;
}

void matrix_transform(CUDAmol& cmol,float* rot,float* tran) {
    float4 newcor;
    float4 oldcor;
    for (uint i = 0; i < cmol.natoms; i++) {
        newcor = oldcor = cmol.atoms[i];
        newcor.x = rot[0]*oldcor.x + rot[1]*oldcor.y + rot[2]*oldcor.z + tran[0];
        newcor.y = rot[3]*oldcor.x + rot[4]*oldcor.y + rot[5]*oldcor.z + tran[1];
        newcor.z = rot[6]*oldcor.x + rot[7]*oldcor.y + rot[8]*oldcor.z + tran[2];
        cmol.atoms[i] = newcor;
    }
}

/*Grid getHostGridFromBox(float3 lb,float3 ub,float res) {
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
    printf("Box extent: %ux%ux%u\n",retval.extent.x,retval.extent.y,retval.extent.z);
    //retval.points=new float[extent.x*extent.y*extent.z];
    retval.points = NULL;
    return retval;
}

Grid getHostGrid(CUDAmol& mol,float res,float margin) {
    // Calculate bounding box
    float3 ub,lb;
    boundingBox(mol,margin,lb,ub);
    return getHostGridFromBox(lb,ub,res);
    //printf("Got bounding box: [%f,%f,%f] - [%f,%f,%f]\n",retval.lb.x,retval.lb.y,retval.lb.z,ub.x,ub.y,ub.z);
}*/

