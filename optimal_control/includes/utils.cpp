#include "utils.h"
#include "RigidBody/GeneralizedCoordinates.h"
#include "RigidBody/GeneralizedTorque.h"

/******************************************************************************
 * Checks to see if a directory exists. Note: This method only checks the
 * existence of the full path AND if path leaf is a dir.
 *
 * @return  >0 if dir exists AND is a dir,
 *           0 if dir does not exist OR exists but not a dir,
 *          <0 if an error occurred (errno is also set)
 *****************************************************************************/
bool dirExists(const char* const path)
{
    struct stat info;

    int statRC = stat( path, &info );
    if( statRC != 0 )
    {
        if (errno == ENOENT)  { return 0; } // something along the path does not exist
        if (errno == ENOTDIR) { return 0; } // something in path prefix is not a dir
        return -1;
    }

    return ( info.st_mode & S_IFDIR ) ? true : false;
}

void createTreePath(const std::string &path)
{
    if (!dirExists(path.c_str()))
        mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

//void Dispatch_Q_Qdot(double *x)
//{
//    for(unsigned int i = 0; i<nQ; ++i){
//        Q[i] = x[i];
//        Qdot[i] = x[i+nQ];
//    }
//}

//void Dispatch_Q_Qdot_Tau(double *x)
//{
//    for(unsigned int i = 0; i<nQ; ++i){
//        Q[i] = x[i];
//        Qdot[i] = x[i+nQ];
//        Tau[i]= x[i+nQ+nQdot];
//    }
//}
