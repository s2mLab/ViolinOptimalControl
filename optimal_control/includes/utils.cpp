#include "utils.h"
#include "RigidBody/GeneralizedCoordinates.h"
#include "RigidBody/GeneralizedTorque.h"
#include <iostream>
#include <fstream>

/******************************************************************************
 * Checks to see if a directory exists. Note: This method only checks the
 * existence of the full path AND if path leaf is a dir.
 *
 * @return  >0 if dir exists AND is a dir,
 *           0 if dir does not exist OR exists but not a dir,
 *          <0 if an error occurred (errno is also set)
 *****************************************************************************/
void createTreePath(const std::string &path)
{
    if (!dirExists(path.c_str()))
        mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

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

void dispatchQ(const double *x) {
    for (unsigned int i = 0; i<nQ; ++i){
        Q[i] = x[i];
    }
}

void dispatchQandQdot(const double *x) {
    for (unsigned int i = 0; i<nQ; ++i){
        Q[i] = x[i];
        Qdot[i] = x[i+nQ];
    }
}

void dispatchActivation(const double *x){
    for (unsigned int i = 0; i<nMus; ++i) {
        musclesStates[i]->setActivation(x[nQ + nQdot + i]);
    }
}

void initializeMuscleStates() {
    for(unsigned int i = 0; i<nMus; ++i){
        musclesStates[i] = std::make_shared<biorbd::muscles::StateDynamics>(
                    biorbd::muscles::StateDynamics());
    }
}

void projectOnXyPlane(double *x, double *g, void *user_data)
{
    dispatchQ(x);
    unsigned int markerToProject = static_cast<unsigned int*>(user_data)[0];
    unsigned int idxSegmentToProjectOn = static_cast<unsigned int*>(user_data)[1];

    biorbd::utils::RotoTrans rt = m.globalJCS(Q, idxSegmentToProjectOn);
    biorbd::rigidbody::NodeBone markerProjected = m.marker(Q, markerToProject, false, false);
    markerProjected.applyRT(rt.transpose());

    g[0] = markerProjected[0];
    g[1] = markerProjected[1];
}

// Permet de retirer les crochets [] d'un fichier de sortie
void removeSquareBracketsInFile(
        const std::string& originFilePath,
        const std::string& targetFilePath) {
    std::ifstream originFile(originFilePath.c_str());
    if (!originFile) {
        std::cout << originFilePath << " could not be opened" << std::endl;
    }

    std::ofstream targetFile(targetFilePath.c_str());
    if (!targetFile) {
        std::cout << targetFilePath << " could not be opened" << std::endl;
    }

    std::string line;
    if (targetFile) {
        while (getline(originFile, line)){
            for (unsigned int i = 0; i < line.size(); ++i) {
                if (line [i] == '[') {
                    line.erase(i,1);
                }
                else if (line [i] == ']') {
                 line.erase(i,1);
                }
            }
            targetFile << line << std::endl;
        }
    }
}

void validityCheck(){
#ifdef CHECK_MAX_FORCE
    for(unsigned int int i=0; i<m.nbTau(); ++i){
        if (Tau[i]>1e5){
            std::vector<int> L;
            for(unsigned int int j=0; j<nMus; ++j){
                if (m.musclesLengthJacobian(m, true, &Q).coeff(j,i)!=0){
                    L.push_back(j);
                }
            }
            std::cout << "Torque "<<i<<" is too high" << std::endl;
            // std::cout << "Check the optimal lenth, the maximal force or the tendon slack lenth of muscles " << L << std::endl;
            L.clear();
        }
    }
#endif

#ifdef CHECK_FORCE_IF_LOW_ACTIVATION
    // Check if muscle forces are not too high if muscle activation is low
    for(unsigned int int i=0; i<m.nbTau(); ++i){
        if (Tau[i]>0.1){
            int c=0;
            std::vector<int> L;
            for(unsigned int int j=0; j<nMus; ++j){
                if (m.musclesLengthJacobian(m, true, &Q).coeff(j,i)!=0){
                    L.push_back(j);
                    if(musclesStates[j].activation()>0.015)
                        c++;
                }
            }
            if (c=0)
                std::cout << "Passive force of the muscles " <<L<< " is too high. Check the tendon slack lenth."<<std::endl;
        }
    }
#endif

#ifdef CHECK_MUSCLE_LENGTH_IS_POSITIVE
    for(unsigned int int i=0; i<m.nbMuscleGroups(); ++i){
        for(unsigned int int j=0; j<m.muscleGroup(i).nbMuscles(); ++j)
            if (m.muscleGroup(i).muscle(j).get()->length(m, Q) <= 0)
                std::cout << "La longueur du muscle " << i << " est inférieur á 0" <<std::endl;
    }
#endif
}
