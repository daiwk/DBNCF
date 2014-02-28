/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * A concurrent Restricted Boltzmann Machine model for the Netflix Prize.
 * 
 * - Author: LOUPPE Gilles
 * - Last changes: February 26, 2010
 * ========================================================================= */

#ifndef _RBM_OPENMP_H_
#define _RBM_OPENMP_H_

#include <iostream>
#include <string>

#include "Dataset.h"
#include "RBM.h"

using namespace std;


/* ========================================================================= *
 * OpenMP RBM model
 * ========================================================================= */

class RBMCF_OPENMP : public RBMCF {
public:
    // Constructors
    RBMCF_OPENMP();
    RBMCF_OPENMP(string filename);
    virtual ~RBMCF_OPENMP();

    // Model
    virtual void train(string dataset="LS", bool reset=true);
    virtual double test(string dataset="TS");
    virtual string toString();
};

#endif
