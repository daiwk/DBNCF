/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Implementation of "Model.h".
 *
 * - Author: LOUPPE Gilles
 * - Last changes: March 29, 2010
 * ========================================================================= */
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

#include "Dataset.h"
#include "Dumb.h"
#include "Ensemble.h"
#include "Model.h"
#include "RBM.h"
#include "RBMCF.h"
#include "DBNCF.h"

using namespace std;


/* ========================================================================= *
 * Netflix model
 * ========================================================================= */

/* ------------------------------------------------------------------------- *
 * Constructors
 * ------------------------------------------------------------------------- */

Model::Model(int id) : __id(id) {
    // Nothing to do
    cout << "model id: " << id << endl;
}

Model::~Model() {
    map<string, void*>::iterator it;

    for (it = parameters.begin(); it != parameters.end(); it++) {
        free((*it).second);
    }
}

/* ------------------------------------------------------------------------- *
 * Factory
 * ------------------------------------------------------------------------- */

Model* Model::load(string filename) {
    ifstream in(filename.c_str(), ios::in);
    cout << "Model: " << filename << endl;
    if (in.fail()) {
        throw runtime_error("I/O exception!");
    }

    int type;
    in >> type;
    in.close();

    switch (type) {
        case CLASS_ENSEMBLE:
            return new Ensemble(filename);
            
        case CLASS_DUMB1:
            return new Dumb1(filename);

        case CLASS_DUMB2:
            return new Dumb2(filename);

        case CLASS_DUMB3:
            return new Dumb3(filename);

        case CLASS_RBM:
            return new RBM(filename);

        case CLASS_RBMCF:
            return new RBMCF(filename);

        case CLASS_DBNCF:
            return new DBNCF(filename);

        default:
            throw runtime_error("Unknown model type");
    }
}

/* ------------------------------------------------------------------------- *
 * Sets
 * ------------------------------------------------------------------------- */

void Model::addSet(string key, Dataset* dataset) {
    sets[key] = dataset;
}

void Model::removeSet(string key) {
    sets.erase(key);
}

/* ------------------------------------------------------------------------- *
 * Parameters
 * ------------------------------------------------------------------------- */

void Model::setParameter(string key, const void* value, size_t size) {
    if (parameters[key] != NULL) {
        printf("setting %s\n", key.c_str());
        free(parameters[key]);
    }
    void* block = malloc(size);
    memcpy(block, value, size);
    parameters[key] = block;
}

void* Model::getParameter(string key) {
    if (parameters.find(key) == parameters.end()) {
        throw runtime_error("Unknown '" + key + "' parameter");
    }

    return parameters[key];
}
