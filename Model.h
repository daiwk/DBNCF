/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Abstract Netflix model.
 *
 * - Author: LOUPPE Gilles
 * - Last changes: March 29, 2010
 * ========================================================================= */

#ifndef _MODEL_H_
#define _MODEL_H_

#include <map>
#include <string>

#include "Dataset.h"

using namespace std;


/* ========================================================================= *
 * Class type
 * ========================================================================= */

const int CLASS_ENSEMBLE = 0;
const int CLASS_DUMB1 = 1;
const int CLASS_DUMB2 = 2;
const int CLASS_DUMB3 = 3;
const int CLASS_RBM = 4;
const int CLASS_RBMCF = 5;
const int CLASS_DBNCF = 6;
const int CLASS_DAECF = 7;


/* ========================================================================= *
 * General model  
 * ========================================================================= */

class Model {
public:
    // Constructors
    Model(int id);
    virtual ~Model();

    // Factory
    static Model* load(string filename);

    // Pure virtual methods
    virtual void train(string dataset="LS", bool reset=true) = 0;
    virtual double test(string dataset="TS") = 0;
    virtual double predict(int user, int movie) = 0;
    virtual void save(string filename) = 0;
    virtual string toString() = 0;

    // Sets
    virtual void addSet(string key, Dataset* dataset);
    virtual void removeSet(string key);
    map<string, Dataset*> sets;

    // Parameters
    virtual void setParameter(string key, const void* value, size_t size);
    virtual void* getParameter(string key);
    map<string, void*> parameters;
    
    // Misc
    int __id;
};

#endif
