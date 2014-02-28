/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Ensemble of models.
 *
 * - Author: LOUPPE Gilles
 * - Last changes: March 29, 2010
 * ========================================================================= */

#ifndef _ENSEMBLE_H_
#define _ENSEMBLE_H_

#include <set>
#include <string>

#include "Dataset.h"
#include "Model.h"

using namespace std;


/* ========================================================================= *
 * Ensemble 
 * ========================================================================= */

class Ensemble : public Model {
public:
    // Constructors
    Ensemble();
    Ensemble(string filename);
    virtual ~Ensemble();

    // Model 
    virtual void train(string dataset="LS", bool reset=true);
    virtual double test(string dataset="TS");
    virtual double predict(int user, int movie);
    virtual void save(string filename);
    virtual string toString();

    // Sets
    virtual void addSet(string key, Dataset* set);
    virtual void removeSet(string key);

    // Sub-models
    virtual void addModel(Model* model);
    virtual void removeModel(Model* model);

    // Attributes
    set<Model*> models;
};

#endif
