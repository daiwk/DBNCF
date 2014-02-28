/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Implementation of "Ensemble.h".
 *
 * - Author: LOUPPE Gilles
 * - Last changes: March 29, 2010
 * ========================================================================= */

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

#include "Configuration.h"
#include "Dataset.h"
#include "Ensemble.h"
#include "Model.h"

using namespace std;
using namespace Config::Netflix;


/* ========================================================================= *
 * Ensemble
 * ========================================================================= */

/* ------------------------------------------------------------------------- *
 * Constructors
 * ------------------------------------------------------------------------- */

Ensemble::Ensemble() : Model(CLASS_ENSEMBLE) {
    // Nothing to do
}

Ensemble::Ensemble(string filename) : Model(CLASS_ENSEMBLE) {
    ifstream in(filename.c_str(), ios::in);

    if (in.fail()) {
        throw runtime_error("I/O exception!");
    }

    int id;
    in >> id;
    assert(id == CLASS_ENSEMBLE);

    int size;
    in >> size;

    for (int i = 0; i < size; i++) {
        string s;
        in >> s;
        models.insert(load(s));
    }

    in.close();
}

Ensemble::~Ensemble() {
    // Nothing to do
}

/* ------------------------------------------------------------------------- *
 * Model
 * ------------------------------------------------------------------------- */

void Ensemble::train(string dataset, bool reset) {
    Dataset* LS = sets[dataset];
    assert(LS != NULL);

    set<Model*>::iterator it;

    for (it = models.begin(); it != models.end(); it++) {
        (*it)->train(dataset, reset);
    }
}

double Ensemble::test(string dataset) {
    Dataset* TS = sets[dataset];
    assert(TS != NULL);

    double error = 0.;

    #pragma omp parallel for schedule(guided) reduction(+: error)
    for (int n = 0; n < TS->nb_rows; n++) {
        if (TS->count[n] <= 0) {
            continue;
        }

        int u = TS->rows[n];

        for (int m = TS->index[n]; m < TS->index[n] + TS->count[n]; m++) {
            double e = TS->ratings[m] - predict(u, TS->ids[m]);
            error += e * e;
        }
    }

    return sqrt(error / TS->nb_ratings);
}

double Ensemble::predict(int user, int movie) {
    // Asserts
    assert(user >= 0);
    assert(user < NB_USERS);
    assert(movie >= 0);
    assert(movie < NB_MOVIES);

    // Prediction
    vector<double> predictions;
    set<Model*>::iterator it1;

    for (it1 = models.begin(); it1 != models.end(); it1++){
        double p = (*it1)->predict(user, movie);

        if (p > 0.){
            predictions.push_back(p);
        }
    }

    // Return average
    if (predictions.size() <= 0) {
        return 3.6033; // TODO: find something better!
    }

    double s = 0.;
    vector<double>::iterator it2;

    for (it2 = predictions.begin(); it2 != predictions.end(); it2++){
        s += *it2;
    }

    return s / predictions.size();
}

void Ensemble::save(string filename) {
    ofstream out(filename.c_str(), ios::out);

    if (out.fail()) {
        throw runtime_error("I/O exception!");
    }

    out << __id << endl;
    out << models.size() << endl;

    set<Model*>::iterator it;
    int i = 1;

    for (it = models.begin(); it != models.end(); it++) {
        stringstream s;
        s << filename << "." << i++;
        out << s.str() << endl;
        (*it)->save(s.str());
    }

    out.close();
}

string Ensemble::toString() {
    stringstream s;
    s << "<Ensemble>" << endl;

    set<Model*>::iterator it;

    for (it = models.begin(); it != models.end(); it++) {
        s << "-- " << (*it)->toString() << endl;
    }

    return s.str();
}

/* ------------------------------------------------------------------------- *
 * Sets
 * ------------------------------------------------------------------------- */

void Ensemble::addSet(string key, Dataset* dataset) {
    Model::addSet(key, dataset);

    set<Model*>::iterator it;

    for (it = models.begin(); it != models.end(); it++) {
        (*it)->addSet(key, dataset);
    }
}

void Ensemble::removeSet(string key) {
    Model::removeSet(key);

    set<Model*>::iterator it;

    for (it = models.begin(); it != models.end(); it++) {
        (*it)->removeSet(key);
    }
}

/* ------------------------------------------------------------------------- *
 * Sub-models
 * ------------------------------------------------------------------------- */

void Ensemble::addModel(Model* model) {
    models.insert(model);
}

void Ensemble::removeModel(Model* model) {
    models.erase(model);
}

