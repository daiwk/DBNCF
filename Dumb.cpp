/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Implementation of "Dumb.h".
 *
 * - Author: LOUPPE Gilles
 * - Last changes: February 26, 2010
 * ========================================================================= */

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>

#include "Configuration.h"
#include "Dataset.h"
#include "Dumb.h"
#include "Model.h"

using namespace std;
using namespace Config::Netflix;


/* ========================================================================= *
 * Dumb model 1
 * ========================================================================= */

/* ------------------------------------------------------------------------- *
 * Constructors
 * ------------------------------------------------------------------------- */

Dumb1::Dumb1() :
    Model(CLASS_DUMB1),
    mean(-1.) {
        // Nothing more to do
}

Dumb1::Dumb1(string filename) : Model(CLASS_DUMB1) {
    ifstream in(filename.c_str(), ios::in);

    if (in.fail()) {
        throw runtime_error("I/O exception!");
    }

    int id;
    in >> id;
    assert(id == CLASS_DUMB1);

    in >> mean;

    in.close();
}

Dumb1::~Dumb1() {
    // Nothing to do
}

/* ------------------------------------------------------------------------- *
 * Model
 * ------------------------------------------------------------------------- */

void Dumb1::train(string dataset, bool reset) {
    if (!reset) return;

    Dataset* LS = sets[dataset];
    assert(LS != NULL);
    double total = 0.;

    for (int r = 0; r < LS->nb_ratings; r++) {
        total += LS->ratings[r];
    }

    mean = total / LS->nb_ratings;
}

double Dumb1::test(string dataset) {
    Dataset* TS = sets[dataset];
    assert(TS != NULL);

    double error = 0.;

    for (int r = 0; r < TS->nb_ratings; r++) {
        error += (TS->ratings[r] - mean) * (TS->ratings[r] - mean);
    }

    return sqrt(error / TS->nb_ratings);
}

double Dumb1::predict(int user, int movie) {
    // Asserts
    assert(user >= 0);
    assert(user < NB_USERS);
    assert(movie >= 0);
    assert(movie < NB_MOVIES);

    // Prediction
    return mean;
}

void Dumb1::save(string filename) {
    ofstream out(filename.c_str(), ios::out);

    if (out.fail()) {
        throw runtime_error("I/O exception!");
    }

    out << __id << endl;
    out << mean;

    out.close();
}

string Dumb1::toString() {
    return "<Dumb 1>";
}


/* ========================================================================= *
 * Dumb model 2
 * ========================================================================= */

/* ------------------------------------------------------------------------- *
 * Constructors
 * ------------------------------------------------------------------------- */

Dumb2::Dumb2() :
    Model(CLASS_DUMB2),
    means(new double[NB_MOVIES]) {
        // Nothing more to do
}

Dumb2::Dumb2(string filename) : Model(CLASS_DUMB2) {
    ifstream in(filename.c_str(), ios::in);

    if (in.fail()) {
        throw runtime_error("I/O exception!");
    }

    int id;
    in >> id;
    assert(id == CLASS_DUMB2);

    means = new double[NB_MOVIES];

    for (int m = 0; m < NB_MOVIES; m++){
        in >> means[m];
    }

    in.close();
}

Dumb2::~Dumb2() {
    delete[] means;
}

/* ------------------------------------------------------------------------- *
 * Model 
 * ------------------------------------------------------------------------- */

void Dumb2::train(string dataset, bool reset) {
    if (!reset) return;

    Dataset* LS = sets[dataset];
    assert(LS != NULL);

    int counts[NB_MOVIES];

    for (int m = 0; m < NB_MOVIES; m++) {
        means[m] = 0.;
        counts[m] = 0;
    }

    for (int r = 0; r < LS->nb_ratings; r++) {
        means[LS->ids[r]] += LS->ratings[r];
        counts[LS->ids[r]]++;
    }

    for (int m = 0; m < NB_MOVIES; m++) {
        if (counts[m] == 0) {
            means[m] = -1.0;
        } else {
            means[m] /= counts[m];
        }
    }
}

double Dumb2::test(string dataset) {
    Dataset* TS = sets[dataset];
    assert(TS != NULL);

    double error = 0.;

    for (int r = 0; r < TS->nb_ratings; r++) {
        double e = TS->ratings[r] - means[TS->ids[r]];
        error += e * e;
    }

    return sqrt(error / TS->nb_ratings);
}

double Dumb2::predict(int user, int movie) {
    // Asserts
    assert(user >= 0);
    assert(user < NB_USERS);
    assert(movie >= 0);
    assert(movie < NB_MOVIES);

    // Prediction
    return means[movie];
}

void Dumb2::save(string filename) {
    ofstream out(filename.c_str(), ios::out);

    if (out.fail()) {
        throw runtime_error("I/O exception!");
    }

    out << __id << endl;

    for (int m = 0; m < NB_MOVIES; m++){
        out << means[m] << endl;
    }

    out.close();
}

string Dumb2::toString() {
    return "<Dumb 2>";
}


/* ========================================================================= *
 * Dumb model 3
 * ========================================================================= */

/* ------------------------------------------------------------------------- *
 * Constructors
 * ------------------------------------------------------------------------- */

Dumb3::Dumb3() :
    Model(CLASS_DUMB3),
    means(new double[NB_USERS]) {
        // Nothing more to do
}

Dumb3::Dumb3(string filename) : Model(CLASS_DUMB3) {
    ifstream in(filename.c_str(), ios::in);

    if (in.fail()) {
        throw runtime_error("I/O exception!");
    }

    int id;
    in >> id;
    assert(id == CLASS_DUMB3);

    means = new double[NB_USERS];

    for (int u = 0; u < NB_USERS; u++){
        in >> means[u];
    }

    in.close();
}

Dumb3::~Dumb3() {
    delete[] means;
}

/* ------------------------------------------------------------------------- *
 * Model 
 * ------------------------------------------------------------------------- */

void Dumb3::train(string dataset, bool reset) {
    if (!reset) return;
    
    Dataset* LS = sets[dataset];
    assert(LS != NULL);

    for (int u = 0; u < NB_USERS; u++) {
        if (LS->contains_user(u)) {
            int n = LS->users[u];

            if (LS->count[n] <= 0){
                means[u] = -1.0;
                continue;
            }

            means[u] = 0.;

            for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
                means[u] += LS->ratings[m];
            }

            means[u] /= LS->count[n];
        } else {
            means[u] = -1.0;
        }
    }
}

double Dumb3::test(string dataset) {
    Dataset* TS = sets[dataset];
    assert(TS != NULL);
    
    double error = 0.;

    for (int n = 0; n < TS->nb_rows; n++){
        if (TS->count[n] <= 0){
            continue;
        }

        int u = TS->rows[n];

        for (int m = TS->index[n]; m < TS->index[n] + TS->count[n]; m++){
            double e = TS->ratings[m] - means[u];
            error += e * e;
        }
    }

    return sqrt(error / TS->nb_ratings);
}

double Dumb3::predict(int user, int movie) {
    // Asserts
    assert(user >= 0);
    assert(user < NB_USERS);
    assert(movie >= 0);
    assert(movie < NB_MOVIES);

    // Prediction
    return means[user];
}

void Dumb3::save(string filename) {
    ofstream out(filename.c_str(), ios::out);

    if (out.fail()) {
        throw runtime_error("I/O exception!");
    }

    out << __id << endl;

    for (int u = 0; u < NB_USERS; u++){
        out << means[u] << endl;
    }

    out.close();
}

string Dumb3::toString() {
    return "<Dumb 3>";
}
