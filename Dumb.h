/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Dumb models.
 *
 * - Author: LOUPPE Gilles
 * - Last changes: February 26, 2010
 * ========================================================================= */

#ifndef _DUMB_H_
#define _DUMB_H_

#include <string>

#include "Dataset.h"
#include "Model.h"

using namespace std;


/* ========================================================================= *
 * Dumb model 1
 *
 * r^(u, m) = mean value of all ratings r
 * ========================================================================= */

class Dumb1 : public Model {
public:
    // Constructors
    Dumb1();
    Dumb1(string filename);
    virtual ~Dumb1();

    // Model 
    virtual void train(string dataset="LS", bool reset=true);
    virtual double test(string dataset="TS");
    virtual double predict(int user, int movie);
    virtual void save(string filename);
    virtual string toString();

    // Attributes
    double mean;
};


/* ========================================================================= *
 * Dumb model 2
 *
 * r^(u, m) = mean value of r(m)
 * ========================================================================= */

class Dumb2 : public Model {
public:
    // Constructors
    Dumb2();
    Dumb2(string filename);
    virtual ~Dumb2();

    // Model 
    virtual void train(string dataset="LS", bool reset=true);
    virtual double test(string dataset="TS");
    virtual double predict(int user, int movie);
    virtual void save(string filename);
    virtual string toString();

    // Attributes
    double* means;
};


/* ========================================================================= *
 * Dumb model 3
 *
 * r^(u, m) = mean value of r(u)
 * ========================================================================= */

class Dumb3 : public Model {
public:
    // Constructors
    Dumb3();
    Dumb3(string filename);
    virtual ~Dumb3();

    // Model
    virtual void train(string dataset="LS", bool reset=true);
    virtual double test(string dataset="TS");
    virtual double predict(int user, int movie);
    virtual void save(string filename);
    virtual string toString();

    // Attributes
    double* means;
};

#endif
