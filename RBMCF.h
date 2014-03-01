/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * A Restricted Boltzmann Machine model for the Netflix Prize. This model
 * is a direct implementation of the model described in [1].
 *
 * [1]: "Restricted Boltzmann Machines for Collaborative Filtering",
 *      Ruslan Salakuthdinov, Geoffrey Hinton, 2007.
 *
 * - Author: LOUPPE Gilles
 * - Last changes: April 8, 2010
 * ========================================================================= */

#ifndef _RBMCF_H_
#define _RBMCF_H_

#include <iostream>
#include <string>

#include "Dataset.h"
#include "Model.h"

#include <sys/time.h>

using namespace std;


/* ========================================================================= *
 * RBMCF model
 * ========================================================================= */

class RBMCF : public Model {
public:
    // Constructors
    RBMCF();
    RBMCF(string filename);
    virtual ~RBMCF();

    // Model
    virtual void train(string dataset="LS", bool reset=true);
    virtual double test(string dataset="TS");
    virtual double validate(string dataset="VS");
    virtual double predict(int user, int movie);
    virtual void save(string filename);
    virtual string toString();

    // RBMCF methods
    virtual void reset(void);

    virtual void update_hidden(double* vs, int* mask, int mask_size, double* hp);
    virtual void update_hidden(double* vs, int* mask, int mask_size, int* seen, int seen_size, double* hp);
    virtual void sample_hidden(double* hp, double* hs);
    virtual void update_visible(double* hs, double* vp, int* mask, int mask_size);
    virtual void sample_visible(double* vs, double* vp, int* mask, int mask_size);

    virtual void update_w(double* w_acc, int* w_count, int nth);
    virtual void update_vb(double* vb_acc, int* vb_count, int nth);
    virtual void update_hb(double* hb_acc, int nth);
    virtual void update_d(double* d_acc, bool* watched, int nth);
    
    // 只更新LS的某些batch
    // batch: 此minbatch的第一个user的下标
    virtual void train_batch(string dataset="LS", bool reset=true, int batch=0);
    
    // Attributes
    int N;
    int M;
    int K;
    int F;
    bool conditional;
    
    double* w;                // Weights
    double* w_inc;            // Weights inc
    double* vb;               // Biases of visible units
    double* vb_inc;           // Biases inc of visible units
    double* hb;               // Biases of hidden units
    double* hb_inc;           // Biases inc of hidden units
    double* d;                // D matrix
    double* d_inc;            // D matrix inc
};

#endif
