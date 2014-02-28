/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * A MapReduce Restricted Boltzmann Machine model for the Netflix Prize.
 *
 * - Author: LOUPPE Gilles
 * - Last changes: April 9, 2010
 * ========================================================================= */

#ifndef _RBM_MAPREDUCE_H_
#define _RBM_MAPREDUCE_H_

#include <iostream>
#include <string>

#include "Dataset.h"
#include "RBM.h"

#include "mapreduce.h"
#include "keyvalue.h"

using namespace std;
using namespace MAPREDUCE_NS;


/* ========================================================================= *
 * MapReduce RBM model
 * ========================================================================= */

/* ------------------------------------------------------------------------- *
 * RBM class
 * ------------------------------------------------------------------------- */

class RBMMapReduce : public RBM {
public:
    // Constructors
    RBMMapReduce(int rank, int np);
    RBMMapReduce(int rank, int np, string filename);
    virtual ~RBMMapReduce();

    // Model
    virtual void train(string dataset="LS", bool reset=true);
    virtual double test(string dataset="TS");
    virtual string toString();

    // MapReduce
    int rank;
    int np;

    /* Train */
    string __train_dataset;
    int __train_epoch;
    map<int, pair<char, int> > __train_keys;

    /* Test */
    string __test_dataset;
    double __test_total;
    int __test_count;
};

/* ------------------------------------------------------------------------- *
 * Train over MapReduce
 * ------------------------------------------------------------------------- */

void train_map(int task, KeyValue* kv, void* ptr);

void train_reduce(char* key, int key_size,
                  char* values, int nb_values, int* value_sizes,
                  KeyValue* kv, void* ptr);

void train_update(uint64_t i,
                  char* key, int key_size,
                  char* value, int value_size,
                  KeyValue* kv, void* ptr);

/* ------------------------------------------------------------------------- *
 * Test over MapReduce
 * ------------------------------------------------------------------------- */

void test_map(int task, KeyValue* kv, void* ptr);

void test_reduce(uint64_t i,
                 char* key, int key_size,
                 char* value, int value_size,
                 KeyValue* kv, void* ptr);

#endif
