// class: DBN
// Author: daiwenkai
// Date: Feb 24, 2014

#ifndef _DBN_H_
#define _DBN_H_

#include "RBM_P.h"

#include <sys/time.h>

using namespace std; 

class DBN {

public:

    // Constructors
    DBN(int tr_num, int tr_size, int* l_sizes, int l_num);
    virtual ~DBN();

    // Model
    virtual void train();
    virtual void test();
    
    // Methods
    
    // Attributes
    int train_num;  // 对应nb_rows?
    int train_size; // 对应nb_columns?
    int* layer_sizes;
    int layer_num;
    RBM_P** rbm_layers;
//    RBM* rbm_layer;
    RBMOpenMP* rbm_layer;

    int train_epochs;

};

#endif
