// class: DBNCF
// Author: daiwenkai
// Date: Feb 24, 2014

#ifndef _DBNCF_H_
#define _DBNCF_H_

#include "RBMBASIC.h"
#include "RBMCF.h"
#include "Model.h"

#include <sys/time.h>

using namespace std; 

class DBNCF : public Model {

public:

    // Constructors
    DBNCF(int tr_num, int tr_size, int* l_sizes, int l_num);
    virtual ~DBNCF();

    // Model
    virtual void train(string dataset="LS", bool reset=true);
    virtual double test(string dataset="TS");
    virtual double predict(int user, int movie);
    virtual void save(string filename);
    virtual string toString();

   
    // Methods
    
    // Attributes
    int train_num;  // 对应nb_rows?
    int train_size; // 对应nb_columns?
    int* layer_sizes;
    int layer_num;
    RBMBASIC** rbm_layers;
//    RBM* rbm_layer;
    RBMCF* rbm_layer;

    int train_epochs;

};

#endif
