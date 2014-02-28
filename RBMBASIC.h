// class: RBMBASIC
// Author: daiwenkai
// Date: Feb 24, 2014

#ifndef _RBMBASIC_H_
#define _RBMBASIC_H_


#include "RBM.h"
#include "RBMOpenMP.h"

// class RBMBASIC : public RBM {
class RBMBASIC : public RBMOpenMP {
// Constructors
public:
    RBMBASIC();
    RBMBASIC(string filename);
    virtual ~RBMBASIC();

// Methods
    virtual void train_full(bool reset, int rbmlayers_id);

    void update_hidden_p(double* vs, int* mask, int mask_size, double* hp) ;

    void update_visible_p(double* hs, double* vp, int* mask, int mask_size);

    void update_w_p(double* w_acc, int* w_count, int nth);

    void update_vb_p(double* vb_acc, int* vb_count, int nth);

    void update_hb_p(double* hb_acc, int nth); 

    void sample_visible_p(double* vp, double* vs, int* mask, int mask_size);

// Attributes


};

#endif
