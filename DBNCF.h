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
    DBNCF();
    DBNCF(string filename);
    virtual ~DBNCF();

    // Model
    virtual void train(string dataset="LS", bool reset=true);
    virtual double test(string dataset="TS");
    virtual double predict(int user, int movie);
    virtual void save(string filename);
    virtual string toString();
   
    // Methods
    
    // Attributes
    
    // 可以通过读配置文件Configuration.h得到
    int train_epochs;  // Config::DBNCF::TRAIN_EPOCHS
    int batch_size;    // Config::DBNCF::BATCH_SIZE

    // 各个隐含层的节点个数，配置中默认每层节点个数相同,默认20
    // 但一般不是通过读配置，而是手动作为参数传入的
    int* hidden_layer_sizes;  // Config::DBNCF::HL_SIZE 
    
    // 隐含层的层数
    // 配置中默认是2
    // hidden_layer_num = sizeof(hidden_layer_sizes) / sizeof(hidden_layer_sizes[0]);
    int hidden_layer_num;     // Config::DBNCF::HL_NUM
    
    // 第一层是RBMCF，中间几层都是RBMBASIC，最后一层是倒置的RBMCF
    RBMCF* input_layer;
    RBMBASIC** hidden_layers;
    RBMCF* output_layer;

};

#endif
