// class: AHRBMCF
// Author: daiwenkai
// Date: Feb 24, 2014

#ifndef _AHRBMCF_H_
#define _AHRBMCF_H_

#include "RBMBASIC.h"
#include "RBMCF.h"
#include "Model.h"

#include <sys/time.h>

using namespace std; 

class AHRBMCF : public Model {

public:

    // 构造函数与析构函数
    // 默认构造函数
    AHRBMCF();
    
    // 读取模型文件生成AHRBMCF的构造函数
    AHRBMCF(string filename);

    // 析构函数
    virtual ~AHRBMCF();

    // Model的函数
    virtual void train(string dataset="LS", bool reset=true);
    virtual double test(string dataset="TS");
    virtual double predict(int user, int movie);
    virtual void save(string filename);
    virtual string toString();
   
    // AHRBMCF的函数
    virtual void pretrain_old_version(string dataset="LS", bool reset=true);
    virtual void pretrain(string dataset="LS", bool reset=true);
    virtual void finetune(string dataset="QS");
    virtual void train_separate(string dataset="LS", bool reset=true);
    
    // 成员变量
    // 可以通过读配置文件Configuration.h得到
    int train_epochs;  // Config::AHRBMCF::TRAIN_EPOCHS
    int batch_size;    // Config::AHRBMCF::BATCH_SIZE

    // 各个隐含层的节点个数，配置中默认每层节点个数相同,默认20
    // 但一般不是通过读配置，而是手动作为参数传入的
    int* hidden_layer_sizes;  // Config::AHRBMCF::HL_SIZE 
    
    // 隐含层的层数
    // 配置中默认是2
    // hidden_layer_num = sizeof(hidden_layer_sizes) / sizeof(hidden_layer_sizes[0]);
    int hidden_layer_num;     // Config::AHRBMCF::HL_NUM
    
    // 第一层是RBMCF，中间几层都是RBMBASIC，最后一层是倒置的RBMCF
    RBMCF* input_layer;
    RBMBASIC** hidden_layers;
    RBMCF* output_layer;

};

#endif
