// class: DBNCF
// Author: daiwenkai
// Date: Feb 24, 2014

#include "DBNCF.h"
#include "RBMBASIC.h"
#include "Configuration.h"
#include "stdio.h"

using namespace std;

// Constructors

DBNCF::DBNCF(int tr_num, int tr_size, int* l_sizes, int l_num) : Model(CLASS_DBNCF)
{
    train_num = tr_num;
    train_num = tr_size;
    layer_num = l_num;
    rbm_layers = new RBMBASIC*[layer_num];
    layer_sizes = l_sizes;
    train_epochs = Config::DBNCF::TRAIN_EPOCHS;

    // 注意，第一层是正常的rbm for cf，使用原来的RBM参数。
    // 后面几个就是K=1, N=1, M=前一层的节点数
    char ss[1000];
    sprintf(ss, "rbm-%d", 0);
    string rbm_name = ss;
//    RBM r = RBM();
    RBMCF r = RBMCF();
    r.setParameter("F", &l_sizes[0], sizeof(int));
//    r.setParameter("epochs", &Config::RBMBASIC::EPOCHS, sizeof(int)); //临时这么初始化
    r.reset();
//            cout << "r.F: "<<r.F <<endl;
    r.save(rbm_name);
//    rbm_layer = new RBM(rbm_name);
    rbm_layer = new RBMCF(rbm_name);



    for(int i = 0; i < layer_num - 1; i++) {
        char ss[1000];
        sprintf(ss, "rbm-%d", i + 1);
        string rbm_name = ss;

        RBMBASIC r_p = RBMBASIC();
        r_p.setParameter("N", &Config::RBMBASIC::N, sizeof(int));
        r_p.setParameter("M", &l_sizes[i], sizeof(int));
        r_p.setParameter("K", &Config::RBMBASIC::K, sizeof(int));
        r_p.setParameter("epochs", &Config::RBMBASIC::EPOCHS, sizeof(int));
        r_p.setParameter("F", &l_sizes[i + 1], sizeof(int));
        r_p.reset();
//             printf("res: N:%d,M:%d,K:%d,F%d\n", r.N, r.M, r.K, r.F);
//             printf("expected: N:%d,M:%d,K:%d,F%d\n", Config::RBM_DBNCF::N, l_sizes[i - 1], Config::RBM_DBNCF::K, l_sizes[i]);
        r_p.save(rbm_name);
        rbm_layers[i] = new RBMBASIC(rbm_name);
    }

}

DBNCF::~DBNCF()
{
    delete rbm_layer;
    for(int i =0; i < layer_num - 1; i++) {
        delete rbm_layers[i];
    }

}

// Model
void DBNCF::train(string dataset="LS", bool reset=true)
{
    
    printf("DBNCF epochs: %d\n", train_epochs);
    rbm_layer->train();
    printf("before iterations...\n");
    printf("RMSE: %lf\n", rbm_layer->test());
    printf("training RMSE: %lf\n", rbm_layer->test("LS"));
    printf("rbm_layer trained\n");

    for(int i = 0; i < layer_num - 1; i++) {

        for(int epoch = 0; epoch < train_epochs; epoch++) {
           
           bool reset = false; 
           rbm_layers[i]->train_full(reset, i); //注意下。。
           printf("layer %d trained\n", i); 

           int hidden_size = rbm_layers[i]->M;
           printf("hidden_size: %d\n", hidden_size);
           for(int m = 0; m < hidden_size; m++) {

               double now_vb = rbm_layers[i]->vb[m];
               printf("now_vb: %lf ", now_vb);
               if( i == 0) {
                   // rbm_layer->hb[m] = rbm_layers[i]->vb[m];
                   rbm_layer->hb[m] = now_vb;
                   printf("rbm0: %lf\n ", rbm_layer->hb[m]);
               }
               else {
                   // rbm_layers[i - 1]->hb[m] = rbm_layers[i]->vb[m];
                   rbm_layers[i - 1]->hb[m] = now_vb;
                   printf("rbm: %lf\n ", rbm_layers[i - 1]->hb[m]);
               }
           }


       }
    }

    printf("after iterations...\n");
    printf("generalization RMSE: %lf\n", rbm_layer->test());
    printf("training RMSE: %lf\n", rbm_layer->test("LS"));
    // 可以用openmp并行
    char ss[1000];
    sprintf(ss, "rbm-%d", 0);
    string rbm_name = ss;
    rbm_layer->save(rbm_name);

    for(int i = 0; i < layer_num - 1; i++) {
        char ss[1000];
        sprintf(ss, "rbm-%d", i + 1);
        string rbm_name = ss;
        rbm_layers[i]->save(rbm_name);
    }


}

double test(string dataset="TS") 
{

}

double DBNCF::predict(int user, int movie)
{

}

void DBNCF::save(string filename)
{
}

string DBNCF::toString()
{
    stringstream s;

    if (!conditional) {
        s << "Basic DBN" << endl;
    } else {
        s << "Conditional DBN" << endl;
    }

    s << "---" << endl;
    s << "Epochs = " << *(int*) getParameter("epochs") << endl;
    s << "Batch size = " << *(int*) getParameter("batch_size") << endl;
    s << "CD steps = " << *(int*) getParameter("cd_steps") << endl;
    s << "Eps W. = " << *(double*) getParameter("eps_w") << endl;
    s << "Eps VB. = " << *(double*) getParameter("eps_vb") << endl;
    s << "Eps HB. = " << *(double*) getParameter("eps_hb") << endl;

    if (conditional) {
        s << "Eps D. = " << *(double*) getParameter("eps_d") << endl;
    }

    s << "Weight cost = " << *(double*) getParameter("weight_cost") << endl;
    s << "Momentum = " << *(double*) getParameter("momentum") << endl;
    s << "Annealing = " << *(bool*) getParameter("annealing") << endl;
    s << "Annealing rate = " << *(double*) getParameter("annealing_rate");

    return s.str();

}
