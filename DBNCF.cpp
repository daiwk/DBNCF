// class: DBNCF
// Author: daiwenkai
// Date: Feb 24, 2014

#include "DBNCF.h"
#include "RBMBASIC.h"
#include "Configuration.h"
#include "stdio.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>


using namespace std;

#define _ikj(i, k, j) ((i) * K * F + (k) * F + (j))
#define _ik(i, k) ((i) * K + (k))
#define _ij(i, j) ((i) * F + (j))

// 默认构造函数
DBNCF::DBNCF() : Model(CLASS_DBNCF)
{

    train_epochs = Config::DBNCF::TRAIN_EPOCHS;
    
    // 初始化input_layer
    // input_layer是正常的rbmcf，使用原来的RBM参数,隐含层节点数使用配置值
    string input_layer_name = "input_layer.rbmcf";
    RBMCF in_cf = RBMCF();
    in_cf.setParameter("F", &hidden_layer_sizes[0], sizeof(int));
    in_cf.reset();
    in_cf.save(input_layer_name);
    input_layer = new RBMCF(input_layer_name);
    
    // 初始化hidden_layers
    // K=1, N=1, M=前一层的节点数
    hidden_layer_num = Config::DBNCF::HL_NUM;

    // 有hidden_layer_num个隐层，就有hidden_layer_num - 1个RBMBASIC
    hidden_layers = new RBMBASIC*[hidden_layer_num - 1]; 
    
    int sizes = Config::DBNCF::HL_SIZE;
    
    // 使用默认构造函数，每层的节点数是一样的，从配置中读取
    for(int i = 0; i < hidden_layer_num; i++) {
    	hidden_layer_sizes[i] = sizes;
    }


    for(int i = 0; i < hidden_layer_num - 1; i++) {
        char ss[1000];
        sprintf(ss, "hidden_layer-%d.rbm", i + 1);
        string rbm_name = ss;

        RBMBASIC r = RBMBASIC();
        r.setParameter("M", &hidden_layer_sizes[i], sizeof(int));
        r.setParameter("F", &hidden_layer_sizes[i + 1], sizeof(int));
        r.reset();
        r.save(rbm_name);
        hidden_layers[i] = new RBMBASIC(rbm_name);
    }

    // 初始化output_layer
    string output_layer_name = "output_layer.rbmcf";
    RBMCF out_cf = RBMCF();
    out_cf.setParameter("F", &hidden_layer_sizes[hidden_layer_num - 1], sizeof(int));
    out_cf.reset();
    out_cf.save(output_layer_name);
    output_layer = new RBMCF(output_layer_name);

    // Default verbose and output
    setParameter("verbose", &Config::DBNCF::VERBOSE, sizeof(bool));

    ostream* log = &cout;
    setParameter("log", &log, sizeof(ostream*));

}

// 读取模型文件生成DBNCF的构造函数
DBNCF::DBNCF(string filename) : Model(CLASS_DBNCF)
{
    // 打开文件
    ifstream in(filename.c_str(), ios::in | ios::binary);
    if (in.fail()) {
        throw runtime_error("I/O exception");
    }

    // 检查ID
    char* id = new char[2];
    in.read(id, 2 * sizeof(char));
    assert(id[0] == (0x30 + __id) && id[1] == 0x0A);

    // 读取参数
    int tmp_int;

    in.read((char*) &tmp_int, sizeof (int));
    setParameter("train_epochs", &tmp_int, sizeof(int));
    train_epochs = tmp_int;

    in.read((char*) &tmp_int, sizeof (int));
    setParameter("batch_size", &tmp_int, sizeof(int));
    batch_size = tmp_int;

    // 初始化input_layer
    // input_layer是正常的rbmcf，使用原来的RBM参数,隐含层节点数使用配置值。
    string input_layer_name = "input_layer.rbmcf";
    RBMCF in_cf = RBMCF();
    in_cf.setParameter("F", &hidden_layer_sizes[0], sizeof(int));
    in_cf.reset();
    in_cf.save(input_layer_name);
    input_layer = new RBMCF(input_layer_name);

    // 初始化hidden_layers
    // K=1, N=1, M=前一层的节点数
    in.read((char*) &tmp_int, sizeof (int));
    setParameter("hidden_layer_num", &tmp_int, sizeof(int));
    hidden_layer_num = tmp_int;

    hidden_layer_sizes = new int[hidden_layer_num];
    
    in.read((char*) hidden_layer_sizes, hidden_layer_num * sizeof (int));

    // 有hidden_layer_num个隐层，就有hidden_layer_num - 1个RBMBASIC
    hidden_layers = new RBMBASIC*[hidden_layer_num - 1]; 

    for(int i = 0; i < hidden_layer_num - 1; i++) {
        char ss[1000];
        sprintf(ss, "hidden_layer-%d.rbm", i + 1);
        string rbm_name = ss;

        RBMBASIC r = RBMBASIC();
        r.setParameter("M", &hidden_layer_sizes[i], sizeof(int));
        r.setParameter("F", &hidden_layer_sizes[i + 1], sizeof(int));
        r.reset();
        r.save(rbm_name);
        hidden_layers[i] = new RBMBASIC(rbm_name);
    }

    // 初始化output_layer
    string output_layer_name = "output_layer.rbmcf";
    RBMCF out_cf = RBMCF();
    out_cf.setParameter("F", &hidden_layer_sizes[hidden_layer_num - 1], sizeof(int));
    out_cf.reset();
    out_cf.save(output_layer_name);
    output_layer = new RBMCF(output_layer_name);

    // 默认的verbose及输出重定向
    setParameter("verbose", &Config::DBNCF::VERBOSE, sizeof(bool));

    ostream* log = &cout;
    setParameter("log", &log, sizeof(ostream*));

    // 关闭文件
    in.close();
}


// 析构函数
DBNCF::~DBNCF()
{
    delete input_layer;
    delete output_layer;
    for(int i =0; i < hidden_layer_num - 1; i++) {
        delete hidden_layers[i];
    }

}

// Model的函数
void DBNCF::train(string dataset, bool reset) {
    
    // Pop parameters
    int epochs = *(int*) getParameter("epochs");
    int batch_size = *(int*) getParameter("batch_size");
    int cd_steps = *(int*) getParameter("cd_steps");
    bool verbose = *(bool*) getParameter("verbose");
    ostream* out = *(ostream**) getParameter("log");

    Dataset* LS = sets[dataset];
    Dataset* QS = sets["QS"];
    assert(LS != NULL);

//    if (conditional) {
//        assert(QS != NULL);
//        assert(LS->nb_rows == QS->nb_rows);
//    }

    for (int i = 0; i < hidden_layer_num - 1; i++) {

        for(int epoch = 0; epoch < train_epochs; epoch++) {
               
            for (int batch = 0; batch < LS->nb_rows; batch += batch_size) {
	        
		    for (int l = 0; l <= i; l++) {
		    
		        // 初始化前面每一层RBM的输入变量
		        if (l == 0) {
			    
			    // sample h from v
			    bool reset = false;
                            input_layer->train_batch(dataset, reset, batch);
		        }
		        else {
                            // sample hl from hl-1
			    bool reset = false;
			    
			    // 注意下标：此函数读rbm-h*-l，输出rbm-h*-l+1
                            hidden_layers[l - 1]->train_full(reset, l); 
                            printf("layer %d trained\n", l - 1); 
            
                            // 训练完后，要更新前一层rbm的隐含层的bias为本层的可见层的bias
			    int hidden_size = hidden_layers[l - 1]->M;
                            printf("hidden_size: %d\n", hidden_size);
                            
			    for(int m = 0; m < hidden_size; m++) {
            
                                double now_vb = hidden_layers[l]->vb[m];

				// l=1: 第一个rbmbasic，它的上一层是input_layer
                                if( l == 1) {
                                    
				    printf("pre_hb: %lf ", input_layer->hb[m]);
                                    input_layer->hb[m] = now_vb;
                                    printf("now_hb: %lf\n ", input_layer->hb[m]);
                                }
                                else {
                                    
				    hidden_layers[l - 2]->hb[m] = now_vb;
                                    printf("now_hb: %lf\n ", hidden_layers[l - 2]->hb[m]);
                                }
                            }
		        }
		    }  // End of for (int l = 0; l <= i; l++)

		    // k-cd of layer i
		    if(i == 0) {
			// input_layer->kcd, 传batch号进去;
			// do nothing
			
		    }
		    else {
			// hidden_layeri->kcd;
			// do nothing

		    }

	    }  // End of for (int batch = 0; batch < LS->nb_rows; batch += batch_size)
       }  // End of for(int epoch = 0; epoch < train_epochs; epoch++)
    }  // End of for (int l = 0; l < hidden_layer_num - 1; l++)

    printf("after iterations...\n");
    printf("generalization RMSE: %lf\n", input_layer->test());
    printf("training RMSE: %lf\n", input_layer->test("LS"));
    // 可以用openmp并行
    char ss[1000];
    sprintf(ss, "rbm-%d", 0);
    string rbm_name = ss;
    input_layer->save(rbm_name);

    for(int i = 0; i < hidden_layer_num - 1; i++) {
        char ss[1000];
        sprintf(ss, "rbm-%d", i + 1);
        string rbm_name = ss;
        hidden_layers[i]->save(rbm_name);
    }



}

// DBNCF的函数
void DBNCF::train_separate(string dataset, bool reset)
{
    
    printf("DBNCF epochs: %d\n", train_epochs);
    input_layer->train();
    printf("before iterations...\n");
    printf("RMSE: %lf\n", input_layer->test());
    printf("training RMSE: %lf\n", input_layer->test("LS"));
    printf("input_layer trained\n");

    for(int i = 0; i < hidden_layer_num - 1; i++) {

        for(int epoch = 0; epoch < train_epochs; epoch++) {
           
           bool reset = false; 
           hidden_layers[i]->train_full(reset, i); //注意下。。
           printf("layer %d trained\n", i); 

           int hidden_size = hidden_layers[i]->M;
           printf("hidden_size: %d\n", hidden_size);
           for(int m = 0; m < hidden_size; m++) {

               double now_vb = hidden_layers[i]->vb[m];
               printf("now_vb: %lf ", now_vb);
               if( i == 0) {
                   // input_layer->hb[m] = hidden_layers[i]->vb[m];
                   input_layer->hb[m] = now_vb;
                   printf("rbm0: %lf\n ", input_layer->hb[m]);
               }
               else {
                   // hidden_layers[i - 1]->hb[m] = hidden_layers[i]->vb[m];
                   hidden_layers[i - 1]->hb[m] = now_vb;
                   printf("rbm: %lf\n ", hidden_layers[i - 1]->hb[m]);
               }
           }


       }
    }

    printf("after iterations...\n");
    printf("generalization RMSE: %lf\n", input_layer->test());
    printf("training RMSE: %lf\n", input_layer->test("LS"));
    // 可以用openmp并行
    char ss[1000];
    sprintf(ss, "rbm-%d", 0);
    string rbm_name = ss;
    input_layer->save(rbm_name);

    for(int i = 0; i < hidden_layer_num - 1; i++) {
        char ss[1000];
        sprintf(ss, "rbm-%d", i + 1);
        string rbm_name = ss;
        hidden_layers[i]->save(rbm_name);
    }

}

double DBNCF::test(string dataset) 
{
//    // Pop LS, QS and TS
//    Dataset* LS = sets["LS"];
//    Dataset* QS = sets["QS"];
//    Dataset* TS = sets[dataset];
//    assert(LS != NULL);
//    assert(TS != NULL);
//    assert(LS->nb_rows == TS->nb_rows);
//
//    if (conditional) {
//        assert(QS != NULL);
//        assert(LS->nb_rows == QS->nb_rows);
//    }
//    
//    // Start calculating the running time
//    struct timeval start;
//    struct timeval end;
//    unsigned long usec;
//    gettimeofday(&start, NULL);
//
//    // Allocate local data structures
//    double* vs = new double[M * K];
//    double* vp = new double[M * K];
//    double* hs = new double[F];
//    double* hp = new double[F];
//
//
//    for(int i = 0; i < F; i++)
//        printf("testing hb[%d]: %lf\n", i, hb[i]);
//    // Initialization
//    double total_error = 0.;
//    int count = 0;
//
//    // Loop through users in the test set
//    for (int n = 0; n < TS->nb_rows; n++) {
//        if (TS->count[n] == 0) {
//            continue;
//        }
//
//        // Set user n data on the visible units
//        for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
//            int i = LS->ids[m];
//            int ik_0 = _ik(i, 0);
//
//            for (int k = 0; k < K; k++) {
//                vs[ik_0 + k] = 0.;
//            }
//
//            vs[ik_0 + LS->ratings[m] - 1] = 1.;
//        }
//
//        // Compute ^p = p(h | V, d) into hp
//        if (!conditional) {
//            update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], hp);
//        } else {
//            update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], &QS->ids[QS->index[n]], QS->count[n], hp);
//        }
//
//        // Compute p(v_ik = 1 | ^p) for all movie i in TS
//        update_visible(hp, vp, &TS->ids[TS->index[n]], TS->count[n]);
//
//        // Predict ratings
//        for (int m = TS->index[n]; m < TS->index[n] + TS->count[n]; m++) {
//            int i = TS->ids[m];
//            int ik_0 = _ik(i, 0);
//            double prediction = 0.;
//
//            for (int k = 0; k < K; k++) {
//                prediction += vp[ik_0 + k] * (k + 1);
//                // cout << "ik_0+k: " << ik_0 + k <<" vp[ik_0 + k]:" << vp[ik_0 + k] << endl;
//            }
//
//            double error = prediction - TS->ratings[m];
//            // cout << "error: " << error << " prediction: " << prediction << " rating: " << TS->ratings[m] << " ik_0:" << ik_0 << " upbound: " << K*M;
//	    // cout << " n: " << n << " ids: " << i << " count: " << count << endl;
//            total_error += error * error;
//            count++;
//        }
//    }
//
////    // Deallocate data structure
////    if (vs != NULL) { 
////        delete[] vs; 
////        vs = NULL; 
////    }
////    if (vp != NULL) { delete[] vp; vp = NULL; }
////    if (hs != NULL) { delete[] hs; hs = NULL; }
////    if (hp != NULL) { delete[] hp; hp = NULL; }
//
//    // cout << "total_error: " << total_error << " count: " << count << endl;
//    
//    // print running time
//    gettimeofday(&end, NULL);
//    usec = 1000000 * (end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
//
////    cout << "File: " << __FILE__ << ", Function: " << __FUNCTION__  << ", Line: " << __LINE__ << endl;
//    printf( "File: %s, Function: %s, Line: %d\n", __FILE__, __FUNCTION__, __LINE__);
////    cout << "Time of test(): " << usec << " usec[" << usec / 1000000. <<" sec]." << endl;
//    printf("Time of test(): %ld usec[ %lf sec].", usec, usec / 1000000.);
//    return sqrt(total_error / count);
return 1;
}

double DBNCF::predict(int user, int movie)
{
//    // Pop LS
//    Dataset* LS = sets["LS"];
//    Dataset* QS = sets["QS"];
//    assert(LS != NULL);
//
//    if (conditional) {
//        assert(QS != NULL);
//        assert(LS->nb_rows == QS->nb_rows);
//    }
//
//    // Asserts
//    assert(user >= 0);
//    assert(user < N);
//    assert(movie >= 0);
//    assert(movie < M);
//
//    // Reject if user is unknown
//    if (!LS->contains_user(user)) {
//        return -1.0;
//    }
//
//    /*
//    if (LS->count[user] <= 0) {
//        cout << "unknown user 2" << endl;
//        return -1.0;
//    }
//    */
//
//    // Reject if movie is unknown
//    if (!LS->contains_movie(movie)){
//        return -1.0;
//    }
//
//    // Allocate local data structures
//    double* vs = new double[M * K];
//    double* vp = new double[M * K];
//    double* hs = new double[F];
//    double* hp = new double[F];
//
//    // Set user data on the visible units
//    int n = LS->users[user];
//
//    for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
//        int i = LS->ids[m];
//        int ik_0 = _ik(i, 0);
//
//        for (int k = 0; k < K; k++) {
//            vs[ik_0 + k] = 0.;
//        }
//
//        vs[ik_0 + LS->ratings[m] - 1] = 1.;
//    }
//
//    // Compute ^p = p(h | V, d) into hp
//    if (!conditional) {
//        update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], hp);
//    } else {
//        update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], &QS->ids[QS->index[n]], QS->count[n], hp);
//    }
//
//    // Compute p(v_ik = 1 | ^p) for i = movie
//    update_visible(hp, vp, &movie, 1);
//
//    // Predict rating
//    double prediction = 0.;
//    int ik_0 = _ik(movie, 0);
//
//    for (int k = 0; k < K; k++) {
//        prediction += vp[ik_0 + k] * (k + 1);
//    }
//
//    // Deallocate data structure
//    delete[] vs;
//    delete[] vp;
//    delete[] hs;
//    delete[] hp;
//
//    return prediction;
return 1;

}

void DBNCF::save(string filename)
{
    // Open file
    ofstream out(filename.c_str(), ios::out | ios::binary);

    if (out.fail()) {
        throw runtime_error("I/O exception!");
    }

    // Write class ID
    char id[2] = {0x30 + __id, 0x0A};
    out.write(id, 2 * sizeof (char));

    // Write parameters
    // 等价于out.write((char*) getParameter("train_epochs"), sizeof (int));
    out.write((char*) train_epochs, sizeof (int));
    
    // 等价于out.write((char*) getParameter("batch_size"), sizeof (int));
    out.write((char*) batch_size, sizeof (int));
    
    // 等价于out.write((char*) getParameter("hidden_layer_num"), sizeof(int));
    out.write((char*) hidden_layer_num, sizeof(int));
    
    out.write((char*) hidden_layer_sizes, hidden_layer_num * sizeof (int));

    out.close();

}

string DBNCF::toString()
{
    stringstream s;

    s << "---" << endl;
    s << "Train Epochs = " << *(int*) getParameter("train_epochs") << endl;
    s << "Batch size = " << *(int*) getParameter("batch_size") << endl;
    s << "Hidden layer num = " << *(int*) getParameter("hidden_layer_num") << endl;
    
    for(int i = 0; i < hidden_layer_num; i++) {
    	s << "Hidden layer" << i << " size: " << *(int*) getParameter("hidden_layer_num") << endl;
    }

    return s.str();

}


