// class: AHRBMCF
// Author: daiwenkai
// Date: Feb 24, 2014

#include "AHRBMCF.h"
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

#include "Misc.h"
#include <omp.h>

using namespace std;

#define _ikj(i, k, j) ((i) * K * F + (k) * F + (j))
#define _ik(i, k) ((i) * K + (k))
#define _ij(i, j) ((i) * F + (j))

// 默认构造函数
AHRBMCF::AHRBMCF() : Model(CLASS_AHRBMCF)
{

	train_epochs = Config::AHRBMCF::TRAIN_EPOCHS;
	setParameter("train_epochs", &train_epochs, sizeof(int));
	batch_size = Config::AHRBMCF::BATCH_SIZE;
	setParameter("batch_size", &batch_size, sizeof(int));


	// 初始化hidden_layers
	// K=1, N=1, M=前一层的节点数
	hidden_layer_num = Config::AHRBMCF::HL_NUM;
	setParameter("hidden_layer_num", &hidden_layer_num, sizeof(int));

	// 有hidden_layer_num个隐层，就有hidden_layer_num - 1个RBMBASIC
	hidden_layers = new RBMBASIC*[hidden_layer_num - 1]; 

	int sizes = Config::AHRBMCF::HL_SIZE;
	setParameter("hidden_layer_size", &sizes, sizeof(int));

	hidden_layer_sizes = new int[hidden_layer_num];
	// 使用默认构造函数，每层的节点数是一样的，从配置中读取
	for(int i = 0; i < hidden_layer_num; i++) {
		hidden_layer_sizes[i] = sizes;
	}

	// 初始化input_layer
	// input_layer是正常的rbmcf，使用原来的RBM参数,隐含层节点数使用配置值
	string input_layer_name = "input_layer.rbmcf";
	RBMCF in_cf = RBMCF();
	in_cf.setParameter("F", &hidden_layer_sizes[0], sizeof(int));
	in_cf.reset();
	in_cf.save(input_layer_name);
	input_layer = new RBMCF(input_layer_name);

	for(int i = 0; i < hidden_layer_num - 1; i++) {
		char ss[1000];
		sprintf(ss, "hidden_layer-%d.rbm", i);
		string rbm_name = ss;

		RBMBASIC r = RBMBASIC();
		r.setParameter("M", &hidden_layer_sizes[i], sizeof(int));
		r.setParameter("F", &hidden_layer_sizes[i + 1], sizeof(int));
		r.setParameter("N", &Config::RBMBASIC::N, sizeof(int));
		r.setParameter("K", &Config::RBMBASIC::K, sizeof(int));
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
	setParameter("verbose", &Config::AHRBMCF::VERBOSE, sizeof(bool));

	ostream* log = &cout;
	setParameter("log", &log, sizeof(ostream*));

}

// 读取模型文件生成AHRBMCF的构造函数
AHRBMCF::AHRBMCF(string filename) : Model(CLASS_AHRBMCF)
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

	// 可能每个隐层的节点数不一样，不过这里暂时假设是一样的，所以setPara的时候暂时只取第一个元素来代表全部
	setParameter("hidden_layer_size", &hidden_layer_sizes[0], sizeof(int));

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
	setParameter("verbose", &Config::AHRBMCF::VERBOSE, sizeof(bool));

	ostream* log = &cout;
	setParameter("log", &log, sizeof(ostream*));

	// 关闭文件
	in.close();
}


// 析构函数
AHRBMCF::~AHRBMCF()
{
	delete input_layer;
	delete output_layer;
	for(int i =0; i < hidden_layer_num - 1; i++) {
		delete hidden_layers[i];
	}

}

// Model的函数
void AHRBMCF::train(string dataset, bool reset) 
{
	pretrain(dataset, reset);
    printf("####after training, use full network to predict...\n");
	printf("generalization RMSE: %lf\n", test("TS"));
	printf("training RMSE: %lf\n\n", test("LS"));

    finetune();
	printf("####after finetune, use full network to predict...\n");
	printf("generalization RMSE: %lf\n", test());
	printf("training RMSE: %lf\n\n", test("LS"));

	//printf("####after finetune, only use output layer to predict..\n");
	//printf("generalization RMSE: %lf\n", dbncf->output_layer->test());
	//printf("training RMSE: %lf\n\n", dbncf->output_layer->test("LS"));
	// dbncf->finetune("LS");


}



void AHRBMCF::pretrain_old_version(string dataset, bool reset) 
{

	// Pop parameters
	int batch_size = *(int*) getParameter("batch_size");
	printf("batch_size: %d\n", batch_size);
	printf("train_epochs: %d\n", train_epochs);
	bool verbose = *(bool*) getParameter("verbose");
	ostream* out = *(ostream**) getParameter("log");

	Dataset* LS = sets[dataset];

	cout<<"LS->nb_rows:"<<LS->nb_rows<<endl;
	Dataset* QS = sets["QS"];
	Dataset* TS = sets["TS"];
	Dataset* VS = sets["VS"];
	assert(LS != NULL);

	//    if (conditional) {
	//        assert(QS != NULL);
	//        assert(LS->nb_rows == QS->nb_rows);
	//    }

	input_layer->addSet(dataset, LS);
	input_layer->addSet("QS", QS);
	input_layer->addSet("VS", VS);
	input_layer->addSet("TS", TS);

	// just test...
	//    input_layer->train();

	output_layer->addSet(dataset, LS);
	output_layer->addSet("QS", QS);
	output_layer->addSet("VS", VS);
	output_layer->addSet("TS", TS);

	*out <<"EPOch\tgen-RMSE\ttrain-RMSE\tinput\\full\\out\n";
	out->flush();

    #pragma omp parallel
	{
	for (int i = 0; i < hidden_layer_num; i++) {

		for(int epoch = 0; epoch < train_epochs; epoch++) {

			#pragma omp for schedule(guided)
			for (int batch = 0; batch < LS->nb_rows; batch += batch_size) {

				printf("\nbatch: %d,\n", batch);
				for (int l = 0; l <= i; l++) {

					// 初始化前面每一层RBM的输入变量
					if (l == 0) {

						// sample h from v
						bool reset = false;
						input_layer->train_batch(dataset, reset, batch);

						//将训练出来的结果给下一层用【train_full里一开始时候就有一个read的操作了,读前一个rbm的hb和vb】
					}
					else {
						// sample hl from hl-1
						bool reset = false;

						// 注意下标：此函数读rbm-h*-l，输出rbm-h*-l+1
						hidden_layers[l - 1]->train_full(reset, l - 1); 
						printf("layer %d trained\n", l - 1); 

						// 注意：在train_batch函数中实现了：
						// 首先读前一层的hs, hb到vs, vb中；然后训练；最后存本层hs,hb到文件中
						// 所以，训练后不用往后考虑，只要往前考虑：
						// 训练完后，要更新前一层rbm的隐含层的bias为本层的可见层的bias
						int hidden_size = hidden_layers[l - 1]->M;
						printf("hidden_size: %d\n", hidden_size);
					    
						for(int m = 0; m < hidden_size; m++) {

							double now_vb = hidden_layers[l - 1]->vb[m];

							// l=1: 第一个rbmbasic，它的上一层是input_layer
							if( l == 1) {

//								printf("pre_hb: %lf ", input_layer->hb[m]);
//#pragma omp critical
								input_layer->hb[m] = now_vb;
//								printf("now_hb: %lf\n ", input_layer->hb[m]);
							}
							else {

//#pragma omp critical
								hidden_layers[l - 2]->hb[m] = now_vb;
//								printf("now_hb: %lf\n ", hidden_layers[l - 2]->hb[m]);
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
		    
			#pragma omp single
			{
			cout << "calc input rmse...\n";
			char rmse_input[1000];
			sprintf(rmse_input, "%d\t%lf\t%lf\tinput\n", epoch, input_layer->test("TS"), input_layer->test("LS"));
			*out << rmse_input;

			cout << "calc full rmse...\n";
		    char rmse_full[1000];
			sprintf(rmse_full, "%d\t%lf\t%lf\tfull\n", epoch, test("TS"), test("LS"));
			*out << rmse_full;
		    
			cout << "calc output rmse...\n";
			char rmse_output[1000];
			sprintf(rmse_output, "%d\t%lf\t%lf\toutput\n", epoch, output_layer->test("TS"), output_layer->test("LS"));
			*out << rmse_output;
			
			out->flush();
			}
		}  // End of for(int epoch = 0; epoch < train_epochs; epoch++)
	}  // End of for (int l = 0; l < hidden_layer_num - 1; l++)

	} //End of #pragma omp parallel

	printf("\n####after pretrain, only use input layer to predict...\n");
//	#pragma omp single
	printf("generalization RMSE: %lf\n", input_layer->test());
	printf("training RMSE: %lf\n", input_layer->test("LS"));

	// 训练完后，要更新output_layer的隐藏层参数为最后一个隐藏层的参数
	// （由于output_layer其实是一个倒置的CRBM）
	// 即他的h就是最后一个隐层的h

	int output_hidden_size = hidden_layers[hidden_layer_num - 2]->F;
	printf("output_hidden_size of last hidden: %d\n", output_hidden_size);

	// 将最后一个隐层的bias赋值给output的隐层的bias
	for(int m = 0; m < output_hidden_size; m++) {

		double now_hb = hidden_layers[hidden_layer_num - 2]->hb[m];

		output_layer->hb[m] = now_hb;
		// printf("output now_hb: %lf\n ", output_layer->hb[m]);
	}

    for (int batch = 0; batch < LS->nb_rows; batch += batch_size) {
		bool reset = false;
		output_layer->train_batch(dataset, reset, batch);
	}

	// 将output的隐层的bias赋值给最后一个隐层
	for(int m = 0; m < output_hidden_size; m++) {

		double now_hb = output_layer->hb[m];

    	hidden_layers[hidden_layer_num - 2]->hb[m] = now_hb;
		// printf("output now_hb: %lf\n ", output_layer->hb[m]);
	}

	printf("\n####after pretrain, only use output layer to predict..\n");
	printf("generalization RMSE: %lf\n", output_layer->test());
	printf("training RMSE: %lf\n\n", output_layer->test("LS"));
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

void AHRBMCF::pretrain(string dataset, bool reset)
{

	// Pop parameters
	int batch_size = *(int*) getParameter("batch_size");
	printf("batch_size: %d\n", batch_size);
	printf("train_epochs: %d\n", train_epochs);
	bool verbose = *(bool*) getParameter("verbose");
	ostream* out = *(ostream**) getParameter("log");

	Dataset* LS = sets[dataset];
	Dataset* QS = sets["QS"];
	Dataset* TS = sets["TS"];
	Dataset* VS = sets["VS"];
	assert(LS != NULL);

	//    if (conditional) {
	//        assert(QS != NULL);
	//        assert(LS->nb_rows == QS->nb_rows);
	//    }

	input_layer->addSet(dataset, LS);
	input_layer->addSet("QS", QS);
	input_layer->addSet("VS", VS);
	input_layer->addSet("TS", TS);

	output_layer->addSet(dataset, LS);
	output_layer->addSet("QS", QS);
	output_layer->addSet("VS", VS);
	output_layer->addSet("TS", TS);

	*out <<"EPOch\tgen-RMSE\ttrain-RMSE\tinput\\full\\out\n";
	out->flush();


    // Initialization
    double total_error = 0.;
    int count = 0;

    printf("starting test...\n");
	//    for (int i = 0; i < hidden_layer_num; i++) {

	int F = input_layer->F;
	int M = input_layer->M;
	int K = input_layer->K;
	double* input_vs = new double[M * K];
	double* input_vp = new double[M * K];
	double* input_hs = new double[F];
	double* input_hp = new double[F];

	double* input_w_acc = new double[M * K * F];
	int* input_w_count = new int[M * K * F];
	double* input_vb_acc = new double[M * K];
	int* input_vb_count = new int[M * K];
	double* input_hb_acc = new double[F];

	// 只有input需要watched
	bool* input_watched = NULL;
	//    if (conditional) {
	input_watched = new bool[M];
	//    }

	double** hidden_vs = new double* [hidden_layer_num - 1];
	double** hidden_vp = new double* [hidden_layer_num - 1];
	double** hidden_hs = new double* [hidden_layer_num - 1];
	double** hidden_hp = new double* [hidden_layer_num - 1];


	for (int l = 0; l < hidden_layer_num - 1; l++) {

		int hidden_M = hidden_layers[l]->M;
		int hidden_K = hidden_layers[l]->K;
		int hidden_F = hidden_layers[l]->F;
	
		hidden_vs[l] = new double[hidden_M * hidden_K];
		hidden_vp[l] = new double[hidden_M * hidden_K];
		hidden_hs[l] = new double[hidden_F];
		hidden_hp[l] = new double[hidden_F];

		int* mask_visible = new int[hidden_M];
	}


	int output_F = output_layer->F;
	int output_M = output_layer->M;
	int output_K = output_layer->K;
	double* output_vs = new double[output_M * output_K];
	double* output_vp = new double[output_M * output_K];
	double* output_hs = new double[output_F];
	double* output_hp = new double[output_F];

    // Start calculating the running time
    struct timeval start;
    struct timeval end;
    unsigned long usec;
    gettimeofday(&start, NULL);

    #pragma omp parallel
	{
	for (int i = 0; i < hidden_layer_num; i++) {

		for(int epoch = 0; epoch < train_epochs; epoch++) {
	
	#pragma omp parallel for
			for (int batch = 0; batch < LS->nb_rows; batch += batch_size) {
	
				// 每个batch,首先赋值给input的vs，然后得到hp，再得到hs
				for (int n = batch; n < min(batch + batch_size, LS->nb_rows); n++) {
	
	
					zero(input_w_acc, M * K * F);
					zero(input_w_count, M * K * F);
					zero(input_vb_acc, M * K);
					zero(input_vb_count, M * K);
					zero(input_hb_acc, F);
	
					//    if (conditional) {
					zero(input_watched, M);
					//    }
	
	
					// Set user n data on the visible units
					for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
						int i = LS->ids[m];
						int ik_0 = i * K; // _ik(i, 0);
	
						for (int k = 0; k < K; k++) {
							input_vs[ik_0 + k] = 0.;
						}
	
						input_vs[ik_0 + LS->ratings[m] - 1] = 1.;
					}
	
					// Compute ^p = p(h | V, d) into hp
					
					input_layer->update_hidden(input_vs, &LS->ids[LS->index[n]], LS->count[n], &QS->ids[QS->index[n]], QS->count[n], input_hp);
					input_layer->sample_hidden(input_hp, input_hs);
					// Deallocate data structures
				} // End of for (int n = batch; n < min(batch + batch_size, LS->nb_rows); n++)
	
				// 这里做前馈传导，用vp去传。。
				// 最后一层hp传给输出层，用sigmoid算最终的hp。。
	
				for (int l = 0; l < hidden_layer_num - 1; l++) {
	
	
					int hidden_M = hidden_layers[l]->M;
					int hidden_K = hidden_layers[l]->K;
					int hidden_F = hidden_layers[l]->F;
	
	//				hidden_vs[l] = new double[hidden_M * hidden_K];
	//				hidden_vp[l] = new double[hidden_M * hidden_K];
	//				hidden_hs[l] = new double[hidden_F];
	//				hidden_hp[l] = new double[hidden_F];
	
					for (int i = 0; i < hidden_M; i++) {
						if (l == 0) {
							hidden_vs[l][i] = input_hs[i];
						}
						else {
							hidden_vs[l][i] = hidden_hs[l - 1][i];
						}
					}
	
					int* mask_visible = new int[hidden_M];
					for(int ind = 0; ind < hidden_M; ind++) {
						mask_visible[ind] = ind;
					}
	
	
					hidden_layers[l]->update_hidden_p(hidden_vs[l], &mask_visible[0], hidden_M, hidden_hp[l]);
					hidden_layers[l]->sample_hidden(hidden_hp[l], hidden_hs[l]);
					
					delete [] mask_visible;
	
				}  // End of for (int l = 0; l < hidden_layer_num - 1; l++)
				
				int output_F = hidden_layers[hidden_layer_num - 2]->F;
				int output_M = output_layer->M;
				
				int* mask_hidden= new int[output_F];
				for(int ind = 0; ind < output_F; ind++) {
					mask_hidden[ind] = ind;
				}
			
				int* mask_visible = new int[output_M];
				for(int ind = 0; ind < output_M; ind++) {
					mask_visible[ind] = ind;
				}
	
				// output:h->v,update visible,算vp的期望
	
				output_layer->update_visible(hidden_hs[hidden_layer_num - 2], output_vp, &mask_hidden[0], output_F);
	
	            delete [] mask_hidden;
				delete [] mask_visible;
	
				// 开始计算残差，准备bp
	
				for (int n = 0; n < min(batch + batch_size, LS->nb_rows); n++) {
					
					// 只更新和这个user有关的那几个节点，每个节点k维
					for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
		
						int i = LS->ids[m];
						int ik_0 = _ik(i, 0);
						double prediction = 0.;
						double *delta_output = new double[output_M * output_K];
	
						for (int k = 0; k < K; k++) {
							
							prediction += output_vp[ik_0 + k] * (k + 1);
							double error = prediction - LS->ratings[m];
	//						cout << "error: " << error << " prediction: " << prediction << " rating: " << TS->ratings[m] << " ik_0:" << ik_0 << " upbound: " << K*M <<endl;
							// cout << " n: " << n << " ids: " << i << " count: " << count << endl;
							
							total_error += error * error;
							count++;
	
						}
	
						delete [] delta_output;
	
					} //遍历完一个batch的所有user
					
					// 开始往回走，更新w和b,但是，d呢？
	
				}
	
	
			}  // End of for (int batch = 0; batch < LS->nb_rows; batch += batch_size)
		}  // End of for(int epoch = 0; epoch < train_epochs; epoch++)
	}  // End of for (int l = 0; l < hidden_layer_num - 1; l++)

	}  // End of omp parallel 
	
	if (input_vs != NULL) delete[] input_vs;
	if (input_vp != NULL) delete[] input_vp;
	if (input_hs != NULL) delete[] input_hs;
	if (input_hp != NULL) delete[] input_hp;

	if (input_w_acc != NULL) delete[] input_w_acc;
	if (input_w_count != NULL) delete[] input_w_count;
	if (input_vb_acc != NULL) delete[] input_vb_acc;
	if (input_vb_count != NULL) delete[] input_vb_count;
	if (input_hb_acc != NULL) delete[] input_hb_acc;
	if (input_watched != NULL) delete[] input_watched;

    for(int i = 0; i < hidden_layer_num - 1; i++) {
		delete hidden_vs[i];
		delete hidden_vp[i];
		delete hidden_hs[i];
		delete hidden_hp[i];
	}

	if (output_vs != NULL) delete[] output_vs;
	if (output_vp != NULL) delete[] output_vp;
	if (output_hs != NULL) delete[] output_hs;
	if (output_hp != NULL) delete[] output_hp;

    // print running time
    gettimeofday(&end, NULL);
    usec = 1000000 * (end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;

    printf( "File: %s, Function: %s, Line: %d\n", __FILE__, __FUNCTION__, __LINE__);
	printf("Time of pretrain(): %ld usec[ %lf sec].\n", usec, usec / 1000000.);
 
}



void AHRBMCF::finetune(string dataset)
{

	Dataset* LS = sets[dataset];
	Dataset* QS = sets["QS"];
	Dataset* TS = sets["TS"];
	Dataset* VS = sets["VS"];
	assert(LS != NULL);

	//    if (conditional) {
	//        assert(QS != NULL);
	//        assert(LS->nb_rows == QS->nb_rows);
	//    }

	input_layer->addSet(dataset, LS);
	input_layer->addSet("QS", QS);
	input_layer->addSet("VS", VS);
	input_layer->addSet("TS", TS);

	//    for (int i = 0; i < hidden_layer_num; i++) {

	int F = input_layer->F;
	int M = input_layer->M;
	int K = input_layer->K;
	double* input_vs = new double[M * K];
	double* input_vp = new double[M * K];
	double* input_hs = new double[F];
	double* input_hp = new double[F];

	double* input_w_acc = new double[M * K * F];
	int* input_w_count = new int[M * K * F];
	double* input_vb_acc = new double[M * K];
	int* input_vb_count = new int[M * K];
	double* input_hb_acc = new double[F];

	// 只有input需要watched
	bool* input_watched = NULL;
	//    if (conditional) {
	input_watched = new bool[M];
	//    }

	double** hidden_vs = new double* [hidden_layer_num - 1];
	double** hidden_vp = new double* [hidden_layer_num - 1];
	double** hidden_hs = new double* [hidden_layer_num - 1];
	double** hidden_hp = new double* [hidden_layer_num - 1];

	for (int l = 0; l < hidden_layer_num - 1; l++) {

		int hidden_M = hidden_layers[l]->M;
		int hidden_K = hidden_layers[l]->K;
		int hidden_F = hidden_layers[l]->F;
	
		hidden_vs[l] = new double[hidden_M * hidden_K];
		hidden_vp[l] = new double[hidden_M * hidden_K];
		hidden_hs[l] = new double[hidden_F];
		hidden_hp[l] = new double[hidden_F];

	}


	int output_F = output_layer->F;
	int output_M = output_layer->M;
	int output_K = output_layer->K;
	double* output_vs = new double[output_M * output_K];
	double* output_vp = new double[output_M * output_K];
	double* output_hs = new double[output_F];
	double* output_hp = new double[output_F];

    // 供BP回去的时候使用
	// BP更新输出层和最后一个隐含层间的权重
	// BP只是回到第一个隐含层，没有更新隐含层和可见层之间的权重，也没更新可见层的bias和d
	// 也就是说BP只更新从hidden_layers[0] - hidden_layers[num - 2] 这num - 1个隐含层的bias，还有他们之间的权重
    // 存放残差
	double** delta = new double* [hidden_layer_num + 1];
	double** w_acc = new double* [hidden_layer_num];
	double** b_acc = new double* [hidden_layer_num];

	for (int l = 0; l < hidden_layer_num - 1; l++) {

		int hidden_M = hidden_layers[l]->M;
		int hidden_K = hidden_layers[l]->K;
		int hidden_F = hidden_layers[l]->F;
	
		delta[l] = new double[hidden_M];
		w_acc[l] = new double[hidden_F * hidden_M];
		b_acc[l] = new double[hidden_F];

	}

	delta[hidden_layer_num - 1] = new double[output_layer->M];  
    w_acc[hidden_layer_num - 1] = new double[output_layer->M];  
	b_acc[hidden_layer_num - 1] = new double[output_layer->M];  
	delta[hidden_layer_num] = new double[output_layer->F];  

	for(int epoch = 0; epoch < train_epochs; epoch++) {

#pragma omp parallel for
		for (int batch = 0; batch < LS->nb_rows; batch += batch_size) {

			// 每个batch,首先赋值给input的vs，然后得到hp，再得到hs
			for (int n = batch; n < min(batch + batch_size, LS->nb_rows); n++) {

				zero(input_w_acc, M * K * F);
				zero(input_w_count, M * K * F);
				zero(input_vb_acc, M * K);
				zero(input_vb_count, M * K);
				zero(input_hb_acc, F);

				//    if (conditional) {
				zero(input_watched, M);
				//    }

				// Set user n data on the visible units
				for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
					int i = LS->ids[m];
					int ik_0 = i * K; // _ik(i, 0);

//#pragma omp critical
					for (int k = 0; k < K; k++) {
						input_vs[ik_0 + k] = 0.;
					}

//#pragma omp critical
					input_vs[ik_0 + LS->ratings[m] - 1] = 1.;
				}

				// Compute ^p = p(h | V, d) into hp
//#pragma omp critical				
				input_layer->update_hidden(input_vs, &LS->ids[LS->index[n]], LS->count[n], &QS->ids[QS->index[n]], QS->count[n], input_hp);
//#pragma omp critical				
				input_layer->sample_hidden(input_hp, input_hs);
				// Deallocate data structures
			} // End of for (int n = batch; n < min(batch + batch_size, LS->nb_rows); n++)

			// 这里做前馈传导，用vp去传。。
			// 最后一层hp传给输出层，用sigmoid算最终的hp。。

			for (int l = 0; l < hidden_layer_num - 1; l++) {

				int hidden_M = hidden_layers[l]->M;
				int hidden_K = hidden_layers[l]->K;
				int hidden_F = hidden_layers[l]->F;

				for (int i = 0; i < hidden_M; i++) {
					if (l == 0) {
//#pragma omp critical
						hidden_vs[l][i] = input_hs[i];
					}
					else {
//#pragma omp critical
						hidden_vs[l][i] = hidden_hs[l - 1][i];
					}
				}

				int* mask_visible = new int[hidden_M];
				for(int ind = 0; ind < hidden_M; ind++) {
					mask_visible[ind] = ind;
				}


//#pragma omp critical
				hidden_layers[l]->update_hidden_p(hidden_vs[l], &mask_visible[0], hidden_M, hidden_hp[l]);
//#pragma omp critical
				hidden_layers[l]->sample_hidden(hidden_hp[l], hidden_hs[l]);


			}  // End of for (int l = 0; l < hidden_layer_num - 1; l++)
			
			int output_F = hidden_layers[hidden_layer_num - 2]->F;
			int output_M = output_layer->M;
			
			int* mask_hidden= new int[output_F];
			for(int ind = 0; ind < output_F; ind++) {
				mask_hidden[ind] = ind;
			}
		
			int* mask_visible = new int[output_M];
			for(int ind = 0; ind < output_M; ind++) {
				mask_visible[ind] = ind;
			}

			// output:h->v,update visible,算vp的期望

//#pragma omp critical
			output_layer->update_visible(hidden_hs[hidden_layer_num - 2], output_vp, &mask_hidden[0], output_F);


			// 开始计算残差，准备bp

			for (int n = 0; n < min(batch + batch_size, LS->nb_rows); n++) {
				
				// 只更新和这个user有关的那几个节点，每个节点k维
				for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
	
					int i = LS->ids[m];
					int ik_0 = _ik(i, 0);
					// double prediction = 0.;

					for (int k = 0; k < K; k++) {
						// prediction += output_vp[ik_0 + k] * (k + 1);
						//
						double p_out = output_vp[ik_0 + k];
////						double fz = hidden_hp[hidden_layer_num - 2][ik_0];  // 最后一个隐层每个节点是一维的
////						delta_output[ik_0 + k] = (-1) * output_layer->w (input_vs[ik_0 + k] - pred_i) * pred_i * (1 - pred_i);

					}

				} //遍历完一个batch的所有user
				
				// 开始往回走，更新w和b,但是，d呢？

			}


		}  // End of for (int batch = 0; batch < LS->nb_rows; batch += batch_size)
	}  // End of for(int epoch = 0; epoch < train_epochs; epoch++)
	//    }  // End of for (int l = 0; l < hidden_layer_num - 1; l++)
	
	if (input_vs != NULL) delete[] input_vs;
	if (input_vp != NULL) delete[] input_vp;
	if (input_hs != NULL) delete[] input_hs;
	if (input_hp != NULL) delete[] input_hp;

	if (input_w_acc != NULL) delete[] input_w_acc;
	if (input_w_count != NULL) delete[] input_w_count;
	if (input_vb_acc != NULL) delete[] input_vb_acc;
	if (input_vb_count != NULL) delete[] input_vb_count;
	if (input_hb_acc != NULL) delete[] input_hb_acc;
	if (input_watched != NULL) delete[] input_watched;

    for(int i = 0; i < hidden_layer_num - 1; i++) {
		delete hidden_vs[i];
		delete hidden_vp[i];
		delete hidden_hs[i];
		delete hidden_hp[i];
	}

	if (output_vs != NULL) delete[] output_vs;
	if (output_vp != NULL) delete[] output_vp;
	if (output_hs != NULL) delete[] output_hs;
	if (output_hp != NULL) delete[] output_hp;
}


double AHRBMCF::test(string dataset)
{

	Dataset* LS = sets[dataset];
	Dataset* QS = sets["QS"];
	Dataset* TS = sets["TS"];
	Dataset* VS = sets["VS"];
	assert(LS != NULL);

	//    if (conditional) {
	//        assert(QS != NULL);
	//        assert(LS->nb_rows == QS->nb_rows);
	//    }

	input_layer->addSet(dataset, LS);
	input_layer->addSet("QS", QS);
	input_layer->addSet("VS", VS);
	input_layer->addSet("TS", TS);

    // Initialization
    double total_error = 0.;
    int count = 0;

    printf("starting test...\n");
	//    for (int i = 0; i < hidden_layer_num; i++) {

	int F = input_layer->F;
	int M = input_layer->M;
	int K = input_layer->K;
	double* input_vs = new double[M * K];
	double* input_vp = new double[M * K];
	double* input_hs = new double[F];
	double* input_hp = new double[F];

	double* input_w_acc = new double[M * K * F];
	int* input_w_count = new int[M * K * F];
	double* input_vb_acc = new double[M * K];
	int* input_vb_count = new int[M * K];
	double* input_hb_acc = new double[F];

	// 只有input需要watched
	bool* input_watched = NULL;
	//    if (conditional) {
	input_watched = new bool[M];
	//    }

	double** hidden_vs = new double* [hidden_layer_num - 1];
	double** hidden_vp = new double* [hidden_layer_num - 1];
	double** hidden_hs = new double* [hidden_layer_num - 1];
	double** hidden_hp = new double* [hidden_layer_num - 1];


	for (int l = 0; l < hidden_layer_num - 1; l++) {

		int hidden_M = hidden_layers[l]->M;
		int hidden_K = hidden_layers[l]->K;
		int hidden_F = hidden_layers[l]->F;
	
		hidden_vs[l] = new double[hidden_M * hidden_K];
		hidden_vp[l] = new double[hidden_M * hidden_K];
		hidden_hs[l] = new double[hidden_F];
		hidden_hp[l] = new double[hidden_F];

		int* mask_visible = new int[hidden_M];
	}


	int output_F = output_layer->F;
	int output_M = output_layer->M;
	int output_K = output_layer->K;
	double* output_vs = new double[output_M * output_K];
	double* output_vp = new double[output_M * output_K];
	double* output_hs = new double[output_F];
	double* output_hp = new double[output_F];

    // Start calculating the running time
    struct timeval start;
    struct timeval end;
    unsigned long usec;
    gettimeofday(&start, NULL);

	for(int epoch = 0; epoch < train_epochs; epoch++) {

#pragma omp parallel for
		for (int batch = 0; batch < LS->nb_rows; batch += batch_size) {

			// 每个batch,首先赋值给input的vs，然后得到hp，再得到hs
			for (int n = batch; n < min(batch + batch_size, LS->nb_rows); n++) {


				zero(input_w_acc, M * K * F);
				zero(input_w_count, M * K * F);
				zero(input_vb_acc, M * K);
				zero(input_vb_count, M * K);
				zero(input_hb_acc, F);

				//    if (conditional) {
				zero(input_watched, M);
				//    }


				// Set user n data on the visible units
				for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
					int i = LS->ids[m];
					int ik_0 = i * K; // _ik(i, 0);

					for (int k = 0; k < K; k++) {
						input_vs[ik_0 + k] = 0.;
					}

					input_vs[ik_0 + LS->ratings[m] - 1] = 1.;
				}

				// Compute ^p = p(h | V, d) into hp
				
				input_layer->update_hidden(input_vs, &LS->ids[LS->index[n]], LS->count[n], &QS->ids[QS->index[n]], QS->count[n], input_hp);
				input_layer->sample_hidden(input_hp, input_hs);
				// Deallocate data structures
			} // End of for (int n = batch; n < min(batch + batch_size, LS->nb_rows); n++)

			// 这里做前馈传导，用vp去传。。
			// 最后一层hp传给输出层，用sigmoid算最终的hp。。

			for (int l = 0; l < hidden_layer_num - 1; l++) {


				int hidden_M = hidden_layers[l]->M;
				int hidden_K = hidden_layers[l]->K;
				int hidden_F = hidden_layers[l]->F;

//				hidden_vs[l] = new double[hidden_M * hidden_K];
//				hidden_vp[l] = new double[hidden_M * hidden_K];
//				hidden_hs[l] = new double[hidden_F];
//				hidden_hp[l] = new double[hidden_F];

				for (int i = 0; i < hidden_M; i++) {
					if (l == 0) {
						hidden_vs[l][i] = input_hs[i];
					}
					else {
						hidden_vs[l][i] = hidden_hs[l - 1][i];
					}
				}

				int* mask_visible = new int[hidden_M];
				for(int ind = 0; ind < hidden_M; ind++) {
					mask_visible[ind] = ind;
				}


				hidden_layers[l]->update_hidden_p(hidden_vs[l], &mask_visible[0], hidden_M, hidden_hp[l]);
				hidden_layers[l]->sample_hidden(hidden_hp[l], hidden_hs[l]);
				
				delete [] mask_visible;

			}  // End of for (int l = 0; l < hidden_layer_num - 1; l++)
			
			int output_F = hidden_layers[hidden_layer_num - 2]->F;
			int output_M = output_layer->M;
			
			int* mask_hidden= new int[output_F];
			for(int ind = 0; ind < output_F; ind++) {
				mask_hidden[ind] = ind;
			}
		
			int* mask_visible = new int[output_M];
			for(int ind = 0; ind < output_M; ind++) {
				mask_visible[ind] = ind;
			}

			// output:h->v,update visible,算vp的期望

			output_layer->update_visible(hidden_hs[hidden_layer_num - 2], output_vp, &mask_hidden[0], output_F);

            delete [] mask_hidden;
			delete [] mask_visible;

			// 开始计算残差，准备bp

			for (int n = 0; n < min(batch + batch_size, LS->nb_rows); n++) {
				
				// 只更新和这个user有关的那几个节点，每个节点k维
				for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
	
					int i = LS->ids[m];
					int ik_0 = _ik(i, 0);
					double prediction = 0.;
					double *delta_output = new double[output_M * output_K];

					for (int k = 0; k < K; k++) {
						
						prediction += output_vp[ik_0 + k] * (k + 1);
						double error = prediction - LS->ratings[m];
//						cout << "error: " << error << " prediction: " << prediction << " rating: " << TS->ratings[m] << " ik_0:" << ik_0 << " upbound: " << K*M <<endl;
						// cout << " n: " << n << " ids: " << i << " count: " << count << endl;
						
						total_error += error * error;
						count++;

					}

					delete [] delta_output;

				} //遍历完一个batch的所有user
				
				// 开始往回走，更新w和b,但是，d呢？

			}


		}  // End of for (int batch = 0; batch < LS->nb_rows; batch += batch_size)
	}  // End of for(int epoch = 0; epoch < train_epochs; epoch++)
	//    }  // End of for (int l = 0; l < hidden_layer_num - 1; l++)
	
	if (input_vs != NULL) delete[] input_vs;
	if (input_vp != NULL) delete[] input_vp;
	if (input_hs != NULL) delete[] input_hs;
	if (input_hp != NULL) delete[] input_hp;

	if (input_w_acc != NULL) delete[] input_w_acc;
	if (input_w_count != NULL) delete[] input_w_count;
	if (input_vb_acc != NULL) delete[] input_vb_acc;
	if (input_vb_count != NULL) delete[] input_vb_count;
	if (input_hb_acc != NULL) delete[] input_hb_acc;
	if (input_watched != NULL) delete[] input_watched;

    for(int i = 0; i < hidden_layer_num - 1; i++) {
		delete hidden_vs[i];
		delete hidden_vp[i];
		delete hidden_hs[i];
		delete hidden_hp[i];
	}

	if (output_vs != NULL) delete[] output_vs;
	if (output_vp != NULL) delete[] output_vp;
	if (output_hs != NULL) delete[] output_hs;
	if (output_hp != NULL) delete[] output_hp;

    // print running time
    gettimeofday(&end, NULL);
    usec = 1000000 * (end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;

    printf( "File: %s, Function: %s, Line: %d\n", __FILE__, __FUNCTION__, __LINE__);
    printf("Time of test(): %ld usec[ %lf sec].\n", usec, usec / 1000000.);
 
    return sqrt(total_error / count);
}



////{
////	//    // pop ls, qs and ts
////	//    dataset* ls = sets["ls"];
////	//    dataset* qs = sets["qs"];
////	//    dataset* ts = sets[dataset];
////	//    assert(ls != null);
////	//    assert(ts != null);
////	//    assert(ls->nb_rows == ts->nb_rows);
////	//
////	//    if (conditional) {
////	//        assert(qs != null);
////	//        assert(ls->nb_rows == qs->nb_rows);
////	//    }
////	//    
////	//    // start calculating the running time
////	//    struct timeval start;
////	//    struct timeval end;
////	//    unsigned long usec;
////	//    gettimeofday(&start, null);
////	//
////	//    // allocate local data structures
////	//    double* vs = new double[m * k];
////	//    double* vp = new double[m * k];
////	//    double* hs = new double[f];
////	//    double* hp = new double[f];
////	//
////	//
////	//    for(int i = 0; i < f; i++)
////	//        printf("testing hb[%d]: %lf\n", i, hb[i]);
////	//    // initialization
////	//    double total_error = 0.;
////	//    int count = 0;
////	//
////	//    // loop through users in the test set
////	//    for (int n = 0; n < ts->nb_rows; n++) {
////	//        if (ts->count[n] == 0) {
////	//            continue;
////	//        }
////	//
////	//        // set user n data on the visible units
////	//        for (int m = ls->index[n]; m < ls->index[n] + ls->count[n]; m++) {
////	//            int i = ls->ids[m];
////	//            int ik_0 = _ik(i, 0);
////	//
////	//            for (int k = 0; k < K; k++) {
////	//                vs[ik_0 + k] = 0.;
////	//            }
////	//
////	//            vs[ik_0 + LS->ratings[m] - 1] = 1.;
////	//        }
////	//
////	//        // Compute ^p = p(h | V, d) into hp
////	//        if (!conditional) {
////	//            update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], hp);
////	//        } else {
////	//            update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], &QS->ids[QS->index[n]], QS->count[n], hp);
////	//        }
////	//
////	//        // Compute p(v_ik = 1 | ^p) for all movie i in TS
////	//        update_visible(hp, vp, &TS->ids[TS->index[n]], TS->count[n]);
////	//
////	//        // Predict ratings
////	//        for (int m = TS->index[n]; m < TS->index[n] + TS->count[n]; m++) {
////	//            int i = TS->ids[m];
////	//            int ik_0 = _ik(i, 0);
////	//            double prediction = 0.;
////	//
////	//            for (int k = 0; k < K; k++) {
////	//                prediction += vp[ik_0 + k] * (k + 1);
////	//                // cout << "ik_0+k: " << ik_0 + k <<" vp[ik_0 + k]:" << vp[ik_0 + k] << endl;
////	//            }
////	//
////	//            double error = prediction - TS->ratings[m];
////	//            // cout << "error: " << error << " prediction: " << prediction << " rating: " << TS->ratings[m] << " ik_0:" << ik_0 << " upbound: " << K*M;
////	//	    // cout << " n: " << n << " ids: " << i << " count: " << count << endl;
////	//            total_error += error * error;
////	//            count++;
////	//        }
////	//    }
////	//
////	////    // Deallocate data structure
////	////    if (vs != NULL) { 
////	////        delete[] vs; 
////	////        vs = NULL; 
////	////    }
////	////    if (vp != NULL) { delete[] vp; vp = NULL; }
////	////    if (hs != NULL) { delete[] hs; hs = NULL; }
////	////    if (hp != NULL) { delete[] hp; hp = NULL; }
////	//
////	//    // cout << "total_error: " << total_error << " count: " << count << endl;
////	//    
////	//    // print running time
////	//    gettimeofday(&end, NULL);
////	//    usec = 1000000 * (end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
////	//
////	////    cout << "File: " << __FILE__ << ", Function: " << __FUNCTION__  << ", Line: " << __LINE__ << endl;
////	//    printf( "File: %s, Function: %s, Line: %d\n", __FILE__, __FUNCTION__, __LINE__);
////	////    cout << "Time of test(): " << usec << " usec[" << usec / 1000000. <<" sec]." << endl;
////	//    printf("Time of test(): %ld usec[ %lf sec].", usec, usec / 1000000.);
////	//    return sqrt(total_error / count);
////	return 1;
////}

double AHRBMCF::predict(int user, int movie)
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

void AHRBMCF::save(string filename)
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

string AHRBMCF::toString()
{
	stringstream s;

	s << "---" << endl;
	s << "Train Epochs = " << *(int*) getParameter("train_epochs") << endl;
	s << "Batch size = " << *(int*) getParameter("batch_size") << endl;
	s << "Hidden layer num = " << *(int*) getParameter("hidden_layer_num") << endl;

	for(int i = 0; i < hidden_layer_num; i++) {
		s << "Hidden layer" << i << " size: " << *(int*) getParameter("hidden_layer_size") << endl;
	}

	return s.str();

}

// AHRBMCF的函数
void AHRBMCF::train_separate(string dataset, bool reset)
{

	printf("AHRBMCF epochs: %d\n", train_epochs);
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


