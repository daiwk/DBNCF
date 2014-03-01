// class: RBMBASIC
// Author: daiwenkai
// Date: Feb 24, 2014

#include "RBMBASIC.h"
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

// Macros
#define _ikj(i, k, j) ((i) * K * F + (k) * F + (j))
#define _ik(i, k) ((i) * K + (k))
#define _ij(i, j) ((i) * F + (j))




// RBMBASIC::RBMBASIC() : RBM() 
RBMBASIC::RBMBASIC() : RBMCF_OPENMP() 
{

}

// RBMBASIC::RBMBASIC(string filename) : RBM(filename) 
RBMBASIC::RBMBASIC(string filename) : RBMCF_OPENMP(filename) 
{

}

RBMBASIC::~RBMBASIC()
{
}

void RBMBASIC::train_full(bool reset = true, int rbmlayers_id = 0)
{
    // Pop parameters
    int epochs = *(int*) getParameter("epochs");
    int batch_size = *(int*) getParameter("batch_size");
    int cd_steps = *(int*) getParameter("cd_steps");
    bool verbose = *(bool*) getParameter("verbose");
//    ostream* out = *(ostream**) getParameter("log");

//    // Pop LS
//    Dataset* LS = sets[dataset];
//    Dataset* QS = sets["QS"];
//    assert(LS != NULL);

//    if (conditional) {
//        assert(QS != NULL);
//        assert(LS->nb_rows == QS->nb_rows);
//    }

    // Start calculating the running time
    struct timeval start;
    struct timeval end;
    unsigned long usec;
    gettimeofday(&start, NULL);

    // Reset parameters
    if (reset) {
        // Reset everything
        this->reset();

//        // Set vb_ik to the logs of their respective base rates
//        // for (int n = 0; n < LS->nb_rows; n++) {
////            for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
////                vb[_ik(LS->ids[m], LS->ratings[m] - 1)] += 1.;
//            }
//        }
//
//        for (int i = 0; i < M; i++) {
//            int ik_0 = _ik(i, 0);
//            double total = 0.;
//
//            for (int k = 0; k < K; k++) {
//                total += vb[ik_0 + k];
//            }
//
//            if (total > 0.) {
//                for (int k = 0; k < K; k++) {
//                    int ik = ik_0 + k;
//
//                    if (vb[ik] == 0.) {
//                        vb[ik] = -10E5;
//                    } else {
//                        vb[ik] = log(vb[ik] / total);
//                    }
//                }
//            }
//        }
    }

    // print running time
    gettimeofday(&end, NULL);
    usec = 1000000 * (end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
//    cout << "File: " << __FILE__ << ", Function: " << __FUNCTION__  << ", Line: " << __LINE__ << endl;
    printf( "File: %s, Function: %s, Line: %d\n", __FILE__, __FUNCTION__, __LINE__);
//    cout << "Time of reset(): " << usec << " usec[" << usec / 1000000. <<" sec]." << endl;
    printf("Time of reset(): %ld usec[ %lf sec].", usec, usec / 1000000.);
    // Print some stats
//    if (verbose) {
//        *out << toString() << endl;
//        *out << "% ---" << endl;
//        *out << "% Epoch\tRMSE\tRMSE-TRAIN\tTIME\n";
//        out->flush();
//        //*out << 0 << "\t" << validate() << "\t" << usec / 1000000. << endl;
//        char res[1000];
//        sprintf(res, "0\t%lf\t%lf\t%lf\t\n", validate(), test("LS"), usec / 1000000.);
//        *out << res;
////        *out << 0 << "\t" << validate() << "\t" << usec / 1000000. << endl;
//	
//	
//        out->flush();
//    }

    // Start calculating the running time
    gettimeofday(&start, NULL);
    
    // Allocate local data structures
    double* vs = new double[M * K];
    double* vp = new double[M * K];
    double* hs = new double[F];
    double* hp = new double[F];

    double* w_acc = new double[M * K * F];
    int* w_count = new int[M * K * F];
    double* vb_acc = new double[M * K];
    int* vb_count = new int[M * K];
    double* hb_acc = new double[F];

//    bool* watched = NULL;
//    if (conditional) {
//        watched = new bool[M];
//    }
    
    char ss[1000];
    // 只有id=0（最开始那层）的时候才调用这个train函数，其他层调用的是train_full函数
    
    // 读取上一层的hs到这层的vs里面    
    sprintf(ss, "rbm-hs-%d", rbmlayers_id);
    string hs_filename_in = ss; 

    ifstream in_hs(hs_filename_in.c_str(), ios::in | ios::binary);
    if (in_hs.fail()) {
        throw runtime_error("I/O exception");
    }
    
    in_hs.read((char*) vs, M * K * sizeof (double));

    in_hs.close();

    for(int dd = 0; dd < M * K; dd++)
        printf("after reading from file: vs[%d]: %lf\n", dd, vs[dd]);

    // 读取上一层的hb到这层的vb里面    
    sprintf(ss, "rbm-hb-%d", rbmlayers_id);
    string hb_filename_in = ss; 

    ifstream in_hb(hb_filename_in.c_str(), ios::in | ios::binary);
    if (in_hb.fail()) {
        throw runtime_error("I/O exception");
    }
    
    in_hb.read((char*) vb, M * K * sizeof (double));

    in_hb.close();

    for(int dd = 0; dd < M * K; dd++)
        printf("after reading from file: vb[%d]: %lf\n", dd, vb[dd]);
    
    // Loop through epochs
    for (int epoch = 1; epoch <= epochs; epoch++) {
        
        // Start calculating the running time
        struct timeval start_train;
        struct timeval end_train;
        unsigned long usec_train;
        gettimeofday(&start_train, NULL);

        // Loop through mini-batches of users
        // for (int batch = 0; batch < LS->nb_rows; batch += batch_size)// 
        for (int batch = 0; batch < N; batch += batch_size) {
            // Reset the accumulators
            zero(w_acc, M * K * F);
            zero(w_count, M * K * F);
            zero(vb_acc, M * K);
            zero(vb_count, M * K);
            zero(hb_acc, F);

//            if (conditional) {
//                zero(watched, M);
//            }

            // Loop through users of current batch
            for (int n = batch; n < min(batch + batch_size, N); n++) {
            //for (int n = batch; n < min(batch + batch_size, LS->nb_rows); n++) // 
//                // Set user n data on the visible units
//                for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
//                    int i = LS->ids[m];
//                    int ik_0 = _ik(i, 0);
//
//                    for (int k = 0; k < K; k++) {
//                        vs[ik_0 + k] = 0.;
//                    }
//
//                    vs[ik_0 + LS->ratings[m] - 1] = 1.;
//                }

                int* mask_hidden = new int[F];
                for(int ind = 0; ind < F; ind++) {
                    mask_hidden[ind] = ind;
                }
                
                int* mask_visible = new int[M];
                for(int ind = 0; ind < M; ind++) {
                    mask_visible[ind] = ind;
                }

                // Compute p(h | V, d) into hp
                if (!conditional) {
//                    update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], hp);
                    update_hidden_p(vs, &mask_visible[0], M, hp);
                    printf("finished first update hidden\n");
                } else {
//                    update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], &QS->ids[QS->index[n]], QS->count[n], hp);
                }

                // Accumulate "v_ik * h_j", "v_ik" and "h_j"s (0)
                for (int m = 0; m < M; m++) {
//                for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) // 
//                    int i = LS->ids[m];
                    int i = m;
//                    int k = LS->ratings[m] - 1;
                    int k = 0;
                    int ikj_0 = _ikj(i, k, 0);

                    for (int j = 0; j < F; j++) {
                        w_acc[ikj_0 + j] += hp[j];
                    }

                    // 原来是对那些受影响的v+=1，现在是要更新所有v
                    vb_acc[_ik(i, k)] += 1.;
                    // 学习hb_acc，+= vp[]...不要乱搞。。。
                    printf("1.vb_acc[%d]: %lf\n", _ik(i, k), vb_acc[_ik(i, k)]);
                    printf("xx.vp[%d]: %lf\n", _ik(i, k), vp[_ik(i, k)]);
                    // vb_acc[_ik(i, k)] += vp[_ik(i, k)];
                    printf("2.vb_acc[%d]: %lf\n", _ik(i, k), vb_acc[_ik(i, k)]);
                }

                for (int j = 0; j < F; j++) {
                    hb_acc[j] += hp[j];
                    printf("hb_acc[%d]: %lf\n", j, hb_acc[j]);
                }

                // Count seen movies
//                if (conditional) {
//                    for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
//                        watched[LS->ids[m]] = true;
//                    }
//
//                    for (int m = QS->index[n]; m < QS->index[n] + QS->count[n]; m++) {
//                        watched[QS->ids[m]] = true;
//                    }
//                }

                // Gibbs sampling
                for (int step = 0; step < cd_steps; step++) {
                    // Sample from p(h | V, d) into hs
                    sample_hidden(hp, hs);

                    // Compute p(V | h) into vp
                    update_visible_p(hs, vp, &mask_hidden[0], F);
//                    update_visible(hs, vp, &LS->ids[LS->index[n]], LS->count[n]);

                    // Sample from p(V | h) into hs
                    sample_visible_p(vp, vs, &mask_visible[0], M);
//                    sample_visible(vp, vs, &LS->ids[LS->index[n]], LS->count[n]);

                    // Compute p(h | V, d) into hp
                    if (!conditional) {
                        update_hidden_p(vs, &mask_visible[0], M, hp);
                        printf("finished second update hidden\n");
//                        update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], hp);
                    } else {
//                        update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], &QS->ids[QS->index[n]], QS->count[n], hp);
                    }
                }

                // Accumulate "-v_ik * h_j", "-v_ik" and "-h_j"s (T)
                // Count the number of terms in the accumulators
                for (int m = 0; m < M; m++) {
//                for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) //
//                    int i = LS->ids[m];
                    int i = m;                    
                    int ik_0 = _ik(i, 0);

                    for (int k = 0; k < K; k++) {
                        int ik = ik_0 + k;
                        printf("after cd, vs[%d]: %lf\n", ik, vs[ik]);

                        for (int j = 0; j < F; j++) {
                            int ikj = _ikj(i, k, j);

                            w_acc[ikj] -= vs[ik] * hp[j];
                            // w_count[ikj]++;
                            w_count[ikj] = M * K * F;
                        }

                        vb_acc[ik] -= vs[ik];
                        vb_count[ik]++;
                        //vb_count[ik] = M * K;
                        printf("vb_count[%d]: %d\n", ik, vb_count[ik]);
                    }
                }

                for (int j = 0; j < F; j++) {
                    hb_acc[j] -= hp[j];
                }
            }

            // Update weights and biases
            // 参数3是用来算annealing rate用的
            update_w_p(w_acc, w_count, (epoch - 1) * N + batch);
            printf("last update_vb\n");
            update_vb_p(vb_acc, vb_count, (epoch - 1) * N + batch);
            update_hb_p(hb_acc, (epoch - 1) * N + batch);

//            if (conditional) {
//                update_d(hb_acc, watched, (epoch - 1) * LS->nb_rows + batch);
//            }
        }

//        // Log RMSE
//        if (verbose) {
//            
//            gettimeofday(&end_train, NULL);
//            usec_train = 1000000 * (end_train.tv_sec-start_train.tv_sec) + end_train.tv_usec - start_train.tv_usec;
//            char res[1000];
//            sprintf(res, "%d\t%lf\t%lf\t%lf\n", epoch, validate(), test("LS"), usec_train / 1000000.);
//            *out << res;
////            *out << epoch << "\t" << validate() << "\t" << usec_train / 1000000. << endl;
//            out->flush();
//        }
    }

    // Added by daiwenkai
    // 把hs的状态输出到文件，供下一层当做vs的输入

    sprintf(ss, "rbm-hs-%d", rbmlayers_id + 1);
    string hs_filename_out = ss; 
    ofstream out_hs(hs_filename_out.c_str(), ios::out | ios::binary);

    if (out_hs.fail()) {
        throw runtime_error("I/O exception! In openning rbm-hs-%d");
    }
    
    out_hs.write((char*) hs, F * sizeof (double));
    out_hs.close();
 
    // 把hb的状态输出到文件，供下一层当做vb的输入

    sprintf(ss, "rbm-hb-%d", rbmlayers_id + 1);
    string hb_filename_out = ss; 
    ofstream out_hb(hb_filename_out.c_str(), ios::out | ios::binary);

    if (out_hb.fail()) {
        throw runtime_error("I/O exception! In openning rbm-hs-%d");
    }
    
    out_hb.write((char*) hb, F * sizeof (double));
    out_hb.close();
   
    for(int dd = 0; dd < M; dd++)
        printf("after train_full: vb[%d]: %lf\n", dd, vb[dd]);

    for(int dd = 0; dd < F; dd++)
        printf("after train_full: hb[%d]: %lf\n", dd, hb[dd]);
    // Deallocate data structures
    if (vs != NULL) delete[] vs;
    if (vp != NULL) delete[] vp;
    if (hs != NULL) delete[] hs;
    if (hp != NULL) delete[] hp;

    if (w_acc != NULL) delete[] w_acc;
    if (w_count != NULL) delete[] w_count;
    if (vb_acc != NULL) delete[] vb_acc;
    if (vb_count != NULL) delete[] vb_count;
    if (hb_acc != NULL) delete[] hb_acc;
//    if (watched != NULL) delete[] watched;
    
    // print running time
    gettimeofday(&end, NULL);
    usec = 1000000 * (end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
//    cout << "File: " << __FILE__ << ", Function: " << __FUNCTION__  << ", Line: " << __LINE__ << endl;
    printf( "File: %s, Function: %s, Line: %d\n", __FILE__, __FUNCTION__, __LINE__);
//    cout << "Time of train(): " << usec << " usec[" << usec / 1000000. <<" sec]." << endl;
    printf("Time of train_full(): %ld usec[ %lf sec].", usec, usec / 1000000.);
}


void RBMBASIC::update_hidden_p(double* vs, int* mask, int mask_size, double* hp) {
    // Compute p(h | V) into hp
    for (int j = 0; j < F; j++) {
        hp[j] = hb[j];
        printf("out hp[%d]: %lf\n", j, hp[j]);
    }
    printf("mask size: %d\n", mask_size);

    for (int m = 0; m < mask_size; m++) {
        int i = mask[m];
        int ik_0 = _ik(i, 0);

        for (int k = 0; k < K; k++){
            int ik = ik_0 + k;
            int ikj_0 = _ikj(i, k, 0);

            for (int j = 0; j < F; j++) {
                hp[j] += vs[ik] * w[ikj_0 + j];
            }
        }
    }

    for (int j = 0; j < F; j++) {
        hp[j] = 1. / (1. + exp(-hp[j]));
        printf("final hp[%d]: %lf\n", j, hp[j]);
    }
}

void RBMBASIC::update_visible_p(double* hs, double* vp, int* mask, int mask_size) {
    // Compute p(V | h) into vp
    // 原来是exp(wh+b)/total(exp(wh+b))
    // 现在改成sigmoid(wh+b)
    for (int j = 0; j < M; j++) {
        vp[j] = vb[j];
        printf("out vp[%d]: %lf\n", j, vp[j]);
    }
    printf("mask size: %d\n", mask_size);

    for (int m = 0; m < mask_size; m++) {
        int i = mask[m];
        int ik_0 = _ik(i, 0);

        for (int k = 0; k < K; k++){
            int ik = ik_0 + k;
            int ikj_0 = _ikj(i, k, 0);

            for (int j = 0; j < M; j++) {
                vp[j] += hs[ik] * w[ikj_0 + j];
            }
        }
    }

    for (int j = 0; j < M; j++) {
        vp[j] = 1. / (1. + exp(-vp[j]));
        printf("final vp[%d]: %lf\n", j, vp[j]);
    }
    
}

void RBMBASIC::update_w_p(double* w_acc, int* w_count, int nth) {
    // Pop parameters
    double eps_w = *(double*) getParameter("eps_w");
    double weight_cost = *(double*) getParameter("weight_cost");
    double momentum = *(double*) getParameter("momentum");
    bool annealing = *(bool*) getParameter("annealing");
    double annealing_rate = *(double*) getParameter("annealing_rate");

    // Update weights
    if (annealing) {
        eps_w /= 1. + (nth / (annealing_rate * N));
    }

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            int ikj_0 = _ikj(i, k, 0);

            for (int j = 0; j < F; j++) {
                int ikj = ikj_0 + j;

                if (w_count[ikj] > 0) {
                    w_acc[ikj] /= w_count[ikj];
                }

                w_inc[ikj] *= momentum;
                w_inc[ikj] += eps_w * (w_acc[ikj] - weight_cost * w[ikj]);
                w[ikj] += w_inc[ikj];
            }
        }
    }
}

void RBMBASIC::update_vb_p(double* vb_acc, int* vb_count, int nth) {
    // Pop parameters
    double eps_vb = *(double*) getParameter("eps_vb");
    double momentum = *(double*) getParameter("momentum");
    bool annealing = *(bool*) getParameter("annealing");
    double annealing_rate = *(double*) getParameter("annealing_rate");

    // Update biases of visible units
    if (annealing) {
        eps_vb /= 1. + (nth / (annealing_rate * N));
    }

    for (int i = 0; i < M; i++) {

        int ik_0 = _ik(i, 0);

        for (int k = 0; k < K; k++) {
            int ik = ik_0 + k;
            
            printf("vb[%d] before: %lf\n", ik, vb[ik]);

            if (vb_count[ik] > 0) {
                vb_acc[ik] /= vb_count[ik];
            }

            printf("vb_inc[%d] before: %lf\n", ik, vb_inc[ik]);
            vb_inc[ik] *= momentum;
            vb_inc[ik] += eps_vb * vb_acc[ik];
            vb[ik] += vb_inc[ik];
            
            printf("vb_inc[%d] now: %lf\n", ik, vb_inc[ik]);
            printf("vb[%d] now: %lf\n", ik, vb[ik]);
        }
    }
}

void RBMBASIC::update_hb_p(double* hb_acc, int nth) {
    // Pop parameters
    int batch_size = *(int*) getParameter("batch_size");
    double eps_hb = *(double*) getParameter("eps_hb");
    double momentum = *(double*) getParameter("momentum");
    bool annealing = *(bool*) getParameter("annealing");
    double annealing_rate = *(double*) getParameter("annealing_rate");
    
    // Update biases of hidden units
    if (annealing) {
        eps_hb /= 1. + (nth / (annealing_rate * N));
    }

    for (int j = 0; j < F; j++) {
        hb_inc[j] *= momentum;
        hb_inc[j] += eps_hb * hb_acc[j] / batch_size;
        hb[j] += hb_inc[j];
    }
}

void RBMBASIC::sample_visible_p(double* vp, double* vs, int* mask, int mask_size) {
// 下面是rbm_for_cf的sample方法
//     // Sample from p(V | h) into vs
//     for (int m = 0; m < mask_size; m++) {
//         int i = mask[m];
//         int ik_0 = _ik(i, 0);
// 
//         double r = uniform();
//         int k = 0;
// 
//         while (k < K && r > vp[ik_0 + k]) {
//             vs[ik_0 + k] = 0.;
//             printf("before r: %lf\n", r);
//             r -= vp[ik_0 + k];
//             printf("now r: %lf\n", r);
//             k++;
//         }
// 
//         if (k == K) { // Can this really happen? // 即不是因为r>vp而跳出循环的
//             vs[ik_0 + k - 1] = 1.;
//         } else {
//             vs[ik_0 + k] = 1.;
//             k++;
// 
//             for (; k < K; k++) {
//                 vs[ik_0 + k] = 0.;
//             }
//         }
//     }
// 这里试着使用和sample_hidden一样的方式进行sample

    for (int j = 0; j < mask_size; j++) {
        vs[j] = (uniform() < vp[j]) ? 1. : 0.;
    }
}
