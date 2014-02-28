/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Implementation of "RBMOpenMP.h".
 *
 * - Author: LOUPPE Gilles
 * - Last changes: February 26, 2010
 * ========================================================================= */

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>

#include "Dataset.h"
#include "Misc.h"
#include "RBMOpenMP.h"

//#include <mpi.h>

using namespace std;


/* ========================================================================= *
 * Macros
 * ========================================================================= */

#define _ikj(i, k, j) ((i) * K * F + (k) * F + (j))
#define _ik(i, k) ((i) * K + (k))
#define _ij(i, j) ((i) * F + (j))


/* ========================================================================= *
 * OpenMP RBM model
 * ========================================================================= */

/* ------------------------------------------------------------------------- *
 * Constructors
 * ------------------------------------------------------------------------- */

RBMOpenMP::RBMOpenMP() : RBM() {
    // Nothing more to do
}

RBMOpenMP::RBMOpenMP(string filename) : RBM(filename) {
    // Nothing more to do
}

RBMOpenMP::~RBMOpenMP() {
    // Nothing more to do
}

/* ------------------------------------------------------------------------- *
 * Model
 * ------------------------------------------------------------------------- */

void RBMOpenMP::train(string dataset, bool reset) {
    // Pop parameters
    int epochs = *(int*) getParameter("epochs");
    int batch_size = *(int*) getParameter("batch_size");
    int cd_steps = *(int*) getParameter("cd_steps");
    bool verbose = *(bool*) getParameter("verbose");
    ostream* out = *(ostream**) getParameter("log");

    // Pop LS
    Dataset* LS = sets[dataset];
    Dataset* QS = sets["QS"];
    assert(LS != NULL);

    if (conditional) {
        assert(QS != NULL);
        assert(LS->nb_rows == QS->nb_rows);
    }

    // Start calculating the running time
    struct timeval start;
    struct timeval end;
    unsigned long usec;
    gettimeofday(&start, NULL);
    
    // Reset parameters
    if (reset) {
        // Reset everything
        this->reset();
        
        // Set vb_ik to the logs of their respective base rates
        for (int n = 0; n < LS->nb_rows; n++) {
            for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
                vb[_ik(LS->ids[m], LS->ratings[m] - 1)] += 1.;
            }
        }

        for (int i = 0; i < M; i++) {
            int ik_0 = _ik(i, 0);
            double total = 0.;

            for (int k = 0; k < K; k++) {
                total += vb[ik_0 + k];
            }

            if (total > 0.) {
                for (int k = 0; k < K; k++) {
                    int ik = ik_0 + k;

                    if (vb[ik] == 0.) {
                        vb[ik] = -10E5; //commented by dwk at 23:23,2,17,2014
                    } else {
                        vb[ik] = log(vb[ik] / total);
                    }
                }
            }
        }
    }

    // print running time
    gettimeofday(&end, NULL);
    usec = 1000000 * (end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
//    cout << "File: " << __FILE__ << ", Function: " << __FUNCTION__  << ", Line: " << __LINE__ << endl;
    printf( "File: %s, Function: %s, Line: %d\n", __FILE__, __FUNCTION__, __LINE__);
//    cout << "Time of reset(): " << usec << " usec[" << usec / 1000000. <<" sec]." << endl;
    printf("Time of reset(): %ld usec[ %lf sec].", usec, usec / 1000000.);
    
    // Print some stats
    if (verbose) {
        *out << toString() << endl;
        *out << "% ---" << endl;
        *out << "% Epoch\tRMSE\tRMSE-TRAIN\tTIME\n";
        out->flush();
        //*out << 0 << "\t" << validate() << "\t" << usec / 1000000. << endl;
        char res[1000];
        sprintf(res, "0\t%lf\t%lf\t%lf\t\n", validate(), test("LS"), usec / 1000000.);
        *out << res;
//        *out << 0 << "\t" << validate() << "\t" << usec / 1000000. << endl;
        out->flush();
    }
    
    // Start calculating the running time
    gettimeofday(&start, NULL);

    // Loop through epochs
    for (int epoch = 1; epoch <= epochs; epoch++) {
   
        // Start calculating the running time
        struct timeval start_train;
        struct timeval end_train;
        unsigned long usec_train;
        gettimeofday(&start_train, NULL);

        
        // Begin parallel section
        #pragma omp parallel
        {
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

        bool* watched = NULL;
        if (conditional) {
            watched = new bool[M];
        }

       

        // Parallel for
        #pragma omp for schedule(guided)

        // Loop through mini-batches of users
        for (int batch = 0; batch < LS->nb_rows; batch += batch_size) {
            // Reset the accumulators
            zero(w_acc, M * K * F);
            zero(w_count, M * K * F);
            zero(vb_acc, M * K);
            zero(vb_count, M * K);
            zero(hb_acc, F);

            if (conditional) {
                zero(watched, M);
            }

            // Loop through users of current batch
            for (int n = batch; n < min(batch + batch_size, LS->nb_rows); n++) {
                // Set user n data on the visible units
                for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
                    int i = LS->ids[m];
                    int ik_0 = _ik(i, 0);

                    for (int k = 0; k < K; k++) {
                        vs[ik_0 + k] = 0.;
                    }

                    vs[ik_0 + LS->ratings[m] - 1] = 1.;
                }

                // Compute p(h | V, d) into hp
                if (!conditional) {
                    update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], hp);
                } else {
                    update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], &QS->ids[QS->index[n]], QS->count[n], hp);
                }

                // Accumulate "v_ik * h_j", "v_ik" and "h_j"s (0)
                for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
                    int i = LS->ids[m];
                    int k = LS->ratings[m] - 1;
                    int ikj_0 = _ikj(i, k, 0);

                    for (int j = 0; j < F; j++) {
                        w_acc[ikj_0 + j] += hp[j];
                    }

                    vb_acc[_ik(i, k)] += 1.;
                }

                for (int j = 0; j < F; j++) {
                    hb_acc[j] += hp[j];
                }

                // Count seen movies
                if (conditional) {
                    for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
                        watched[LS->ids[m]] = true;
                    }

                    for (int m = QS->index[n]; m < QS->index[n] + QS->count[n]; m++) {
                        watched[QS->ids[m]] = true;
                    }
                }

                // Gibbs sampling
                for (int step = 0; step < cd_steps; step++) {
                    // Sample from p(h | V, d) into hs
                    sample_hidden(hp, hs);

                    // Compute p(V | h) into vp
                    update_visible(hs, vp, &LS->ids[LS->index[n]], LS->count[n]);

                    // Sample from p(V | h) into hs
                    sample_visible(vp, vs, &LS->ids[LS->index[n]], LS->count[n]);

                    // Compute p(h | V, d) into hp
                    if (!conditional) {
                        update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], hp);
                    } else {
                        update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], &QS->ids[QS->index[n]], QS->count[n], hp);
                    }
                }

                // Accumulate "-v_ik * h_j", "-v_ik" and "-h_j"s (T)
                // Count the number of terms in the accumulators
                for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
                    int i = LS->ids[m];
                    int ik_0 = _ik(i, 0);

                    for (int k = 0; k < K; k++) {
                        int ik = ik_0 + k;

                        for (int j = 0; j < F; j++) {
                            int ikj = _ikj(i, k, j);

                            w_acc[ikj] -= vs[ik] * hp[j];
                            w_count[ikj]++;
                        }

                        vb_acc[ik] -= vs[ik];
                        vb_count[ik]++;
                    }
                }

                for (int j = 0; j < F; j++) {
                    hb_acc[j] -= hp[j];
                }
            }

            // Update weights and biases
            #pragma omp critical
            update_w(w_acc, w_count, (epoch - 1) * LS->nb_rows + batch);

            #pragma omp critical
            update_vb(vb_acc, vb_count, (epoch - 1) * LS->nb_rows + batch);

            #pragma omp critical
            update_hb(hb_acc, (epoch - 1) * LS->nb_rows + batch);

            if (conditional) {
                #pragma omp critical
                update_d(hb_acc, watched, (epoch - 1) * LS->nb_rows + batch);
            }
        }
        
// 	// Deallocate data structures
//         if (vs != NULL) delete[] vs;
//         if (vp != NULL) delete[] vp;
//         if (hs != NULL) delete[] hs;
//         if (hp != NULL) delete[] hp;
// 
//         if (w_acc != NULL) delete[] w_acc;
//         if (w_count != NULL) delete[] w_count;
//         if (vb_acc != NULL) delete[] vb_acc;
//         if (vb_count != NULL) delete[] vb_count;
//         if (hb_acc != NULL) delete[] hb_acc;
//         if (watched != NULL) delete[] watched;


        // End parallel section
        }

        
        // Log RMSE
        if (verbose) {
   
            gettimeofday(&end_train, NULL);
            usec_train = 1000000 * (end_train.tv_sec-start_train.tv_sec) + end_train.tv_usec - start_train.tv_usec;
            char res[1000];
            sprintf(res, "%d\t%lf\t%lf\t%lf\n", epoch, validate(), test("LS"), usec_train / 1000000.);
            *out << res;
//            *out << epoch << "\t" << validate() << "\t" << usec_train / 1000000. << endl;
            out->flush();
        }
    }
    
    // print running time
    gettimeofday(&end, NULL);
    usec = 1000000 * (end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
//    cout << "File: " << __FILE__ << ", Function: " << __FUNCTION__  << ", Line: " << __LINE__ << endl;
    printf( "File: %s, Function: %s, Line: %d\n", __FILE__, __FUNCTION__, __LINE__);
//    cout << "Time of train(): " << usec << " usec[" << usec / 1000000. <<" sec]." << endl;
    printf("Time of train(): %ld usec[ %lf sec].", usec, usec / 1000000.);
}

double RBMOpenMP::test(string dataset) {
    // Pop LS, QS and TS
    Dataset* LS = sets["LS"];
    Dataset* QS = sets["QS"];
    Dataset* TS = sets[dataset];
    assert(LS != NULL);
    assert(TS != NULL);
    assert(LS->nb_rows == TS->nb_rows);

    if (conditional) {
        assert(QS != NULL);
        assert(LS->nb_rows == QS->nb_rows);
    }

    // Start calculating the running time
    struct timeval start;
    struct timeval end;
    unsigned long usec;
    gettimeofday(&start, NULL);
    
    // Initialization
    double total_error = 0.;
    int count = 0;


    // Begin parallel section
    #pragma omp parallel
    {
    // Allocate local data structures
    double* vs = new double[M * K];
    double* vp = new double[M * K];
    double* hs = new double[F];
    double* hp = new double[F];


    // Parallel for
    #pragma omp for schedule(guided) reduction(+: total_error, count)

    // Loop through users in the test set
    for (int n = 0; n < TS->nb_rows; n++) {
        if (TS->count[n] == 0) {
            continue;
        }

        // Set user n data on the visible units
        for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
            int i = LS->ids[m];
            int ik_0 = _ik(i, 0);

            for (int k = 0; k < K; k++) {
                vs[ik_0 + k] = 0.;
            }

            vs[ik_0 + LS->ratings[m] - 1] = 1.;
	    // cout << "n: " << n << "index: " << m << " count: " << LS->count[n] << " ids: " << LS->ids[n] << " rating: " << LS->ratings[m];
        }

        // Compute ^p = p(h | V, d) into hp
        if (!conditional) {
            update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], hp);
        } else {
            update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], &QS->ids[QS->index[n]], QS->count[n], hp);
        }

        // Compute p(v_ik = 1 | ^p) for all movie i in TS
        update_visible(hp, vp, &TS->ids[TS->index[n]], TS->count[n]);

        // Predict ratings
        for (int m = TS->index[n]; m < TS->index[n] + TS->count[n]; m++) {
            int i = TS->ids[m];
            int ik_0 = _ik(i, 0);
            double prediction = 0.;

            for (int k = 0; k < K; k++) {
                prediction += vp[ik_0 + k] * (k + 1);
                // cout << "ik_0+k: " << ik_0 + k <<" vp[ik_0 + k]: " << vp[ik_0 + k] << endl;
            }

            double error = prediction - TS->ratings[m];
            // cout << "error: " << error << " prediction: " << prediction << " rating: " << TS->ratings[m] << " ik_0:" << ik_0 << " upbound: " << K*M;
	    // cout << " n: " << n << " ids: " << i << " count: " << count << endl;
            total_error += error * error;
            count++;
        }
    }

//    // Deallocate data structure
//    if (vs != NULL) { 
//        delete[] vs; 
//        vs = NULL; 
//    }
//    if (vp != NULL) { delete[] vp; vp = NULL; }
//    if (hs != NULL) { delete[] hs; hs = NULL; }
//    if (hp != NULL) { delete[] hp; hp = NULL; }


    // End parallel section
    }

    // cout << "total_error: " << total_error << " count: " << count << endl;
    
    // print running time
    gettimeofday(&end, NULL);
    usec = 1000000 * (end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
//    cout << "File: " << __FILE__ << ", Function: " << __FUNCTION__  << ", Line: " << __LINE__ << endl;
    printf( "File: %s, Function: %s, Line: %d\n", __FILE__, __FUNCTION__, __LINE__);
//    cout << "Time of test(): " << usec << " usec[" << usec / 1000000. <<" sec]." << endl;
    printf("Time of test(): %ld usec[ %lf sec].", usec, usec / 1000000.);

    return sqrt(total_error / count);
}

string RBMOpenMP::toString() {
    stringstream s;
    s << "Multi-threaded" << endl;
    s << RBM::toString();

    return s.str();
}

