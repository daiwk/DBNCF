/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Implementation of "RBMCF.h".
 *
 * - Author: LOUPPE Gilles
 * - Last changes: April 8, 2010
 * ========================================================================= */

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "Configuration.h"
#include "Dataset.h"
#include "Misc.h"
#include "RBMCF.h"

using namespace std;


/* ========================================================================= *
 * Macros
 * ========================================================================= */

#define _ikj(i, k, j) ((i) * K * F + (k) * F + (j))
#define _ik(i, k) ((i) * K + (k))
#define _ij(i, j) ((i) * F + (j))


/* ========================================================================= *
 * RBMCF model
 * ========================================================================= */

/* ------------------------------------------------------------------------- *
 * Constructors
 * ------------------------------------------------------------------------- */

RBMCF::RBMCF() : Model(CLASS_RBMCF) {
    // Set default parameters
    cout << "RBMCF default constructor " <<endl;
    N = Config::RBMCF::N;
    M = Config::RBMCF::M;
    K = Config::RBMCF::K;
    F = Config::RBMCF::F;
    conditional = Config::RBMCF::CONDITIONAL;

    setParameter("N", &Config::RBMCF::N, sizeof(int));
    setParameter("M", &Config::RBMCF::M, sizeof(int));
    setParameter("K", &Config::RBMCF::K, sizeof(int));
    setParameter("F", &Config::RBMCF::F, sizeof(int));
    setParameter("conditional", &Config::RBMCF::CONDITIONAL, sizeof(bool));

    setParameter("epochs", &Config::RBMCF::EPOCHS, sizeof(int));
    setParameter("batch_size", &Config::RBMCF::BATCH_SIZE, sizeof(int));
    setParameter("cd_steps", &Config::RBMCF::CD_STEPS, sizeof(int));
    setParameter("eps_w", &Config::RBMCF::EPS_W, sizeof(double));
    setParameter("eps_vb", &Config::RBMCF::EPS_VB, sizeof(double));
    setParameter("eps_hb", &Config::RBMCF::EPS_HB, sizeof(double));
    setParameter("eps_d", &Config::RBMCF::EPS_D, sizeof(double));
    setParameter("weight_cost", &Config::RBMCF::WEIGHT_COST, sizeof(double));
    setParameter("momentum", &Config::RBMCF::MOMENTUM, sizeof(double));
    setParameter("annealing", &Config::RBMCF::ANNEALING, sizeof(bool));
    setParameter("annealing_rate", &Config::RBMCF::ANNEALING_RATE, sizeof(double));
    setParameter("verbose", &Config::RBMCF::VERBOSE, sizeof(bool));

    ostream* log = &cout;
    setParameter("log", &log, sizeof(ostream*));

    // Set data structures to NULL
    w = NULL;
    w_inc = NULL;
    vb = NULL;
    vb_inc = NULL;
    hb = NULL;
    hb_inc = NULL;
    d = NULL;
    d_inc = NULL;
}

RBMCF::RBMCF(string filename) : Model(CLASS_RBMCF) {
    // Open file
    ifstream in(filename.c_str(), ios::in | ios::binary);
    cout << "RBMCF: " << filename << endl;
    if (in.fail()) {
        throw runtime_error("I/O exception");
    }

    // Check ID
    char* id = new char[2];
    in.read(id, 2 * sizeof(char));
    // for ( int i = 0; i < 2; i++ ) {
	    //in >> id[i];
        // cout <<"idxxxx:" << id[i] <<endl;

    // }
    // cout<<"xxx" << char(0x30 + __id) <<"xxx"<<endl;
    // cout<<"xxx" << char(0x0A) <<"xxx"<<endl;
    assert(id[0] == (0x30 + __id) && id[1] == 0x0A);

    // Load parameters
    int tmp_int;
    int tmp_double;
    int tmp_bool;

    in.read((char*) &N, sizeof(int));
    // in >> N;
    setParameter("N", &N, sizeof(int));
    in.read((char*) &M, sizeof(int));
    // in >> M;
    setParameter("M", &M, sizeof(int));
    in.read((char*) &K, sizeof(int));
    // in >> K;
    setParameter("K", &K, sizeof(int));
    in.read((char*) &F, sizeof(int));
    // in >> F;
    setParameter("F", &F, sizeof(int));
    in.read((char*) &conditional, sizeof(bool));
    // in >> conditional;
    setParameter("conditional", &conditional, sizeof(bool));

    in.read((char*) &tmp_int, sizeof (int));
    // in >> tmp_int;
    setParameter("epochs", &tmp_int, sizeof(int));
    // in >> tmp_int;
    in.read((char*) &tmp_int, sizeof (int));
    setParameter("batch_size", &tmp_int, sizeof(int));
    //in >> tmp_int;
    in.read((char*) &tmp_int, sizeof (int));
    setParameter("cd_steps", &tmp_int, sizeof(int));
    // in >> tmp_double;
    in.read((char*) &tmp_double, sizeof (double));
    setParameter("eps_w", &tmp_double, sizeof(double));
    // in >> tmp_double;
    in.read((char*) &tmp_double, sizeof (double));
    setParameter("eps_vb", &tmp_double, sizeof(double));
    // in >> tmp_double;
    in.read((char*) &tmp_double, sizeof (double));
    setParameter("eps_hb", &tmp_double, sizeof(double));
    // in >> tmp_double;
    in.read((char*) &tmp_double, sizeof (double));
    setParameter("eps_d", &tmp_double, sizeof(double));
    // in >> tmp_double;
    in.read((char*) &tmp_double, sizeof (double));
    setParameter("weight_cost", &tmp_double, sizeof(double));
    // in >> tmp_double;
    in.read((char*) &tmp_double, sizeof (double));
    setParameter("momentum", &tmp_double, sizeof(double));
    // in >> tmp_double;
    in.read((char*) &tmp_bool, sizeof (bool));
    setParameter("annealing", &tmp_bool, sizeof(bool));
    // in >> tmp_double;
    in.read((char*) &tmp_double, sizeof (double));
    setParameter("annealing_rate", &tmp_double, sizeof(double));

    // Load weights and biases
    w = new double[M * K * F];
    vb = new double[M * K];
    hb = new double[F];
    in.read((char*) w, M * K * F * sizeof (double));
    // for ( int i = 0; i < M * K * F; i++ )
   	// in >> w[i]; 
    in.read((char*) vb, M * K * sizeof (double));
    // for ( int i = 0; i < M * K; i++ )
   	// in >> vb[i]; 
    in.read((char*) hb, F * sizeof (double));
    // for ( int i = 0; i < F; i++ )
   	// in >> hb[i]; 

    w_inc = new double[M * K * F];
    vb_inc = new double[M * K];
    hb_inc = new double[F];
    zero(w_inc, M * K * F);
    zero(vb_inc, M * K);
    zero(hb_inc, F);


    if (conditional) {
        d = new double[M * F];
        in.read((char*) d, M * F * sizeof (double));
    	// for ( int i = 0; i < M * F; i++ )
   	    // in >> d[i]; 


        d_inc = new double[M * F];
        zero(d_inc, M * F);
    } else{
        d = NULL;
        d_inc = NULL;
    }

    // Default verbose and output
    setParameter("verbose", &Config::RBMCF::VERBOSE, sizeof(bool));

    ostream* log = &cout;
    setParameter("log", &log, sizeof(ostream*));

    // Close file
    in.close();
}

RBMCF::~RBMCF() {
    delete[] w;
    delete[] w_inc;
    delete[] vb;
    delete[] vb_inc;
    delete[] hb;
    delete[] hb_inc;
    delete[] d;
    delete[] d_inc;
}

/* ------------------------------------------------------------------------- *
 * Model
 * ------------------------------------------------------------------------- */

void RBMCF::train(string dataset, bool reset) {
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
                        vb[ik] = -10E5;
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

    // Loop through epochs
    for (int epoch = 1; epoch <= epochs; epoch++) {
        
        // Start calculating the running time
        struct timeval start_train;
        struct timeval end_train;
        unsigned long usec_train;
        gettimeofday(&start_train, NULL);

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
            update_w(w_acc, w_count, (epoch - 1) * LS->nb_rows + batch);
            update_vb(vb_acc, vb_count, (epoch - 1) * LS->nb_rows + batch);
            update_hb(hb_acc, (epoch - 1) * LS->nb_rows + batch);

            if (conditional) {
                update_d(hb_acc, watched, (epoch - 1) * LS->nb_rows + batch);
            }
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

    // Added by daiwenkai
    // 把hs的状态输出到文件，供RBMCF_P当做输入


    char ss[1000];
    // 只有rbmlayers_id=0（最开始那层）的时候才调用这个train函数，其他层调用的是train_full函数
    int rbmlayers_id = 0;
    sprintf(ss, "rbm-hs-%d", rbmlayers_id);
    string hs_filename = ss; 
    ofstream out_hs(hs_filename.c_str(), ios::out | ios::binary);

    if (out_hs.fail()) {
        throw runtime_error("I/O exception! In openning rbm-hs-%d");
    }
    
    for(int dd = 0; dd < F; dd++)
        printf("after train: hs[%d]: %lf\n", dd, hs[dd]);
    
    out_hs.write((char*) hs, F * sizeof (double));
    out_hs.close();

    // 把hb的状态输出到文件，供RBMCF_P当做输入

    sprintf(ss, "rbm-hb-%d", rbmlayers_id);
    string hb_filename_out = ss; 
    ofstream out_hb(hb_filename_out.c_str(), ios::out | ios::binary);

    if (out_hb.fail()) {
        throw runtime_error("I/O exception! In openning rbm-hs-%d");
    }
    
    out_hb.write((char*) hb, F * sizeof (double));
    out_hb.close();
    
 
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
    if (watched != NULL) delete[] watched;
    
    // print running time
    gettimeofday(&end, NULL);
    usec = 1000000 * (end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
//    cout << "File: " << __FILE__ << ", Function: " << __FUNCTION__  << ", Line: " << __LINE__ << endl;
    printf( "File: %s, Function: %s, Line: %d\n", __FILE__, __FUNCTION__, __LINE__);
//    cout << "Time of train(): " << usec << " usec[" << usec / 1000000. <<" sec]." << endl;
    printf("Time of train(): %ld usec[ %lf sec].", usec, usec / 1000000.);
}

double RBMCF::test(string dataset) {
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

    // Allocate local data structures
    double* vs = new double[M * K];
    double* vp = new double[M * K];
    double* hs = new double[F];
    double* hp = new double[F];


    for(int i = 0; i < F; i++)
        printf("testing hb[%d]: %lf\n", i, hb[i]);
    // Initialization
    double total_error = 0.;
    int count = 0;

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
                // cout << "ik_0+k: " << ik_0 + k <<" vp[ik_0 + k]:" << vp[ik_0 + k] << endl;
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

double RBMCF::validate(string dataset) {
    return test(dataset);
}

double RBMCF::predict(int user, int movie) {
    // Pop LS
    Dataset* LS = sets["LS"];
    Dataset* QS = sets["QS"];
    assert(LS != NULL);

    if (conditional) {
        assert(QS != NULL);
        assert(LS->nb_rows == QS->nb_rows);
    }

    // Asserts
    assert(user >= 0);
    assert(user < N);
    assert(movie >= 0);
    assert(movie < M);

    // Reject if user is unknown
    if (!LS->contains_user(user)) {
        return -1.0;
    }

    /*
    if (LS->count[user] <= 0) {
        cout << "unknown user 2" << endl;
        return -1.0;
    }
    */

    // Reject if movie is unknown
    if (!LS->contains_movie(movie)){
        return -1.0;
    }

    // Allocate local data structures
    double* vs = new double[M * K];
    double* vp = new double[M * K];
    double* hs = new double[F];
    double* hp = new double[F];

    // Set user data on the visible units
    int n = LS->users[user];

    for (int m = LS->index[n]; m < LS->index[n] + LS->count[n]; m++) {
        int i = LS->ids[m];
        int ik_0 = _ik(i, 0);

        for (int k = 0; k < K; k++) {
            vs[ik_0 + k] = 0.;
        }

        vs[ik_0 + LS->ratings[m] - 1] = 1.;
    }

    // Compute ^p = p(h | V, d) into hp
    if (!conditional) {
        update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], hp);
    } else {
        update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], &QS->ids[QS->index[n]], QS->count[n], hp);
    }

    // Compute p(v_ik = 1 | ^p) for i = movie
    update_visible(hp, vp, &movie, 1);

    // Predict rating
    double prediction = 0.;
    int ik_0 = _ik(movie, 0);

    for (int k = 0; k < K; k++) {
        prediction += vp[ik_0 + k] * (k + 1);
    }

    // Deallocate data structure
    delete[] vs;
    delete[] vp;
    delete[] hs;
    delete[] hp;

    return prediction;
}

void RBMCF::save(string filename) {
    // Open file
    ofstream out(filename.c_str(), ios::out | ios::binary);

    if (out.fail()) {
        throw runtime_error("I/O exception!");
    }
    
    printf("saving...%d,%d,%d,%d to %s\n", N, M, K, F, filename.c_str());

    // Write class ID
    char id[2] = {0x30 + __id, 0x0A};
    out.write(id, 2 * sizeof (char));

    // Write parameters
    out.write((char*) &N, sizeof (int));
    out.write((char*) &M, sizeof (int));
    out.write((char*) &K, sizeof (int));
    out.write((char*) &F, sizeof (int));
    out.write((char*) &conditional, sizeof (bool));

    out.write((char*) getParameter("epochs"), sizeof (int));
    out.write((char*) getParameter("batch_size"), sizeof (int));
    out.write((char*) getParameter("cd_steps"), sizeof (int));
    out.write((char*) getParameter("eps_w"), sizeof (double));
    out.write((char*) getParameter("eps_vb"), sizeof (double));
    out.write((char*) getParameter("eps_hb"), sizeof (double));
    out.write((char*) getParameter("eps_d"), sizeof (double));
    out.write((char*) getParameter("weight_cost"), sizeof (double));
    out.write((char*) getParameter("momentum"), sizeof (double));
    out.write((char*) getParameter("annealing"), sizeof (bool));
    out.write((char*) getParameter("annealing_rate"), sizeof (double));

    out.write((char*) w, M * K * F * sizeof (double));
    out.write((char*) vb, M * K * sizeof (double));
    out.write((char*) hb, F * sizeof (double));

    if (conditional) {
        out.write((char*) d, M * F * sizeof (double));
    }

    out.close();
}

string RBMCF::toString() {
    stringstream s;

    if (!conditional) {
        s << "Basic RBMCF" << endl;
    } else {
        s << "Conditional RBMCF" << endl;
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

/* ------------------------------------------------------------------------- *
 * RBMCF
 * ------------------------------------------------------------------------- */

void RBMCF::reset(void) {
    // Pop parameters
    N = *(int*) getParameter("N");
    M = *(int*) getParameter("M");
    K = *(int*) getParameter("K");
    F = *(int*) getParameter("F");
    conditional = *(bool*) getParameter("conditional");

    // Deallocate old data structures
    if (w != NULL) delete[] w;
    if (w_inc != NULL) delete[] w_inc;
    if (vb != NULL) delete[] vb;
    if (vb_inc != NULL) delete [] vb_inc;
    if (hb != NULL) delete[] hb;
    if (hb_inc != NULL) delete[] hb_inc;
    if (d != NULL) delete[] d;
    if (d_inc != NULL) delete[] d_inc;

    // Allocate data structures
    w = new double[M * K * F];
    w_inc = new double[M * K * F];
    vb = new double[M * K];
    vb_inc = new double[M * K];
    hb = new double[F];
    hb_inc = new double[F];

    if (conditional) {
        d = new double[M * F];
        d_inc = new double[M * F];
    }

    // Initialize weights and biases
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < F; j++) {
                w[_ikj(i, k, j)] = gaussian(0., 0.01);
            }
        }

        if (conditional) {
            for (int j = 0; j < F; j++){
                d[_ij(i, j)] = gaussian(0., 0.001);
            }
        }
    }

    // Zero out everything else
    zero(vb, M * K);
    zero(hb, F);
    zero(w_inc, M * K * F);
    zero(vb_inc, M * K);
    zero(hb_inc, F);

    if (conditional) {
        zero(d_inc, M * F);
    }
}

void RBMCF::update_hidden(double* vs, int* mask, int mask_size, double* hp) {
    // Compute p(h | V) into hp
    for (int j = 0; j < F; j++) {
        hp[j] = hb[j];
    }

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
    }
}

void RBMCF::update_hidden(double* vs, int* mask, int mask_size, int* seen, int seen_size, double* hp) {
    // Compute p(h | V) into hp
    for (int j = 0; j < F; j++) {
        hp[j] = hb[j];
    }

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

        int ij_0 = _ij(i, 0);

        for (int j = 0; j < F; j++) {
            hp[j] += d[ij_0 + j];
        }
    }

    for (int m = 0; m < seen_size; m++) {
        int i = seen[m];
        int ij_0 = _ij(i, 0);

        for (int j = 0; j < F; j++) {
            hp[j] += d[ij_0 + j];
        }
    }


    for (int j = 0; j < F; j++) {
        hp[j] = 1. / (1. + exp(-hp[j]));
    }
}

void RBMCF::sample_hidden(double* hp, double* hs) {
    // Sample from p(h | V, d) into hs
    for (int j = 0; j < F; j++) {
        hs[j] = (uniform() < hp[j]) ? 1. : 0.;
    }
}

void RBMCF::update_visible(double* hs, double* vp, int* mask, int mask_size) {
    // Compute p(V | h) into vp
    for (int m = 0; m < mask_size; m++) {
        int i = mask[m];
        int ik_0 = _ik(i, 0);
        double total = 0.;

        for (int k = 0; k < K; k++) {
            int ik = ik_0 + k;
            int ikj_0 = _ikj(i, k, 0);

            vp[ik] = vb[ik];

            for (int j = 0; j < F; j++) {
                // cout << "vp[ik]: " << vp[ik] << " hs[j]: " << hs[j] << " w[ikj_0 + j]: " << w[ikj_0 + j] << endl;
                vp[ik] += hs[j] * w[ikj_0 + j];
            }

            vp[ik] = exp(vp[ik]);
            total += vp[ik];
            // cout << "total: " <<total << " vp[ik]: " << vp[ik] << endl;
        }

        for (int k = 0; k < K; k++) {
            // cout << "update_visible: vp[ik_0 + k]: "<< vp[ik_0 + k] <<endl;
            // cout << "update_visible: total: "<< total <<endl;
            vp[ik_0 + k] /= total;
            // cout << "update_visible: res: "<< vp[ik_0 + k] <<endl;
        }
    }
}

void RBMCF::sample_visible(double* vp, double* vs, int* mask, int mask_size) {
    // Sample from p(V | h) into vs
    for (int m = 0; m < mask_size; m++) {
        int i = mask[m];
        int ik_0 = _ik(i, 0);

        double r = uniform();
        int k = 0;

        while (k < K && r > vp[ik_0 + k]) {
            vs[ik_0 + k] = 0.;
            r -= vp[ik_0 + k];
            k++;
        }

        if (k == K) { // Can this really happen?
            vs[ik_0 + k - 1] = 1.;
        } else {
            vs[ik_0 + k] = 1.;
            k++;

            for (; k < K; k++) {
                vs[ik_0 + k] = 0.;
            }
        }
    }
}

void RBMCF::update_w(double* w_acc, int* w_count, int nth) {
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

void RBMCF::update_vb(double* vb_acc, int* vb_count, int nth) {
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
            

            if (vb_count[ik] > 0) {
                vb_acc[ik] /= vb_count[ik];
            }

            vb_inc[ik] *= momentum;
            vb_inc[ik] += eps_vb * vb_acc[ik];
            vb[ik] += vb_inc[ik];
            
        }
    }
}

void RBMCF::update_hb(double* hb_acc, int nth) {
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

void RBMCF::update_d(double* d_acc, bool* watched, int nth) {
    // Pop parameters
    int batch_size = *(int*) getParameter("batch_size");
    double eps_d = *(double*) getParameter("eps_d");
    double momentum = *(double*) getParameter("momentum");
    bool annealing = *(bool*) getParameter("annealing");
    double annealing_rate = *(double*) getParameter("annealing_rate");

    // Update biases of hidden units
    if (annealing) {
        eps_d /= 1. + (nth / (annealing_rate * N));
    }

    for (int i = 0; i < M; i++) {
        if (watched[i]) {
            int ij_0 = _ij(i, 0);

            for (int j = 0; j < F; j++) {
                int ij = ij_0 + j;

                d_inc[ij] *= momentum;
                d_inc[ij] += eps_d * d_acc[j] / batch_size;
                d[ij] += d_inc[ij];
            }
        }
    }
}
