/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Implementation of "RBMMapReduce.h".
 *
 * - Author: LOUPPE Gilles
 * - Last changes: April 9, 2010
 * ========================================================================= */

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#include "Configuration.h"
#include "Dataset.h"
#include "Misc.h"
#include "RBMMapReduce.h"

#include <mpi.h>
#include "mapreduce.h"
#include "keyvalue.h"

using namespace std;
using namespace MAPREDUCE_NS;


/* ========================================================================= *
 * Macros
 * ========================================================================= */

#define _ikj(i, k, j) ((i) * K * F + (k) * F + (j))
#define _ik(i, k) ((i) * K + (k))
#define _ij(i, j) ((i) * F + (j))


/* ========================================================================= *
 * MapReduce RBM model
 * ========================================================================= */

/* ------------------------------------------------------------------------- *
 * Constructors
 * ------------------------------------------------------------------------- */

RBMMapReduce::RBMMapReduce(int rank, int np) :
    RBM(),
    rank(rank), np(np) {
        // Set default parameters
        setParameter("roundrobin", &Config::MapReduce::ROUNDROBIN, sizeof(int));
}

RBMMapReduce::RBMMapReduce(int rank, int np, string filename) :
    RBM(filename),
    rank(rank), np(np) {
        // Set default parameters
        setParameter("roundrobin", &Config::MapReduce::ROUNDROBIN, sizeof(int));
}

RBMMapReduce::~RBMMapReduce() {
    // Nothing more to do
}

/* ------------------------------------------------------------------------- *
 * Model
 * ------------------------------------------------------------------------- */

string RBMMapReduce::toString() {
    stringstream s;
    s << "MapReduce" << endl;
    s << RBM::toString() << endl;
    s << "Roundrobin = " << *(bool*) getParameter("roundrobin") << endl;

    return s.str();
}

/* ------------------------------------------------------------------------- *
 * Train over MapReduce
 * ------------------------------------------------------------------------- */

void RBMMapReduce::train(string dataset, bool reset) {
    // Pop parameters
    int epochs = *(int*) getParameter("epochs");
    bool roundrobin = *(bool*) getParameter("roundrobin");
    bool verbose = *(bool*) getParameter("verbose");
    ostream* out = *(ostream**) getParameter("log");
    __train_dataset = dataset;

    // Pop LS (+ QS)
    Dataset* LS = sets[dataset];
    Dataset* QS = sets["QS"];
    assert(LS != NULL);

    if (conditional) {
        assert(QS != NULL);
        assert(LS->nb_rows == QS->nb_rows);
    }

    // Reset parameters
    if (reset) {
        // Reset everything
        this->reset();

        // Set vb_ik to the logs of their respective base rates
        if (rank == 0) {
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

        // Broadcast the initial parameters
        MPI_Bcast(w, M * K * F, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(vb, M * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(hb, F, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (conditional) {
            MPI_Bcast(d, M * F, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    // Print some stats
    if (verbose) {
        double rmse = validate();

        if (rank == 0) {
            *out << toString() << endl;
            *out << "% ---" << endl;
            *out << "% Epoch\tRMSE" << endl;
            out->flush();
            *out << 0 << "\t" << rmse << endl;
            out->flush();
        }
    }

    // Loop through epochs
    int round = 1;

    for (__train_epoch = 1; __train_epoch <= epochs; __train_epoch++) {
        // Initialize MapReduce
        MapReduce* mr = new MapReduce(MPI_COMM_WORLD);
        mr->verbosity = 0;
        mr->timer = 0;
        mr->memsize = 64;

        // Sum
        mr->map(np, &train_map, this);

        // Collate
        mr->collate(NULL);

        // Average
        mr->reduce(&train_reduce, this);

        // Update weights and biases
        mr->gather(1);
        mr->map(mr, &train_update, this);

        if (roundrobin) {
            if (rank == 0) {
                MPI_Send(w, M * K * F, MPI_DOUBLE, round, 0, MPI_COMM_WORLD);
                MPI_Send(vb, M * K, MPI_DOUBLE, round, 0, MPI_COMM_WORLD);
                MPI_Send(hb, F, MPI_DOUBLE, round, 0, MPI_COMM_WORLD);

                if (conditional) {
                    MPI_Send(d, M * F, MPI_DOUBLE, round, 0, MPI_COMM_WORLD);
                }
            } else if (rank == round) {
                MPI_Status s;
                MPI_Recv(w, M * K * F, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &s);
                MPI_Recv(vb, M * K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &s);
                MPI_Recv(hb, F, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &s);

                if (conditional) {
                    MPI_Recv(d, M * F, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &s);
                }
            }

            round++;
            if (round >= np) round = 1;
        } else {
            MPI_Bcast(w, M * K * F, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(vb, M * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(hb, F, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (conditional) {
                MPI_Bcast(d, M * F, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
        }

        // Clear
        delete mr;

        // Log RMSE
        if (verbose) {
            double rmse = validate();

            if (rank == 0) {
                *out << __train_epoch << "\t" << rmse << endl;
                out->flush();
            }
        }

        // Synchronize
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void train_map(int task, KeyValue* kv, void* ptr) {
    // Pop parameters
    RBMMapReduce* r = (RBMMapReduce*) ptr;

    int M = r->M;
    int K = r->K;
    int F = r->F;

    bool conditional = r->conditional;
    int cd_steps = *(int*) r->getParameter("cd_steps");

    // Pop LS (+ QS)
    Dataset* LS = r->sets[r->__train_dataset];
    Dataset* QS = r->sets["QS"];
    assert(LS != NULL);

    if (conditional) {
        assert(QS != NULL);
        assert(LS->nb_rows == QS->nb_rows);
    }

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

    zero(vs, M * K);
    zero(vp, M * K);
    zero(hs, F);
    zero(hp, F);

    zero(w_acc, M * K * F);
    zero(w_count, M * K * F);
    zero(vb_acc, M * K);
    zero(vb_count, M * K);
    zero(hb_acc, F);

    // Loop through users
    int start = task * LS->nb_rows / r->np;
    int end = min((task + 1) * LS->nb_rows / r->np, LS->nb_rows);

    for (int n = start; n < end; n++) {
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
            r->update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], hp);
        } else {
            r->update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], &QS->ids[QS->index[n]], QS->count[n], hp);
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

        // Gibbs sampling
        for (int step = 0; step < cd_steps; step++) {
            // Sample from p(h | V, d) into hs
            r->sample_hidden(hp, hs);

            // Compute p(V | h) into vp
            r->update_visible(hs, vp, &LS->ids[LS->index[n]], LS->count[n]);

            // Sample from p(V | h) into hs
            r->sample_visible(vp, vs, &LS->ids[LS->index[n]], LS->count[n]);

            // Compute p(h | V, d) into hp
            if (!conditional) {
                r->update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], hp);
            } else {
                r->update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], &QS->ids[QS->index[n]], QS->count[n], hp);
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

    // Generate pairs
    r->__train_keys.clear();

    int key = 0;
    int size = 100 * K * F * (sizeof(double) + sizeof(int));
    char* data = new char[size];

    for (int i = 0; i < M; i += 100) {
        int offset = _ikj(i, 0, 0);
        int mul = min(100, M - i);

        memcpy(data, &w_acc[offset], mul * (K * F * sizeof(double)));
        memcpy(data + mul * K * F * sizeof(double), &w_count[offset], mul * (K * F * sizeof(int)));

        r->__train_keys[key] = make_pair(0, offset);
        kv->add((char*) &key, sizeof(int), data, mul * K * F * (sizeof(double) + sizeof(int)));
        key++;
    }

    for (int i = 0; i < M; i += 100 * F) {
        int offset = _ik(i, 0);
        int mul = min(100 * F, M - i);

        memcpy(data, &vb_acc[offset], mul * (K * sizeof(double)));
        memcpy(data + mul * (K * sizeof(double)), &vb_count[offset], mul * (K * sizeof(int)));

        r->__train_keys[key] = make_pair(1, offset);
        kv->add((char*) &key, sizeof(int), data, mul * K * (sizeof(double) + sizeof(int)));
        key++;
    }

    int nb = end - start;
    memcpy(data, hb_acc, F * sizeof(double));
    memcpy(data + F * sizeof(double), &nb, sizeof(int));

    r->__train_keys[key] = make_pair(2, 0);
    kv->add((char*) &key, sizeof(int), data, F * sizeof(double) + sizeof(int));
    key++;

    delete[] data;

    // Deallocate data structures
    delete[] vs;
    delete[] vp;
    delete[] hs;
    delete[] hp;

    delete[] w_acc;
    delete[] w_count;
    delete[] vb_acc;
    delete[] vb_count;
    delete[] hb_acc;
}

void train_reduce(char* key, int key_size,
                  char* values, int nb_values, int* value_sizes,
                  KeyValue* kv, void* ptr) {
    // Pop parameters
    assert(values != NULL);
    RBMMapReduce* r = (RBMMapReduce*) ptr;

    int K = r->K;
    int F = r->F;

    // Reduce data
    int id = *(int*) key;

    if (r->__train_keys[id].first == 0 || r->__train_keys[id].first == 1){
        double* sum = new double[100 * K * F];
        int* count = new int[100 * K * F];

        zero(sum, 100 * K * F);
        zero(count, 100 * K * F);

        for (int i = 0; i < nb_values; i++) {
            int length = value_sizes[i] / (sizeof(double) + sizeof(int));

            double* _sum = (double*) values;
            int* _count = (int*) (values + length * sizeof(double));

            for (int j = 0; j < length; j++) {
                sum[j] += _sum[j];
                count[j] += _count[j];
            }

            values += value_sizes[i];
        }

        for (int i = 0; i < 100 * K * F; i++) {
            if (count[i] != 0) {
                sum[i] /= count[i];
            } else {
                sum[i] = 0.;
            }
        }

        kv->add(key, key_size, (char*) sum, 100 * K * F * sizeof(double));

        delete[] sum;
        delete[] count;
    } else {       
        double* sum = new double[F];
        int nb;

        zero(sum, F);
        nb = 0;

        for (int i = 0; i < nb_values; i++) {
            double* _sum = (double*) values;
            int _nb = *(int*) (values + F * sizeof(double));

            for (int j = 0; j < F; j++) {
                sum[j] += _sum[j];
            }

            nb += _nb;
            values += value_sizes[i];
        }

        for (int i = 0; i < F; i++) {
            if (nb != 0) {
                sum[i] /= nb;
            } else {
                sum[i] = 0.;
            }
        }

        kv->add(key, key_size, (char*) sum, F * sizeof(double));

        delete[] sum;
    }
}

void train_update(uint64_t i,
                  char* key, int key_size,
                  char* value, int value_size,
                  KeyValue* kv, void* ptr){
    // Pop parameters
    RBMMapReduce* r = (RBMMapReduce*) ptr;
    if (r->rank != 0) return;

    int M = r->M;
    int K = r->K;
    int F = r->F;

    double eps_w = *(double*) r->getParameter("eps_w");
    double eps_vb = *(double*) r->getParameter("eps_vb");
    double eps_hb = *(double*) r->getParameter("eps_hb");
    double eps_d = *(double*) r->getParameter("eps_d");
    double weight_cost = *(double*) r->getParameter("weight_cost");
    double momentum = *(double*) r->getParameter("momentum");
    bool annealing = *(bool*) r->getParameter("annealing");
    double annealing_rate = *(double*) r->getParameter("annealing_rate");

    // Update weights and biases
    int id = *(int*) key;

    if (r->__train_keys[id].first == 0) {
        double* data = (double*) value;
        int size = value_size / sizeof(double);
        int offset = r->__train_keys[id].second;

        if (offset + size > M * K * F) {
            size = M * K * F - offset;
        }

        if (annealing) {
            eps_w /= 1. + (r->__train_epoch / annealing_rate);
        }

        for (int i = 0; i < size; i++) {
            r->w_inc[offset + i] *= momentum;
            r->w_inc[offset + i] += eps_w * (data[i] - weight_cost * r->w[offset + i]);
            r->w[offset + i] += r->w_inc[offset + i];
        }
    } else if (r->__train_keys[id].first == 1) {
        double* data = (double*) value;
        int size = value_size / sizeof(double);
        int offset = r->__train_keys[id].second;

        if (offset + size > M * K) {
            size = M * K - offset;
        }

        if (annealing) {
            eps_vb /= 1. + (r->__train_epoch / annealing_rate);
        }

        for (int i = 0; i < size; i++) {
            r->vb_inc[offset + i] *= momentum;
            r->vb_inc[offset + i] += eps_vb * data[i];
            r->vb[offset + i] += r->vb_inc[offset + i];
        }
    } else {
        double* data = (double*) value;

        if (annealing) {
            eps_hb /= 1. + (r->__train_epoch / annealing_rate);
            eps_d /= 1. + (r->__train_epoch / annealing_rate);
        }

        for (int j = 0; j < F; j++) {
            r->hb_inc[j] *= momentum;
            r->hb_inc[j] += eps_hb * data[j];
            r->hb[j] += r->hb_inc[j];
        }

        if (r->conditional) {
            for (int i = 0; i < M; i++) {
                int ij_0 = _ij(i, 0);

                for (int j = 0; j < F; j++) {
                    int ij = ij_0 + j;

                    r->d_inc[ij] *= momentum;
                    r->d_inc[ij] += eps_d * data[j];
                    r->d[ij] += r->d_inc[ij];
                }
            }
        }
    }
}

/* ------------------------------------------------------------------------- *
 * Test over MapReduce
 * ------------------------------------------------------------------------- */

double RBMMapReduce::test(string dataset) {
    // Set parameters
    __test_dataset = dataset;
    __test_total = 0.;
    __test_count = 0;

    // Initialize MapReduce
    MapReduce* mr = new MapReduce(MPI_COMM_WORLD);
    mr->verbosity = 0;
    mr->timer = 0;
    mr->memsize = 64;

    // Map
    mr->map(np, &test_map, this);

    // Reduce
    mr->gather(1);
    mr->map(mr, &test_reduce, this);

    double rmse = 0;

    if (rank == 0) {
        rmse = sqrt(__test_total / __test_count);
    }

    // Clear
    delete mr;

    return rmse;
}

void test_map(int task, KeyValue* kv, void* ptr) {
    // Pop parameters
    RBMMapReduce* r = (RBMMapReduce*) ptr;

    int M = r->M;
    int K = r->K;
    int F = r->F;

    bool conditional = r->conditional;

    // Pop LS, QS and TS
    Dataset* LS = r->sets["LS"];
    Dataset* QS = r->sets["QS"];
    Dataset* TS = r->sets[r->__test_dataset];
    assert(LS != NULL);
    assert(TS != NULL);
    assert(LS->nb_rows == TS->nb_rows);

    if (conditional) {
        assert(QS != NULL);
        assert(LS->nb_rows == QS->nb_rows);
    }

    // Initialization
    double total_error = 0.;
    int count = 0;

    // Allocate local data structures
    double* vs = new double[M * K];
    double* vp = new double[M * K];
    double* hs = new double[F];
    double* hp = new double[F];

    // Loop through users in the test set
    int start = task * TS->nb_rows / r->np;
    int end = min((task + 1) * TS->nb_rows / r->np, TS->nb_rows);

    for (int n = start; n < end; n++) {
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
            r->update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], hp);
        } else {
            r->update_hidden(vs, &LS->ids[LS->index[n]], LS->count[n], &QS->ids[QS->index[n]], QS->count[n], hp);
        }

        // Compute p(v_ik = 1 | ^p) for all movie i in TS
        r->update_visible(hp, vp, &TS->ids[TS->index[n]], TS->count[n]);

        // Predict ratings
        for (int m = TS->index[n]; m < TS->index[n] + TS->count[n]; m++) {
            int i = TS->ids[m];
            int ik_0 = _ik(i, 0);
            double prediction = 0.;

            for (int k = 0; k < K; k++) {
                prediction += vp[ik_0 + k] * (k + 1);
            }

            double error = prediction - TS->ratings[m];
            total_error += error * error;
            count++;
        }
    }

    // Deallocate data structure
    delete[] vs;
    delete[] vp;
    delete[] hs;
    delete[] hp;

    // Generate pair
    int key = 0;
    int size = sizeof(double) + sizeof(int);
    char* data = new char[size];

    memcpy(data, &total_error, sizeof(double));
    memcpy(data + sizeof(double), &count, sizeof(int));

    kv->add((char*) &key, sizeof(int), data, size);

    delete[] data;
}

void test_reduce(uint64_t i,
                 char* key, int key_size,
                 char* value, int value_size,
                 KeyValue* kv, void* ptr){
    // Pop parameters
    RBMMapReduce* r = (RBMMapReduce*) ptr;
    if (r->rank != 0) return;

    // Reduce RMSE
    r->__test_total += *(double*) value;
    r->__test_count += *(int*) (value + sizeof(double));
}
