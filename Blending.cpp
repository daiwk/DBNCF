/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Blending.
 *
 * - Author: LOUPPE Gilles
 * - Last changes: May 27, 2010
 * ========================================================================= */

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "Configuration.h"
#include "Dataset.h"

using namespace std;


/* ========================================================================= *
 * main
 * ========================================================================= */

int main(int argc, char** argv) {
    // Load datasets
    Dataset VS(Config::Sets::VS);

    // Initialization
    double* total = new double[VS.nb_ratings];
    int* count = new int[VS.nb_ratings];

    for (int r = 0; r < VS.nb_ratings; r++){
        total[r] = 0.;
        count[r] = 0;
    }

    // Load predictions
    argv++;
    argc--;
    double* predictions = new double[VS.nb_ratings];

    for (int i = 0; i < argc; i++){
        ifstream in(argv[i], ios::in | ios::binary);
        // in.read((char*) predictions, VS.nb_ratings * sizeof (double));
	for (int i = 0; i < VS.nb_ratings; i++ )
		in >> predictions[i];
        in.close();

        for (int r = 0; r < VS.nb_ratings; r++){
            if (predictions[r] > 0.){
                total[r] += predictions[r];
                count[r]++;
            }
        }
    }

    // RMSE
    double error = 0.;

    for (int r = 0; r < VS.nb_ratings; r++) {
        if (count[r] <= 0){
        total[r] = 3.6033;
        count[r] = 1;
    }

        double e = VS.ratings[r] - (total[r]/count[r]);
        error += e * e;
    }

    double rmse = sqrt(error / VS.nb_ratings);
    cout << "RMSE = " << rmse << endl;

    // Cleanup
    delete total;
    delete count;
    delete predictions;

    return EXIT_SUCCESS;
}
