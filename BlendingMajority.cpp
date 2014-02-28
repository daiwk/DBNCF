/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Blending
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
    int* count = new int[VS.nb_ratings * 5];

    for (int r = 0; r < VS.nb_ratings * 5; r++){
        count[r] = 0;
    }

    // Load predictions
    argv++;
    argc--;
    double* predictions = new double[VS.nb_ratings];

    for (int i = 0; i < argc; i++){
        ifstream in(argv[i], ios::in | ios::binary);
        // in.read((char*) predictions, VS.nb_ratings * sizeof (double));
        for (int i = 0; i < VS.nb_ratings; i++) 
	    in >> predictions[i];
        in.close();

        for (int r = 0; r < VS.nb_ratings; r++){
            if (predictions[r] > 0.){
                int p = (int) (floor(predictions[r] + 0.5) - 1);
                count[r * 5 + p]++;
            }
        }
    }

    // RMSE
    double error = 0.;

    for (int r = 0; r < VS.nb_ratings; r++) {
        if ((count[r * 5] + count[r * 5 + 1] + count[r * 5 + 2] + count[r * 5 + 3] + count[r * 5 + 4]) <= 0){
            double e = VS.ratings[r] - 3.6033;
            error += e * e;
        } else {
            int majority_p = 1;
            int majority_count = count[r * 5];

            for (int i = 0; i < 5; i++){
                if (count[r * 5 + i] >= majority_count) {
                    majority_p = i + 1;
                    majority_count = count[r * 5 + i];
                }
            }

            double e = VS.ratings[r] - majority_p;
            error += e * e;
        }
    }

    double rmse = sqrt(error / VS.nb_ratings);
    cout << "RMSE = " << rmse << endl;

    // Cleanup
    delete count;
    delete predictions;

    return EXIT_SUCCESS;
}
