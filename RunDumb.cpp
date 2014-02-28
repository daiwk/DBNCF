/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Example of Dumb models.
 *
 * - Author: LOUPPE Gilles
 * - Last changes: March 1, 2010
 * ========================================================================= */

#include <cstdlib>
#include <iostream>

#include "Dataset.h"
#include "Dumb.h"
#include "Ensemble.h"

using namespace std;


/* ========================================================================= *
 * main
 * ========================================================================= */

int main(int argc, char** argv) {
    // Datasets
    Dataset LS("../../data/bin/full_users.bin");
    Dataset TS("../../data/bin/TS.bin");

    // Build models
    Dumb1 d1;
    d1.addSet("LS", &LS);
    d1.addSet("TS", &TS);

    Dumb2 d2;
    d2.addSet("LS", &LS);
    d2.addSet("TS", &TS);

    Dumb3 d3;
    d3.addSet("LS", &LS);
    d3.addSet("TS", &TS);

    // Train + Test
    d1.train();
    d2.train();
    d3.train();

    cout << "Dumb1 RMSE = " << d1.test() << endl;
    cout << "Dumb2 RMSE = " << d2.test() << endl;
    cout << "Dumb3 RMSE = " << d3.test() << endl;

    // Ensemble
    Ensemble e;
    e.addModel(&d2);
    e.addModel(&d3);
    e.addSet("LS", &LS);
    e.addSet("TS", &TS);
    e.train();

    cout << "Ensemble RMSE = " << e.test() << endl;

    return EXIT_SUCCESS;
}


