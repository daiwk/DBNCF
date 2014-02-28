/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Implementation of "Misc.h"
 *
 * - Author: LOUPPE Gilles
 * - Last changes: February 25, 2010
 * ========================================================================= */

#include <cmath>
#include <cstdlib>
#include <ctime>

#include "Misc.h"

using namespace std;


/* ========================================================================= *
 * Stats
 * ========================================================================= */

double uniform(void) {
    static bool seeded = false;
    
    if (!seeded){
        srand(time(NULL));
        seeded = true;
    }

    return rand() / (double) RAND_MAX;
}

double noise(void) {
    double a = 0.;
    double b = 0.;

    while (a * b == 0.) {
        a = uniform();
        b = uniform();   
    }

    return sqrt(-2. * log(a)) * cos(6.2831853071795862 * b);
}

double gaussian(double mean, double stddev) {
    return 1. * mean + stddev * noise();
}
