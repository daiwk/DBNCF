/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Miscellaneous.
 *
 * - Author: LOUPPE Gilles
 * - Last changes: February 25, 2010
 * ========================================================================= */

#ifndef _MISC_H_
#define _MISC_H_

#include <cstdlib>
#include <cstring>

using namespace std;


/* ========================================================================= *
 * Comparators
 * ========================================================================= */

#define max(a, b) ((a) < (b) ? (b) : (a))
#define min(a, b) ((a) < (b) ? (a) : (b))


/* ========================================================================= *
 * Arrays
 * ========================================================================= */

/* ------------------------------------------------------------------------- *
 * Fills the given array with zeroes.
 *
 * ARGUMENTS:
 * - array: the array
 * - size: the array size
 * ------------------------------------------------------------------------- */

template <typename T> void zero(T* array, int size) {
    memset(array, 0, size * sizeof (T));
}


/* ========================================================================= *
 * Stats
 * ========================================================================= */

/* ------------------------------------------------------------------------- *
 * Generates a pseudo-random number between 0 and 1, according to a uniform
 * distribution.
 *
 * RETURN:
 * A pseudo-random number between 0 and 1.
 * ------------------------------------------------------------------------- */

double uniform(void);

/* ------------------------------------------------------------------------- *
 * Generates gaussian noise, i.e., a pseudo-random number according to a
 * gaussian distribution of mean=0 and stddev=1.
 *
 * RETURN:
 * A pseudo-random number according to N(0, 1).
 * ------------------------------------------------------------------------- */

double noise(void);

/* ------------------------------------------------------------------------- *
 * Generates a pseudo-random number according to a gaussian distribution.
 *
 * ARGUMENTS:
 * - mean: the mean of the gaussian distribution
 * - stddev: the standard deviation of the gaussian distribution
 *
 * RETURN:
 * A pseudo-random number according to N(mean, stddev^2).
 * ------------------------------------------------------------------------- */

double gaussian(double mean, double stddev);

#endif
