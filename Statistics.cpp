/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Some statistics.
 *
 * - Author: LOUPPE Gilles
 * - Last changes: May 7, 2010
 * ========================================================================= */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "Dataset.h"

using namespace std;


/* ========================================================================= *
 * main
 * ========================================================================= */

int main(int argc, char** argv) {
    // Load data
    Dataset users("../../data/bin/full_users.bin");
    Dataset movies("../../data/bin/full_movies.bin");
    map<int, string> titles = load_titles("../../data/text/movies.txt");

    // Mean and variance (movies)
    /*
    int total_r = 0;
    int total_rr = 0;

    for (int r = 0; r < movies.nb_ratings; r++){
        total_r += movies.ratings[r];
        total_rr += (movies.ratings[r] * movies.ratings[r]);
    }

    cout << "Mean rating = " << 1. * total_r / movies.nb_ratings << endl;
    cout << "Std. deviation = " << sqrt(1 * total_rr / movies.nb_ratings - (1. * total_r / movies.nb_ratings) * (1. * total_r / movies.nb_ratings)) << endl;
    cout << "Mean number of ratings / movie = " << 1. * movies.nb_ratings / movies.nb_rows << endl;
    cout << endl;
    */

    // Number of ratings per movie
    /*
    vector<int> counts;

    for (int m = 0; m < movies.nb_rows; m++) {
        counts.push_back(movies.count[m]);
    }

    sort(counts.begin(), counts.end());
    vector<int>::reverse_iterator it;

    for (it = counts.rbegin(); it != counts.rend(); it++) {
        cout << *it << endl;
    }
    */

    // Number of movies with average rating of
    /*
    int buckets[51];
    for (int i = 0; i <= 50; i++) buckets[i] = 0;

    for (int m = 0; m < movies.nb_rows; m++){
        int total_r = 0;

        for (int i = movies.index[m]; i < movies.index[m] + movies.count[m]; i++){
            total_r += movies.ratings[i];
        }

        double mean = 1. * total_r / movies.count[m];
        buckets[(int) (mean * 10)]++;
    }

    for (int i = 0; i <= 50; i++) {
        cout << 1. * i / 10 << "\t" << buckets[i] << endl;
    }

    cout << endl;
    */

    // Mean and variance (users)
    /*
    cout << "Mean number of ratings / users = " << 1. * users.nb_ratings / users.nb_rows << endl;
    cout << endl;
    */

    // Number of users with average rating of
    /*
    for (int i = 0; i <= 50; i++) buckets[i] = 0;

    for (int n = 0; n < users.nb_rows; n++){
        int total_r = 0;

        for (int i = users.index[n]; i < users.index[n] + users.count[n]; i++){
            total_r += users.ratings[i];
        }

        double mean = 1. * total_r / users.count[n];
        buckets[(int) (mean * 10)]++;
    }

    for (int i = 0; i <= 50; i++) {
        cout << 1. * i / 10 << "\t" << buckets[i] << endl;
    }

    cout << endl;
    */

    // Number of ratings per users
    vector<int> counts;

    for (int n = 0; n < users.nb_rows; n++) {
        counts.push_back(users.count[n]);
	cout << "count: " << n << endl;
    }

    sort(counts.begin(), counts.end());

    for (int i = counts.size() - 1; i >= 0; i -= 16) {
        cout << counts[i] << endl;
    }

    // Average std of users with average ratings of
    /*
    int buckets_count[51];
    double buckets_std[51];

    for (int i = 0; i <= 50; i++) buckets_count[i] = 0;
    for (int i = 0; i <= 50; i++) buckets_std[i] = 0.;

    for (int n = 0; n < users.nb_rows; n++){
        int total_r = 0;
        int total_rr = 0;

        for (int i = users.index[n]; i < users.index[n] + users.count[n]; i++){
            total_r += users.ratings[i];
            total_rr += users.ratings[i] * users.ratings[i];
        }

        double mean = 1. * total_r / users.count[n];
        double std = sqrt(1. * total_rr / users.count[n] - (1. * total_r / users.count[n]) * (1. * total_r / users.count[n]));

        buckets_count[(int) (mean * 10)]++;
        buckets_std[(int) (mean * 10)] += std;
    }

    for (int i = 0; i <= 50; i++) {
        cout << 1. * i / 10 << "\t" << buckets_std[i] / buckets_count[i] << endl;
    }

    cout << endl;
   */ 

    // Exit
    return EXIT_SUCCESS;
}
