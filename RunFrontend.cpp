/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Frontend.
 *
 * - Author: LOUPPE Gilles
 * - Last changes: March 1, 2010
 * ========================================================================= */

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "Configuration.h"
#include "Dataset.h"
#include "Dumb.h"

using namespace std;


/* ========================================================================= *
 * main
 * ========================================================================= */

int main(int argc, char** argv) {
    // Load data
    Dataset LS("../../data/bin/full_users.bin");
    Dataset movies("../../data/bin/full_movies.bin");
    map<int, string> titles = load_titles("../../data/text/movies.txt");

    // Train some model
    Model* model = new Dumb3();
    model->addSet("LS", &LS);
    model->train();

    // Display some stats about user
    int user = atoi(argv[1]);

    cout << "===========================================================" 
         << endl << endl
         << "User ID = " << user << endl
         << "Number of rated movies = " << LS.count[user] << endl;

    double total = 0.;

    for (int m = LS.index[user]; m < LS.index[user] + LS.count[user]; m++) {
        total += LS.ratings[m];
    }

    cout << "Mean rating = " << total / LS.count[user] << endl;

    // Display movies seen by that user
    cout << endl 
         << "-----------------------------------------------------------"
         << endl << endl
         << "Rated movies: "
         << endl << endl;

    vector< pair<double, int> > ratings;
    set<int> seen;

    for (int m = LS.index[user]; m < LS.index[user] + LS.count[user]; m++) {
        ratings.push_back(make_pair(LS.ratings[m], LS.ids[m]));
        seen.insert(LS.ids[m]);
    }

    sort(ratings.rbegin(), ratings.rend());
    vector< pair<double, int> >::iterator it;

    for (it = ratings.begin(); it != ratings.end(); it++){
        cout << setw(10) << (*it).first
             << setw(10) << (*it).second
             << setw(10) << movies.count[(*it).second]
             << "     " << titles[(*it).second] << endl; 
    }

    // Display 100 first recommendations
    cout << endl 
         << "-----------------------------------------------------------"
         << endl << endl
         << "100 first recommendations: "
         << endl << endl;

    vector< pair<double, int> > recommendations;

    for (int m = 0; m < Config::Netflix::NB_MOVIES; m++){
        if (seen.find(m) == seen.end()){
            recommendations.push_back(make_pair(model->predict(user, m), m));
        }
    }

    sort(recommendations.rbegin(), recommendations.rend());
    int i = 1;

    for (it = recommendations.begin(); it != recommendations.end(); it++){
        cout << setw(10) << (*it).first
             << setw(10) << (*it).second
             << setw(10) << movies.count[(*it).second]
             << "     " << titles[(*it).second] << endl;

        if (i++ >= 100) break;
    }

    cout << endl 
         << "==========================================================="
         << endl;

    // Cleanup
    delete model;
    
    return EXIT_SUCCESS;
}
