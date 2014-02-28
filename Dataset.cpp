/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Implementation of "Dataset.h".
 *
 * - Author: LOUPPE Gilles
 * - Last changes: April 15, 2010
 * ========================================================================= */

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include "Configuration.h"
#include "Dataset.h"
#include "Misc.h"

using namespace std;
using namespace Config::Netflix;


/* ========================================================================= *
 * Dataset
 * ========================================================================= */

Dataset::Dataset() :
    nb_rows(0),
    nb_columns(0),
    nb_ratings(0),
    index(NULL),
    count(NULL),
    ids(NULL),
    ratings(NULL),
    rows(),
    users(),
    movies() {
    // Nothing more to do
}

Dataset::Dataset(string filename) {
    // Open
    ifstream in(filename.c_str(), ios::in | ios::binary);
    cout << "loading " << filename << endl;

    if (in.fail()) {
        throw runtime_error("I/O error!");
    }

    // Load headers
    // in.read((char*) &nb_rows, sizeof (int));
    in >> nb_rows;
    // cout << "nb_rows:" << nb_rows <<endl;
    // in.read((char*) &nb_columns, sizeof (int));
    in >> nb_columns;
    // cout << "nb_columns: " << nb_columns <<endl;
    // in.read((char*) &nb_ratings, sizeof (int));
    in >> nb_ratings;
    // cout << "nb_ratings:" << nb_ratings <<endl;

    // Allocate arrays
    index = new int[nb_rows];
    count = new int[nb_rows];
    ids = new int[nb_ratings];
    // ratings = new char[nb_ratings];
    ratings = new int[nb_ratings];

    // Load arrays
    // in.read((char*) index, nb_rows * sizeof (int));
    // cout << "reading indices..." <<endl; 
    for (int i = 0; i < nb_rows; i++) {
	in >> index[i];
	// cout << "i: " << i << "; index: " << index[i] <<endl;
    }

    //in.read((char*) count, nb_rows * sizeof (int));
    // cout << "reading count..." <<endl; 
    for (int i = 0; i < nb_rows; i++) {
	in >> count[i];
	// cout << "i: " << i << "; count: " << count[i] <<endl;
    }
    // in.read((char*) ids, nb_ratings * sizeof (int));
    // cout << "reading ids..." <<endl; 
    for (int i = 0; i < nb_ratings; i++) {
	in >> ids[i];
	// cout << "i: " << i << "; ids: " << ids[i] <<endl;
    }
    // in.read((char*) ratings, nb_ratings * sizeof (char));
    // cout << "reading ratings..." <<endl; 
    for (int i = 0; i < nb_ratings; i++) {
	in >> ratings[i];
	// cout << "i: " << i << "; ratings: " << ratings[i] <<endl;
    }
    // Load rows + users
    int u;

    for (int r = 0; r < nb_rows; r++) {
        // in.read((char*) &u, sizeof (int));
        in >> u;
	// cout << "u: " << u << endl;
	// cout << "r:" << r << endl;
        rows[r] = u;
        users[u] = r;
    }

    // Load movies
    int m;

    for (int c = 0; c < nb_columns; c++){
        // in.read((char*) &m, sizeof (int));
	in >> m;
	// cout << "m: " << m << endl;
        movies.insert(m);
    }

    // Close
    in.close();
}

Dataset::~Dataset() {
    delete[] index;
    delete[] count;
    delete[] ids;
    delete[] ratings;
}

bool Dataset::contains_movie(int m) {
    return movies.find(m) != movies.end();
}

bool Dataset::contains_user(int u) {
    return users.find(u) != users.end();
}

Dataset* Dataset::pick_movies(set<int> sample) {
    // Initialization
    Dataset* s = new Dataset();

    s->nb_rows = nb_rows;
    s->nb_columns = sample.size();
    s->nb_ratings = 0;

    s->index = new int[nb_rows];
    s->count = new int[nb_rows];
    s->ids = new int[nb_ratings];
    // s->ratings = new char[nb_ratings];
    s->ratings = new int[nb_ratings];

    s->rows = rows;
    s->users = users;
    s->movies = sample;

    // Keep the ratings of selected movies
    int index_r = 0;

    for (int r = 0; r < nb_rows; r++){
        int i = 0;
        s->index[r] = index_r;

        for (int m = index[r]; m < index[r] + count[r]; m++){
            if (s->contains_movie(ids[m])){
                s->ids[index_r + i] = ids[m];
                s->ratings[index_r + i] = ratings[m];
                i++;
            }
        }

        s->count[r] = i;
        index_r += i;
    }

    s->nb_ratings = index_r;

    // Realloc arrays
    int* tmp1 = new int[s->nb_ratings];

    for (int r = 0; r < s->nb_ratings; r++){
        tmp1[r] = s->ids[r];
    }

    delete[] s->ids;
    s->ids = tmp1;

    // char* tmp2 = new char[s->nb_ratings];
    int* tmp2 = new int[s->nb_ratings];

    for (int r = 0; r < s->nb_ratings; r++) {
        tmp2[r] = s->ratings[r];
    }

    delete[] s->ratings;
    s->ratings = tmp2;

    // Return
    return s;
}

Dataset* Dataset::pick_users(set<int> sample) {
    // Initalization
    Dataset* s = new Dataset();

    s->nb_rows = sample.size();
    s->nb_columns = 0;
    s->nb_ratings = 0;

    s->index = new int[sample.size()];
    s->count = new int[sample.size()];
    s->ids = new int[nb_ratings];
    // s->ratings = new char[nb_ratings];
    s->ratings = new int[nb_ratings];

    s->rows.clear();
    s->users.clear();
    s->movies.clear();

    // Keep the ratings of the selected users
    set<int>::iterator it;

    int i = 0;
    int index_i = 0;

    for (it = sample.begin(); it != sample.end(); it++){
        if (contains_user(*it)){
            int r = users[*it];

            s->index[i] = index_i;
            s->count[i] = count[r];

            for (int j = 0; j < count[r]; j++){
                s->ids[index_i + j] = ids[index[r] + j];
                s->ratings[index_i + j] = ratings[index[r] + j];
                s->movies.insert(ids[index[r] + j]);
            }


            index_i += count[r];
        } else {
            s->index[i] = -1;
            s->count[i] = 0;
        }

        s->rows[i] = *it;
        s->users[*it] = i;

        i++;
    }

    s->nb_columns = s->movies.size();
    s->nb_ratings = index_i;

    // Realloc arrays
    int* tmp1 = new int[s->nb_ratings];

    for (int r = 0; r < s->nb_ratings; r++){
        tmp1[r] = s->ids[r];
    }

    delete[] s->ids;
    s->ids = tmp1;

    // char* tmp2 = new char[s->nb_ratings];
    int* tmp2 = new int[s->nb_ratings];

    for (int r = 0; r < s->nb_ratings; r++) {
        tmp2[r] = s->ratings[r];
    }

    delete[] s->ratings;
    s->ratings = tmp2;

    // Return
    return s;
}

void Dataset::save(string filename) {
    // Open
    ofstream out(filename.c_str(), ios::out | ios::binary);

    // Write headers
    out.write((char*) &nb_rows, sizeof (int));
    out.write((char*) &nb_columns, sizeof (int));
    out.write((char*) &nb_ratings, sizeof (int));

    // Write arrays
    out.write((char*) index, nb_rows * sizeof (int));
    out.write((char*) count, nb_rows * sizeof (int));
    out.write((char*) ids, nb_ratings * sizeof (int));
    // out.write((char*) ratings, nb_ratings * sizeof (char));
    out.write((char*) ratings, nb_ratings * sizeof (int));

    // Write users
    int u;

    for (int r = 0; r < nb_rows; r++) {
        u = rows[r];
        out.write((char*) &u, sizeof (int));
    }

    // Write movies
    set<int>::iterator it;
    int m;

    for (it = movies.begin(); it != movies.end(); it++){
        m = *it;
        out.write((char*) &m, sizeof (int));
    }

    // Close
    out.close();
}

void Dataset::split_ratings(Dataset* s1, Dataset* s2, double p) {
    // Allocate data structures
    s1->nb_rows = nb_rows;
    s1->nb_columns = nb_columns;
    s1->nb_ratings = 0;
    s1->index = new int[nb_rows];
    s1->count = new int[nb_rows];
    s1->ids = new int[nb_ratings];
    // s1->ratings = new char[nb_ratings];
    s1->ratings = new int[nb_ratings];
    s1->rows = rows;
    s1->users = users;
    s1->movies.clear();

    s2->nb_rows = nb_rows;
    s2->nb_columns = nb_columns;
    s2->nb_ratings = 0;
    s2->index = new int[nb_rows];
    s2->count = new int[nb_rows];
    s2->ids = new int[nb_ratings];
    // s2->ratings = new char[nb_ratings];
    s2->ratings = new int[nb_ratings];
    s2->rows = rows;
    s2->users = users;
    s2->movies.clear();

    // Split current set of ratings into two subsets
    int index1 = 0;
    int count1 = 0;

    int index2 = 0;
    int count2 = 0;

    for (int n = 0; n < nb_rows; n++) {
        s1->index[n] = index1;
        count1 = 0;

        s2->index[n] = index2;
        count2 = 0;

        for (int m = index[n]; m < index[n] + count[n]; m++) {
            if (uniform() < p) {
                s1->ids[index1 + count1] = ids[m];
                s1->ratings[index1 + count1] = ratings[m];
                s1->movies.insert(ids[m]);
                count1++;
            } else {
                s2->ids[index2 + count2] = ids[m];
                s2->ratings[index2 + count2] = ratings[m];
                s2->movies.insert(ids[m]);
                count2++;
            }
        }

        index1 += count1;
        s1->count[n] = count1;
        s1->nb_ratings += count1;

        index2 += count2;
        s2->count[n] = count2;
        s2->nb_ratings += count2;
    }

    s1->nb_columns = s1->movies.size();
    s2->nb_columns = s2->movies.size();
}


/* ========================================================================= *
 * Functions
 * ========================================================================= */

map<int, string> load_titles(string filename) {
    // Open
    ifstream in(filename.c_str(), ios::in | ios::binary);
    cout << "load_titles: " << filename <<endl;
    // Fill map
    map<int, string> titles;
    int id;
    char* buffer = new char[1024];

    cout << "NB_MOVIES: " << NB_MOVIES << endl;
    for (int i = 0; i < NB_MOVIES; i++){
        in >> id;
        in.getline(buffer, 1024);
        cout <<"id: " << id << endl;
        titles[id] = buffer + 6;
	cout << "title: " << titles[id] << endl;
    }

    // Cleanup
    delete[] buffer;
    in.close();

    // Return
    return titles;
}

set<int> random_movies(int nb) {
    set<int> sample;

    while ((int) sample.size() < nb) {
        sample.insert((int) (uniform() * NB_MOVIES));
    }

    return sample;
}

set<int> random_users(int nb) {
    set<int> sample;

    while ((int) sample.size() < nb) {
        sample.insert((int) (uniform() * NB_USERS));
    }

    return sample;
}
