/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Utilities for handling the Netflix data.
 *
 * - Author: LOUPPE Gilles
 * - Last changes: April 15, 2010
 * ========================================================================= */

#ifndef _DATASET_H_
#define _DATASET_H_

#include <map>
#include <set>
#include <string>

using namespace std;


/* ========================================================================= *
 * Classes
 * ========================================================================= */

class Dataset {
public:
    // Attributes
    int nb_rows;           // Number of rows
    int nb_columns;        // Number of columns (at most)
    int nb_ratings;        // Total number of non-missing ratings

    int* index;            // row ID -> index of the first entry in ids
    int* count;            // row ID -> number of ratings in that row
    int* ids;              // Store the ids of non-missing elements in all rows
    // char* ratings;         // Store the corresponding ratings
    int* ratings;         // Store the corresponding ratings

    map<int, int> rows;    // internal row ID -> user ID
    map<int, int> users;   // user ID -> internal row ID
    set<int> movies;       // movie IDs

    // Constructors
    /* --------------------------------------------------------------------- *
     * Creates a new empty dataset.
     * --------------------------------------------------------------------- */
    Dataset();

    /* --------------------------------------------------------------------- *
     * Loads a dataset.
     *
     * ARGUMENTS:
     * - filename: the dataset to be loaded
     * --------------------------------------------------------------------- */
    Dataset(string filename);

    /* --------------------------------------------------------------------- *
     * Destroys this dataset.
     * --------------------------------------------------------------------- */
    virtual ~Dataset();

    // Methods
    /* --------------------------------------------------------------------- *
     * Returns whether this dataset contains any rating for the specified
     * movie.
     *
     * ARGUMENTS:
     * - m the movie id
     *
     * RETURN:
     * True if this dataset contains the given movie, and false otherwise.
     * --------------------------------------------------------------------- */
    virtual bool contains_movie(int m);

    /* --------------------------------------------------------------------- *
     * Returns whether this dataset contains any rating for the specified
     * user.
     *
     * ARGUMENTS:
     * - u the user id
     *
     * RETURN:
     * True if this dataset contains the given user, and false otherwise.
     * --------------------------------------------------------------------- */
    virtual bool contains_user(int u);

    /* --------------------------------------------------------------------- *
     * Builds a subset S of this dataset such that
     *
     *     S = {(u,m,r) | m in sample, (u,m,r) in this dataset}
     *
     * ARGUMENTS:
     * - sample: the subset of movies
     *
     * RETURN:
     * A subset of this dataset.
     * --------------------------------------------------------------------- */
    virtual Dataset* pick_movies(set<int> sample);

    /* --------------------------------------------------------------------- *
     * Builds a subset S of this dataset such that
     *
     *     S = {(u,m,r) | u in sample, (u,m,r) in this dataset}
     *
     * ARGUMENTS:
     * - sample: the subset of users
     *
     * RETURN:
     * A subset of this dataset.
     * --------------------------------------------------------------------- */
    virtual Dataset* pick_users(set<int> sample);

    /* --------------------------------------------------------------------- *
     * Saves this dataset into a file.
     *
     * ARGUMENTS:
     * - filename: the file to save this dataset to
     * --------------------------------------------------------------------- */
    virtual void save(string filename);

    /* --------------------------------------------------------------------- *
     * Randomly splits this dataset into two disjoint datasets.
     *
     * ARGUMENTS:
     * - s1: the dataset into which put the first part of the whole dataset
     * - s2: the dataset into which put the second part of the whole dataset
     * - p: the probability that a rating is put into s1
     * --------------------------------------------------------------------- */
    virtual void split_ratings(Dataset* s1, Dataset* s2, double p);
};


/* ========================================================================= *
 * Functions
 * ========================================================================= */

/* ------------------------------------------------------------------------- *
 * Loads the movie titles.
 *
 * ARGUMENTS:
 * - filename: the list of movie titles
 *
 * RETURN:
 * The list of movie titles, as a mapping <movie ID> -> <movie title>.
 * ------------------------------------------------------------------------- */

map<int, string> load_titles(string filename);

/* ------------------------------------------------------------------------- *
 * Creates a random set of movies.
 *
 * ARGUMENTS:
 * - nb: the number of distinct movies
 *
 * RETURN:
 * A set of movie IDs.
 * ------------------------------------------------------------------------- */

set<int> random_movies(int nb);

/* ------------------------------------------------------------------------- *
 * Creates a random set of users.
 *
 * ARGUMENTS:
 * - nb: the number of distinct users
 *
 * RETURN:
 * A set of user IDs.
 * ------------------------------------------------------------------------- */

set<int> random_users(int nb);

#endif
