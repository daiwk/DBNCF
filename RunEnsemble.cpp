/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Runner (Ensemble)
 *
 * - Author: LOUPPE Gilles
 * - Last changes: March 1, 2010
 * ========================================================================= */

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <boost/program_options.hpp>

#include "Configuration.h"
#include "Dataset.h"
#include "RBMOpenMP.h"

#include "Dumb.h"

using namespace std;
using namespace boost::program_options;


/* ========================================================================= *
 * main
 * ========================================================================= */

int main(int argc, char** argv) {
    // Program options
    options_description opt_algorithm("Algorithm");
    opt_algorithm.add_options()
        ("conditional", value<bool>()->default_value(Config::RBM::CONDITIONAL), "Conditional?");

    options_description opt_sets("Sets");
    opt_sets.add_options()
        ("LS", value<string>()->default_value(Config::Sets::LS), "Learning set")
        ("VS", value<string>()->default_value(Config::Sets::VS), "Validation set")
        ("TS", value<string>()->default_value(Config::Sets::TS), "Test set")
        ("QS", value<string>()->default_value(Config::Sets::QS), "Qualification set");

    options_description opt_ensemble("Ensemble parameters");
    opt_ensemble.add_options()
        ("nb-movies", value<int>()->default_value(Config::Ensemble::NB_MOVIES), "Number of movies")
        ("nb-users", value<int>()->default_value(Config::Ensemble::NB_USERS), "Number of users");

    options_description opt_parameters("RBM Parameters");
    opt_parameters.add_options()
        ("F", value<int>()->default_value(Config::RBM::F), "Number of hidden units")
        ("epochs", value<int>()->default_value(Config::RBM::EPOCHS), "Number of epochs")
        ("batch-size", value<int>()->default_value(Config::RBM::BATCH_SIZE), "Batch size")
        ("cd-steps", value<int>()->default_value(Config::RBM::CD_STEPS), "Number of CD steps")
        ("eps-w", value<double>()->default_value(Config::RBM::EPS_W), "Learning rate (weights)")
        ("eps-vb", value<double>()->default_value(Config::RBM::EPS_VB), "Learning rate (visible units)")
        ("eps-hb", value<double>()->default_value(Config::RBM::EPS_HB), "Learning rate (hidden units)")
        ("eps-d", value<double>()->default_value(Config::RBM::EPS_D), "Learning rate (D matrix)")
        ("weight-cost", value<double>()->default_value(Config::RBM::WEIGHT_COST), "Weight cost")
        ("momentum", value<double>()->default_value(Config::RBM::MOMENTUM), "Momentum")
        ("annealing", value<bool>()->default_value(Config::RBM::ANNEALING), "Annealing?")
        ("delta", value<double>()->default_value(Config::RBM::ANNEALING_RATE), "Annealing rate");

    options_description opt_misc("Miscellaneous");
    opt_misc.add_options()
        ("help,h", "Show this help")
        ("verbose", value<bool>()->default_value(Config::RBM::VERBOSE), "Verbose mode")
        ("log", value<string>()->default_value("predictions.bin"), "Output file");

    options_description all("Options");
    all.add(opt_algorithm);
    all.add(opt_sets);
    all.add(opt_ensemble);
    all.add(opt_parameters);
    all.add(opt_misc);

    // Parse options
    variables_map vm;

    try{
        store(parse_command_line(argc, argv, all), vm);
    } catch (exception& e){
        cerr << e.what() << endl << endl;
        cerr << all << endl;
        return EXIT_FAILURE;
    }

    if (vm.count("help")){
        cerr << all << endl;
        return EXIT_FAILURE;
    }

    // Load datasets
    Dataset LS(vm["LS"].as<string>());
    Dataset VS(vm["VS"].as<string>());
    //Dataset TS(vm["TS"].as<string>());
    Dataset QS(vm["QS"].as<string>());

    // Configure the RBM
    //RBMOpenMP* r = new RBMOpenMP();
    Dumb3* r = new Dumb3();

    /*
    set<int> movies = random_movies(vm["nb-movies"].as<int>());
    Dataset* sub_ls = LS.pick_movies(movies);
    Dataset* sub_qs = QS.pick_movies(movies);

    if (vm["nb-users"].as<int>() < Config::Netflix::NB_USERS) {
        set<int> users = random_users(vm["nb-users"].as<int>());

        Dataset* tmp;

        tmp = sub_ls;
        sub_ls = sub_ls->pick_users(users);
        delete tmp;

        tmp = sub_qs;
        sub_qs = sub_qs->pick_users(users);
        delete tmp;
    }
    */

    r->addSet("LS", &LS);
    r->addSet("VS", &VS);
    //r->addSet("TS", &TS);
    r->addSet("QS", &QS);

    // Parameters
    r->setParameter("N", &Config::RBM::N, sizeof(int));
    r->setParameter("M", &Config::RBM::M, sizeof(int));
    r->setParameter("K", &Config::RBM::K, sizeof(int));
    r->setParameter("F", &vm["F"].as<int>(), sizeof(int));

    r->setParameter("conditional", &vm["conditional"].as<bool>(), sizeof(bool));
    r->setParameter("epochs", &vm["epochs"].as<int>(), sizeof(int));
    r->setParameter("batch_size", &vm["batch-size"].as<int>(), sizeof(int));
    r->setParameter("cd_steps", &vm["cd-steps"].as<int>(), sizeof(int));
    r->setParameter("eps_w", &vm["eps-w"].as<double>(), sizeof(double));
    r->setParameter("eps_vb", &vm["eps-vb"].as<double>(), sizeof(double));
    r->setParameter("eps_hb", &vm["eps-hb"].as<double>(), sizeof(double));
    r->setParameter("eps_d", &vm["eps-d"].as<double>(), sizeof(double));
    r->setParameter("weight_cost", &vm["weight-cost"].as<double>(), sizeof(double));
    r->setParameter("momentum", &vm["momentum"].as<double>(), sizeof(double));
    r->setParameter("annealing", &vm["annealing"].as<bool>(), sizeof(bool));
    r->setParameter("annealing_rate", &vm["delta"].as<double>(), sizeof(double));
    r->setParameter("verbose", &vm["verbose"].as<bool>(), sizeof(bool));

    // Train + validate
    r->train();
    ofstream out(vm["log"].as<string>().c_str(), ios::out | ios::binary);

    for (int n = 0; n < VS.nb_rows; n++) {
        if (VS.count[n] <= 0) {
            continue;
        }

        int u = VS.rows[n];

        for (int m = VS.index[n]; m < VS.index[n] + VS.count[n]; m++) {
            double rating = r->predict(u, VS.ids[m]);
            out.write((char*) &rating, sizeof(double));
        }
    }

    out.close();

    // RMSE
    /*
    ifstream in("predictions.bin", ios::in | ios::binary);
    double* predictions = new double[VS.nb_ratings];
    in.read((char*) predictions, VS.nb_ratings * sizeof (double));
    in.close();

    double error = 0.;

    for (int r = 0; r < VS.nb_ratings; r++) {
        if (predictions[r] <= 0.0) predictions[r] = 3.6033;

        double e = VS.ratings[r] - predictions[r];
        error += e * e;
    }

    delete predictions;
    */

    return EXIT_SUCCESS;
}
