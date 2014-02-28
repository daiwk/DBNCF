/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Runner (RBM over MapReduce)
 *
 * - Author: LOUPPE Gilles
 * - Last changes: April 9, 2010
 * ========================================================================= */

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include "Configuration.h"
#include "Dataset.h"
#include "RBMMapReduce.h"

#include <mpi.h>

using namespace std;
using namespace boost::program_options;


/* ========================================================================= *
 * main
 * ========================================================================= */

int main(int argc, char** argv) {
    // MPI initialization
    MPI_Init(&argc, &argv);

    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // Program options
    options_description opt_algorithm("Algorithm");
    opt_algorithm.add_options()
        ("conditional", value<bool>()->default_value(Config::RBM::CONDITIONAL), "Conditional?")
        ("roundrobin", value<bool>()->default_value(Config::MapReduce::ROUNDROBIN), "Round-robin?");

    options_description opt_sets("Sets");
    opt_sets.add_options()
        ("LS", value<string>()->default_value(Config::Sets::LS), "Learning set")
        ("VS", value<string>()->default_value(Config::Sets::VS), "Validation set")
        ("TS", value<string>()->default_value(Config::Sets::TS), "Test set")
        ("QS", value<string>()->default_value(Config::Sets::QS), "Qualification set");

    options_description opt_parameters("RBM Parameters");
    opt_parameters.add_options()
        ("F", value<int>()->default_value(Config::RBM::F), "Number of hidden units")
        ("epochs", value<int>()->default_value(Config::RBM::EPOCHS), "Number of epochs")
        ("cd-steps", value<int>()->default_value(Config::RBM::CD_STEPS), "Number of CD steps")
        ("eps-w", value<double>()->default_value(Config::RBM::EPS_W), "Learning rate (weights)")
        ("eps-vb", value<double>()->default_value(Config::RBM::EPS_VB), "Learning rate (visible units)")
        ("eps-hb", value<double>()->default_value(Config::RBM::EPS_HB), "Learning rate (hidden units)")
        ("eps-d", value<double>()->default_value(Config::RBM::EPS_D), "Learning rate (D matrix)")
        ("weight-cost", value<double>()->default_value(Config::RBM::WEIGHT_COST), "Weight cost")
        ("momentum", value<double>()->default_value(Config::RBM::MOMENTUM), "Momentum")
        ("annealing", value<bool>()->default_value(Config::RBM::ANNEALING), "Annealing?")
        ("annealing-rate", value<double>()->default_value(Config::RBM::ANNEALING_RATE), "Annealing rate");

    options_description opt_misc("Miscellaneous");
    opt_misc.add_options()
        ("help,h", "Show this help")
        ("verbose", value<bool>()->default_value(Config::RBM::VERBOSE), "Verbose mode")
        ("log", value<string>(), "Output file");

    options_description all("Options");
    all.add(opt_algorithm);
    all.add(opt_sets);
    all.add(opt_parameters);
    all.add(opt_misc);

    // Parse options
    variables_map vm;

    try {
        store(parse_command_line(argc, argv, all), vm);
    } catch (exception& e) {
        cerr << e.what() << endl << endl;
        cerr << all << endl;
        return EXIT_FAILURE;
    }

    if (vm.count("help")) {
        cerr << all << endl;
        return EXIT_FAILURE;
    }

    // Load datasets
    Dataset LS(vm["LS"].as<string>());
    Dataset VS(vm["VS"].as<string>());
    Dataset TS(vm["TS"].as<string>());
    Dataset QS(vm["QS"].as<string>());

    // RBM
    RBMMapReduce r(rank, np);

    // Sets
    r.addSet("LS", &LS); 
    r.addSet("VS", &LS);
    r.addSet("TS", &TS);
    r.addSet("QS", &QS);

    // Parameters
    r.setParameter("N", &Config::RBM::N, sizeof(int));
    r.setParameter("M", &Config::RBM::M, sizeof(int));
    r.setParameter("K", &Config::RBM::K, sizeof(int));
    r.setParameter("F", &vm["F"].as<int>(), sizeof(int));

    r.setParameter("conditional", &vm["conditional"].as<bool>(), sizeof(bool));
    r.setParameter("roundrobin", &vm["roundrobin"].as<bool>(), sizeof(bool));
    r.setParameter("epochs", &vm["epochs"].as<int>(), sizeof(int));
    r.setParameter("cd_steps", &vm["cd-steps"].as<int>(), sizeof(int));
    r.setParameter("eps_w", &vm["eps-w"].as<double>(), sizeof(double));
    r.setParameter("eps_vb", &vm["eps-vb"].as<double>(), sizeof(double));
    r.setParameter("eps_hb", &vm["eps-hb"].as<double>(), sizeof(double));
    r.setParameter("eps_d", &vm["eps-d"].as<double>(), sizeof(double));
    r.setParameter("weight_cost", &vm["weight-cost"].as<double>(), sizeof(double));
    r.setParameter("momentum", &vm["momentum"].as<double>(), sizeof(double));
    r.setParameter("annealing", &vm["annealing"].as<bool>(), sizeof(bool));
    r.setParameter("annealing_rate", &vm["annealing-rate"].as<double>(), sizeof(double));
    r.setParameter("verbose", &vm["verbose"].as<bool>(), sizeof(bool));

    if (vm.count("log")) {
        ostream* log = new ofstream(vm["log"].as<string>().c_str(), ios::out);
        r.setParameter("log", &log, sizeof(ostream*));
    } else {
        ostream* log = &cout;
        r.setParameter("log", &log, sizeof(ostream*));
    }

    // Train!
    r.train();

    // Cleanup
    MPI_Finalize();

    return EXIT_SUCCESS;
}
