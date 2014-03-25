/* ========================================================================= *
 * Master thesis "Collaborative filtering, a neural network approach".
 *
 * Runner (RBMs).
 *
 * - Author: LOUPPE Gilles
 * - Last changes: March 1, 2010
 * ========================================================================= */

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

#include "Configuration.h"
#include "Dataset.h"
#include "RBMCF.h"
#include "RBMCF_OPENMP.h"

#include <vector>
#include <algorithm>
using namespace std;
using namespace boost::program_options;

extern "C" 
{
bool comp(pair<int, double> a, pair<int, double> b) {
    return a.second > b.second;
}
double predict_ratings(int userid, int movieid)
{
    RBMCF* r = new RBMCF("model_rbmcf");
    Dataset LS("svn_dir/data/bin/LS.bin.small-changeUserNum-1");
    Dataset TS("svn_dir/data/bin/TS.bin.small-changeUserNum-1");
    Dataset QS("svn_dir/data/bin/TS.bin.small-changeUserNum-1");
    Dataset VS("svn_dir/data/bin/TS.bin.small-changeUserNum-1");
//  Dataset LS("/home/daiwk/media-dir/DATA/daiwk/mpi/data/bin/LS.bin.small-changeUserNum-1");
//  Dataset TS("/home/daiwk/media-dir/DATA/daiwk/mpi/data/bin/TS.bin.small-changeUserNum-1");
//  Dataset QS("/home/daiwk/media-dir/DATA/daiwk/mpi/data/bin/TS.bin.small-changeUserNum-1");
//  Dataset VS("/home/daiwk/media-dir/DATA/daiwk/mpi/data/bin/TS.bin.small-changeUserNum-1");
    
    r->addSet("LS", &LS);
    r->addSet("VS", &VS);
    r->addSet("TS", &TS);
    r->addSet("QS", &QS);
    double pred = r->predict(userid,movieid);
    printf("pred: %lf\n", pred);
    return pred;
}
void recommend_top_k(int userid, int k, int* res)
{
    cout << "userid:" << userid << endl;
    RBMCF* r = new RBMCF("model_rbmcf");
    Dataset LS("svn_dir/data/bin/LS.bin.small-changeUserNum-1");
    Dataset TS("svn_dir/data/bin/TS.bin.small-changeUserNum-1");
    Dataset QS("svn_dir/data/bin/TS.bin.small-changeUserNum-1");
    Dataset VS("svn_dir/data/bin/TS.bin.small-changeUserNum-1");
//  Dataset LS("/home/daiwk/media-dir/DATA/daiwk/mpi/data/bin/LS.bin.small-changeUserNum-1");
//  Dataset TS("/home/daiwk/media-dir/DATA/daiwk/mpi/data/bin/TS.bin.small-changeUserNum-1");
//  Dataset QS("/home/daiwk/media-dir/DATA/daiwk/mpi/data/bin/TS.bin.small-changeUserNum-1");
//  Dataset VS("/home/daiwk/media-dir/DATA/daiwk/mpi/data/bin/TS.bin.small-changeUserNum-1");
    
//    r->addSet("LS", &LS);
    // 这里是要对测试集进行预测，而原来的predict函数是对LS进行预测
    r->addSet("LS", &TS);
    r->addSet("VS", &VS);
    r->addSet("TS", &TS);
    r->addSet("QS", &QS);

    vector<pair<int, double> > vec_final;


    int n = TS.users[userid];
    for(int m = TS.index[n]; m < TS.index[n] + TS.count[n]; m++) {
        int i = TS.ids[m];
        vec_final.push_back(pair<int, double>(i, r->predict(userid, i)));
    }
    sort(vec_final.begin(), vec_final.end(), comp);

//    res = new int[k];
    for(int i = 0; i < k; i++) {
        cout << vec_final[i].first << " " << vec_final[i].second <<endl;
        res[i] = vec_final[i].first;
    }

    return;
}

}

/* ========================================================================= *
 * main
 * ========================================================================= */

int main(int argc, char** argv) {
    
    // Start calculating the running time
    struct timeval start;
    struct timeval end;
    unsigned long usec;
    gettimeofday(&start, NULL);
    
    // Program options
    options_description opt_algorithm("Algorithm");
    opt_algorithm.add_options()
        ("conditional", value<bool>()->default_value(Config::RBMCF::CONDITIONAL), "Conditional?")
        ("openmp", value<bool>()->default_value(Config::RBMCF::OPENMP), "Multithread?");

    options_description opt_sets("Sets");
    opt_sets.add_options()
        ("LS", value<string>()->default_value(Config::Sets::LS), "Learning set")
        ("VS", value<string>()->default_value(Config::Sets::VS), "Validation set")
        ("TS", value<string>()->default_value(Config::Sets::TS), "Test set")
        ("QS", value<string>()->default_value(Config::Sets::QS), "Qualification set");

    options_description opt_parameters("RBMCF Parameters");
    opt_parameters.add_options()
        ("F", value<int>()->default_value(Config::RBMCF::F), "Number of hidden units")
        ("epochs", value<int>()->default_value(Config::RBMCF::EPOCHS), "Number of epochs")
        ("batch-size", value<int>()->default_value(Config::RBMCF::BATCH_SIZE), "Batch size")
        ("cd-steps", value<int>()->default_value(Config::RBMCF::CD_STEPS), "Number of CD steps")
        ("eps-w", value<double>()->default_value(Config::RBMCF::EPS_W), "Learning rate (weights)")
        ("eps-vb", value<double>()->default_value(Config::RBMCF::EPS_VB), "Learning rate (visible units)")
        ("eps-hb", value<double>()->default_value(Config::RBMCF::EPS_HB), "Learning rate (hidden units)")
        ("eps-d", value<double>()->default_value(Config::RBMCF::EPS_D), "Learning rate (D matrix)")
        ("weight-cost", value<double>()->default_value(Config::RBMCF::WEIGHT_COST), "Weight cost")
        ("momentum", value<double>()->default_value(Config::RBMCF::MOMENTUM), "Momentum")
        ("annealing", value<bool>()->default_value(Config::RBMCF::ANNEALING), "Annealing?")
        ("annealing-rate", value<double>()->default_value(Config::RBMCF::ANNEALING_RATE), "Annealing rate");

    options_description opt_misc("Miscellaneous");
    opt_misc.add_options()
        ("help,h", "Show this help")
        ("verbose", value<bool>()->default_value(Config::RBMCF::VERBOSE), "Verbose mode")
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
    // Start calculating the running time
    struct timeval start_data;
    struct timeval end_data;
    unsigned long usec_data;
    gettimeofday(&start_data, NULL);
    
    Dataset LS(vm["LS"].as<string>());
    Dataset VS(vm["VS"].as<string>());
    Dataset TS(vm["TS"].as<string>());
    Dataset QS(vm["QS"].as<string>());

    gettimeofday(&end_data, NULL);
    usec_data = 1000000 * (end_data.tv_sec - start_data.tv_sec) + end_data.tv_usec - start_data.tv_usec;
    cout << "File: " << __FILE__ << ", Function: " << __FUNCTION__  << ", Line: " << __LINE__ << endl;
    cout << "Time of loading datasets: " << usec_data << " usec[" << usec_data / 1000000. <<" sec]." << endl;

    // RBMCF
    RBMCF* r;

    if (!vm["openmp"].as<bool>()) {
	cout << "using rbm..." << endl;
        r = new RBMCF();
    } else {
	cout << "using openmp rbm..." << endl;
        r = new RBMCF_OPENMP();
    }

    // Sets
    r->addSet("LS", &LS);
    r->addSet("VS", &VS);
    r->addSet("TS", &TS);
    r->addSet("QS", &QS);
    cout << "finished add set" << endl;
    // Parameters
    r->setParameter("N", &Config::RBMCF::N, sizeof(int));
    r->setParameter("M", &Config::RBMCF::M, sizeof(int));
    r->setParameter("K", &Config::RBMCF::K, sizeof(int));
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
    r->setParameter("annealing_rate", &vm["annealing-rate"].as<double>(), sizeof(double));
    r->setParameter("verbose", &vm["verbose"].as<bool>(), sizeof(bool));

    if (vm.count("log")) {
        ostream* log = new ofstream(vm["log"].as<string>().c_str(), ios::out);
        r->setParameter("log", &log, sizeof(ostream*));
    } else {
        ostream* log = &cout;
        r->setParameter("log", &log, sizeof(ostream*));
    }
    
    cout << "training..." << endl;
    // Train!
    r->train();
	string model_name="model_rbmcf";
	r->save(model_name);
/*    cout << "finished training..." << endl;
    if (!vm["openmp"].as<bool>()) {
	cout << "saving rbm..." << endl;
    	r->save("rbm-model.rbm");
    } else {
	cout << "saving openmp rbm..." << endl;
    	r->save("rbm-openmp-model.rbm");
    }
*/
    // r->test("TS");
    // cout << "finished testing..." << endl;
    // Cleanup
    delete r;
    
    // print running time
    gettimeofday(&end, NULL);
    usec = 1000000 * (end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
    cout << "File: " << __FILE__ << ", Function: " << __FUNCTION__  << ", Line: " << __LINE__ << endl;
    cout << "Time of main(): " << usec << " usec[" << usec / 1000000. <<" sec][" << usec / 1000000. / 3600  << "h]. " << endl;

    return EXIT_SUCCESS;
}
