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

#include "AHRBMCF.h"
// #include <mpi.h>

using namespace std;
using namespace boost::program_options;


/* ========================================================================= *
 * main
 * ========================================================================= */

int main(int argc, char** argv) {

  //    int myid, numprocs;
  //    int namelen;
  //    char processor_name[MPI_MAX_PROCESSOR_NAME];
  //
  //    MPI_Init (&argc, &argv);
  //    MPI_Comm_rank (MPI_COMM_WORLD, &myid);
  //    MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
  //    MPI_Get_processor_name (processor_name, &namelen);
  //    fprintf (stderr, "Hello World! Process %d of %d on %s\n", myid, numprocs, processor_name);

  // Start calculating the running time
		struct timeval start;
		struct timeval end;
		unsigned long usec;
		gettimeofday(&start, NULL);

		// Program options
		options_description opt_algorithm("Algorithm");
		opt_algorithm.add_options()
				("conditional", value<bool>()->default_value(Config::RBM::CONDITIONAL), "Conditional?")
				("openmp", value<bool>()->default_value(Config::RBM::OPENMP), "Multithread?");

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
				("batch-size", value<int>()->default_value(Config::RBM::BATCH_SIZE), "Batch size")
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

		int layer_sizes[] = {20, 30, 30, 30};
		int layer_num = sizeof(layer_sizes) / sizeof(layer_sizes[0]);

		AHRBMCF* ahrbmcf = new AHRBMCF();
		//AHRBMCF ahrbmcf;

		ahrbmcf->addSet("LS", &LS);
		ahrbmcf->addSet("VS", &VS);
		ahrbmcf->addSet("TS", &TS);
		ahrbmcf->addSet("QS", &QS);
	//	ahrbmcf.addSet("LS", &LS);
	//	ahrbmcf.addSet("VS", &VS);
	//	ahrbmcf.addSet("TS", &TS);
	//	ahrbmcf.addSet("QS", &QS);

		// RBM* r = ahrbmcf->rbm_layer;


		/*    if (!vm["openmp"].as<bool>()) {
		  cout << "using rbm..." << endl;
		  r = new RBM();
		  } else {
		  cout << "using openmp rbm..." << endl;
		  r = new RBMOpenMP();
		  }
		  */
		// Sets
	//     r->addSet("LS", &LS);
  //     r->addSet("VS", &VS);
  //     r->addSet("TS", &TS);
  //     r->addSet("QS", &QS);
		cout << "finished add set" << endl;
		/*    // Parameters
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
		  r->setParameter("annealing_rate", &vm["annealing-rate"].as<double>(), sizeof(double));
		  r->setParameter("verbose", &vm["verbose"].as<bool>(), sizeof(bool));
		  */    

		  if (vm.count("log")) {
		  ostream* log = new ofstream(vm["log"].as<string>().c_str(), ios::out);
		  ahrbmcf->setParameter("log", &log, sizeof(ostream*));
		  } else {
		  ostream* log = &cout;
		  ahrbmcf->setParameter("log", &log, sizeof(ostream*));
		  }
		printf("training...\n");
		// Train!
		//ahrbmcf->train();
		ahrbmcf->pretrain();
		 
		//ahrbmcf.train();

		// double RMSE= ahrbmcf->test("TS");
		// printf("RMSE: %lf\n", RMSE);
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

  // delete r;


		delete ahrbmcf;

		//    MPI_Finalize ();

	// print running time
		gettimeofday(&end, NULL);
		usec = 1000000 * (end.tv_sec-start.tv_sec) + end.tv_usec - start.tv_usec;
		cout << "File: " << __FILE__ << ", Function: " << __FUNCTION__  << ", Line: " << __LINE__ << endl;
		cout << "Time of main(): " << usec << " usec[" << usec / 1000000. <<" sec][" << usec / 1000000. / 3600  << "h]. " << endl;

		return EXIT_SUCCESS;
}
