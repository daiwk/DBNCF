#!/bin/bash
function init()
{

	date="date"
	date_str=`date "+%Y%m%d%H%M%S"`

	rmsedir="dbncf_rmse_dir_mpi"
	logdir="dbncf_log_dir_mpi"
	datadir="/home/daiwk/media-dir/DATA/daiwk/mpi"
	mkdir ${rmsedir}
	mkdir ${logdir}

    machines="machines" #for single machine: machines-single
}


function run_small_mpi()
{

	type=$1
	F=$2
	
	epoch=30
	batch=3000
	hsize=20
	hlnum=3
	
	eps_w=0.0015 #default 0.0015
	eps_vb=0.0012 #default 0.0012
	eps_hb=0.1 #default 0.1
	eps_d=0.001 #default 0.001
	weight_cost=0.0001 #default 0.0001
	momentum=0.95 #default 0.95
	annealing=2

	LS=${datadir}/"data/bin/LS.bin.small-changeUserNum-1"
	TS=${datadir}/"data/bin/TS.bin.small-changeUserNum-1"
	log="log-openmp-small"
	rmse="rmse-openmp-small"

	echo "running small"
	echo "starting openmp ${type} at:"
	eval ${date}

	#run_cmd="./RunDBNCF 
	#run_cmd="mpirun -machinefile ${machines} -np 4 RunDBNCF
	run_cmd="./RunDBNCF 
		--LS ${LS}
		--TS ${TS} 
		--QS ${TS} 
		--VS ${TS} 
		--F ${F} 
		--openmp ${type} > ${logdir}/${type}_${log}_F${F}_E${epoch}_B${batch}_HS${hsize}_HL${hlnum}_EW${eps_w}_EVB${eps_vb}_EHB${eps_hb}_ED${eps_d}_WC${weight_cost}_M${momentum}_A${annealing}
		--verbose 1 
		--log ${rmsedir}/${type}_${rmse}_F${F}_E${epoch}_B${batch}_HS${hsize}_HL${hlnum}_EW${eps_w}_EVB${eps_vb}_EHB${eps_hb}_ED${eps_d}_WC${weight_cost}_M${momentum}_A${annealing}"

	echo ${run_cmd}
	eval ${run_cmd}

	echo "finished openmp 1 at:"
	eval ${date}


}


function run_full_mpi()
{

	type=$1
	F=$2

	epoch=30
	batch=100
	hsize=200
	hlnum=3
	
	eps_w=0.0015 #default 0.0015
	eps_vb=0.0012 #default 0.0012
	eps_hb=0.1 #default 0.1
	eps_d=0.001 #default 0.001
	weight_cost=0.0001 #default 0.0001
	momentum=0.95 #default 0.95
	annealing=2

	LS=${datadir}/"data/bin/LS.bin.fengge-changeUserNum-1"
	TS=${datadir}/"data/bin/TS.bin.fengge-changeUserNum-1"
	log="log-openmp-full"
	rmse="rmse-openmp-full"

	echo "running full-free"
	echo "starting openmp ${type} at:"
	eval ${date}

	#run_cmd="./RunDBNCF 
	#run_cmd="mpirun -machinefile ${machines} -np 4 RunDBNCF 
	run_cmd="./RunDBNCF 
		--LS ${LS}
		--TS ${TS} 
		--QS ${TS} 
		--VS ${TS} 
		--F ${F} 
		--openmp ${type} > ${logdir}/${type}_${log}_F${F}_E${epoch}_B${batch}_HS${hsize}_HL${hlnum}_EW${eps_w}_EVB${eps_vb}_EHB${eps_hb}_ED${eps_d}_WC${weight_cost}_M${momentum}_A${annealing}
		--verbose 1 
		--log ${rmsedir}/${type}_${rmse}_F${F}_E${epoch}_B${batch}_HS${hsize}_HL${hlnum}_EW${eps_w}_EVB${eps_vb}_EHB${eps_hb}_ED${eps_d}_WC${weight_cost}_M${momentum}_A${annealing}"

	echo ${run_cmd}
	eval ${run_cmd}

	echo "finished openmp ${type} at:"
	eval ${date}


}



function main()
{

	make clean && make -j 9
	rm -rf *layer* rbm-*
	init

	# params:
	# $1: type(0: not openmp, 1: openmp)
	# $2: F
	run_small_mpi 1 28
 #	run_small_mpi 0 28
# 	run_full_mpi 1 28
 #	run_full 0 20

}

main
