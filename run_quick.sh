#!/bin/bash
sh run_dbncf_mpi.sh > nohup_dbncf.log 2> nohup_dbncf.log.wf
sh run_daecf_mpi.sh > nohup_daecf.log 2> nohup_daecf.log.wf
sh run_ahrbmcf_mpi.sh > nohup_ahrbmcf.log 2> nohup_ahrbmcf.log.wf
