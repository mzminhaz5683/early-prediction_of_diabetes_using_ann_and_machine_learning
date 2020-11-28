#!/bin/bash
clear

#python3 -m programs.project_v3_actual_split
#python3 -m programs.project_v5_random_split


python3 -m programs.project_model_v11

#python3 -m test
#kill -9 PIDs


##################################################################################################
#for ((a=01; a <= 1 ; a++))
#do

#python3 -m programs.project_model_v11_2
#python3 -m programs.project_v3_actual_split
#python3 -m programs.ann_model_v4

#read -t 20 -p "wait for 20 seconds only ..."
#done

#echo 'Finish run.sh\n'