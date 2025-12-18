#!/bin/bash
set -e

# 循环遍历从 6 到 20
# for i in {6..20}
# do
#     echo "Current progress: Running iteration $i / 20"
    
#     python -m cs224r.scripts.run_hw1 \
#         --expert_policy_file cs224r/policies/experts/Ant.pkl \
#         --env_name Ant-v4 \
#         --exp_name dA_ant_$i \
#         --n_iter $i \
#         --expert_data "cs224r/expert_data/expert_data_Ant-v4.pkl" \
#         --do_dagger
        
#     echo "Finished iteration $i"
#     echo "----------------------------------------"
# done

for i in {2..13}
do
    echo "Current progress: Running iteration $i / 20"
    
    python -m cs224r.scripts.run_hw1 \
        --expert_policy_file cs224r/policies/experts/HalfCheetah.pkl \
        --env_name HalfCheetah-v4 \
        --exp_name dA_HalfCheetah_$i \
        --n_iter $i \
        --expert_data "cs224r/expert_data/expert_data_HalfCheetah-v4.pkl" \
        --do_dagger
        
    echo "Finished iteration $i"
    echo "----------------------------------------"
done