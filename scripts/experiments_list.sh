#!/bin/bash

# Define model dimension mapping
declare -A MODEL_DIMS=(
  ["roberta-base"]="768"
  ["roberta-base_vanilla"]="768"
  ["bert-base-uncased"]="768"
  ["roberta-base_no_train"]="768"
  ["bert-base-uncased_no_train"]="768"
  ["all-MiniLM-L6-v2"]="384"
  ["all-MiniLM-L6-v2_base"]="384"
  ["all-MiniLM-L6-v2_vanilla"]="384"
  ["ada"]="1536"
  
)

# Define dataset paths
declare -A DATASET_PATHS=(
["G3"]=""
 )

n_tool=10
# Define unseen tool paths
declare -A ADD_PATHS=(
# percent
)

# Experiment parameter space
DATASETS=( "G3") 
LAYERS=(3)
TOP_Is=(4)
TOP_Ts=(2)


MODEL_NAME="bert-base-uncased"
PHASE_ARGS="--phase train"
RETRIEVAL_TYPE="--retrieval_type bert"
without_logic=0
without_query_transfer=0
without_tool_transfer=0
history=0
COMMON_ARGS="--lr 0.001 --decay 1e-4"
gnn_name=lightgcn

total_experiments=0
for dataset in "${DATASETS[@]}"; do
  for layer in "${LAYERS[@]}"; do
    for topI in "${TOP_Is[@]}"; do
      for topT in "${TOP_Ts[@]}"; do
        experiment_args="--dataset ${dataset} --model_name ${MODEL_NAME} --layer ${layer} --topI ${topI} --topT ${topT} ${PHASE_ARGS} ${RETRIEVAL_TYPE} --retrieval_path ${DATASET_PATHS[$dataset]} --n_tool ${n_tool} --add_path ${ADD_PATHS[$dataset]} --without_logic ${without_logic} --without_query_transfer ${without_query_transfer} --without_tool_transfer ${without_tool_transfer} ${COMMON_ARGS} --gnn_name ${gnn_name} --history ${history}"
        
        experiments+=("$experiment_args")
        ((total_experiments++))
      done
    done
  done
done

echo "Total number of generated experiment configurations: ${total_experiments}"
# Export variables
export experiments
export COMMON_ARGS
export MODEL_DIMS