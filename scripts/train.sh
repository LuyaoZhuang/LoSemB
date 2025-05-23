#!/bin/bash

for experiment in "${experiments[@]}"; do
    FULL_ARGS="$COMMON_ARGS $experiment"
    dataset=$(echo "$FULL_ARGS" | grep -oP -- '--dataset \K\S+')
    model_name=$(echo "$FULL_ARGS" | grep -oP -- '--model_name \K\S+')
    phase=$(echo "$FULL_ARGS" | grep -oP -- '--phase \K\S+')
    layer=$(echo "$FULL_ARGS" | grep -oP -- '--layer \K\S+')
    topI=$(echo "$FULL_ARGS" | grep -oP -- '--topI \K\S+' || echo "default")
    topT=$(echo "$FULL_ARGS" | grep -oP -- '--topT \K\S+' || echo "default")
    without_logic=$(echo "$FULL_ARGS" | grep -oP -- '--without_logic \K\S+' || echo "default")
    without_query_transfer=$(echo "$FULL_ARGS" | grep -oP -- '--without_query_transfer \K\S+' || echo "default")
    without_tool_transfer=$(echo "$FULL_ARGS" | grep -oP -- '--without_tool_transfer \K\S+' || echo "default")
    gnn_name=$(echo "$FULL_ARGS" | grep -oP -- '--gnn_name \K\S+' || echo "default")
    recdim=${MODEL_DIMS[$model_name]}
    FULL_ARGS="$FULL_ARGS --recdim $recdim"
    history=$(echo "$FULL_ARGS" | grep -oP -- '--history \K\S+' || echo "default")
    TIMESTAMP=$(date +"%Y%m%d_%H%M")

    # Create configuration parameter suffix
    config_suffix=""
    if [[ "$FULL_ARGS" == *"--without_logic 1"* ]]; then
        config_suffix="${config_suffix}-wl"
    fi
    if [[ "$FULL_ARGS" == *"--without_query_transfer 1"* ]]; then
        config_suffix="${config_suffix}-wq"
    fi
    if [[ "$FULL_ARGS" == *"--without_tool_transfer 1"* ]]; then
        config_suffix="${config_suffix}-wt"
    fi
    if [[ "$FULL_ARGS" == *"--history 1"* ]]; then
        config_suffix="${config_suffix}-history"
    fi
    
    if [[ "$FULL_ARGS" == *"--phase orig"* ]] && [[ "$FULL_ARGS" == *"--n_tool 0"* ]] && [[ "$FULL_ARGS" == *"--history 0"* ]]; then
        output_dir="outputs/${dataset}/${gnn_name}/${model_name}/${phase}/${TIMESTAMP}"
    elif [[ "$FULL_ARGS" == *"--phase orig"* ]] && [[ "$FULL_ARGS" == *"--n_tool 0"* ]] && [[ "$FULL_ARGS" == *"--history 1"* ]]; then
        output_dir="outputs/${dataset}/${gnn_name}/${model_name}/${phase}/history/${TIMESTAMP}"
    elif [[ "$FULL_ARGS" == *"--n_tool 0"* ]]; then
        output_dir="outputs/${dataset}/${gnn_name}/${model_name}/${phase}/lgn-${layer}-${topI}-${topT}${config_suffix}-${TIMESTAMP}"
    else
        file_name=$(basename "$(echo "$FULL_ARGS" | grep -oP -- '--add_path \K\S+' || echo "default")" .json)
        output_dir="outputs/${dataset}/${gnn_name}/${model_name}/percent_${n_tool}/${file_name}/${phase}/lgn-${layer}-${topI}-${topT}${config_suffix}-${TIMESTAMP}"
    fi

    FULL_ARGS="$FULL_ARGS --output_dir $output_dir"
    mkdir -p "$output_dir" || { echo "Failed to create output directory"; continue; }
    LOG_FILE="${output_dir}/experiment.log"

    echo "=== Starting experiment ===" | tee "$LOG_FILE"
    echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i $CUDA_DEVICE)" | tee -a "$LOG_FILE"
    echo "Output dir: $output_dir" | tee -a "$LOG_FILE"
    echo "Full command: $PYTHON_PATH $MAIN_SCRIPT $FULL_ARGS" | tee -a "$LOG_FILE"

    (
        source /home/anaconda3/bin/activate "$CONDA_ENV" && \
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
        "$PYTHON_PATH" "$MAIN_SCRIPT" $FULL_ARGS 2>&1 | tee -a "$LOG_FILE"
    )

done