#! /usr/bin/env bash
set -e
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

export UCX_DIR="$PWD/ucx-1.15.0/install"
export PATH=$UCX_DIR/bin:$PATH
export LD_LIBRARY_PATH=$UCX_DIR/lib:$LD_LIBRARY_PATH

./download_dataset.sh
./download_weights.sh

batch_sizes=( 1 2 4 8 16 )

# rm -rf ./FlexFlow/inference/output || true
mkdir -p ./FlexFlow/inference/output

# single node, single GPU
ncpus=8
ngpus=1
llm_model_name="facebook/opt-13b"
ssm_model_name="facebook/opt-125m"
for bs in "${batch_sizes[@]}"
do
./FlexFlow/build/inference/spec_infer/spec_infer -ll:cpu 8 -ll:util 8 -ll:gpu 1 -ll:fsize 21800 -ll:zsize 80000 -llm-model $llm_model_name -ssm-model $ssm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_offloading.json --max-requests-per-batch $bs --expansion-degree -1 -offload -offload-reserve-space-size 8000 --fusion -output-file ./FlexFlow/inference/output/offloading_small_${bs}.txt > ./FlexFlow/inference/output/offloading_small_${bs}.out
done

llm_model_name="facebook/opt-30b"
for bs in "${batch_sizes[@]}"
do
./FlexFlow/build/inference/spec_infer/spec_infer -ll:cpu 8 -ll:util 8 -ll:gpu 1 -ll:fsize 21800 -ll:zsize 80000 -llm-model $llm_model_name -ssm-model $ssm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_offloading.json --max-requests-per-batch $bs --expansion-degree -1 -offload -offload-reserve-space-size 8000 --fusion -output-file ./FlexFlow/inference/output/offloading_large_${bs}.txt > ./FlexFlow/inference/output/offloading_large_${bs}.out
done