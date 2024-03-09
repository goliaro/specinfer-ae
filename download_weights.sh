#! /usr/bin/env bash
set -e
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

export UCX_DIR="$PWD/ucx-1.15.0/install"
export PATH=$UCX_DIR/bin:$PATH
export LD_LIBRARY_PATH=$UCX_DIR/lib:$LD_LIBRARY_PATH

python ./FlexFlow/inference/utils/download_hf_model.py --half-precision-only JackFram/llama-68m
python ./FlexFlow/inference/utils/download_hf_model.py --half-precision-only huggyllama/llama-7b 
python ./FlexFlow/inference/utils/download_hf_model.py --half-precision-only huggyllama/llama-65b
python ./FlexFlow/inference/utils/download_hf_model.py --half-precision-only facebook/opt-125m
python ./FlexFlow/inference/utils/download_hf_model.py --half-precision-only facebook/opt-13b
python ./FlexFlow/inference/utils/download_hf_model.py --half-precision-only facebook/opt-30b

./FlexFlow/build/inference/spec_infer/spec_infer -ll:cpu 8 -ll:util 8 -ll:gpu 1 -ll:fsize 21800 -ll:zsize 80000 -llm-model facebook/opt-13b -ssm-model facebook/opt-125m -prompt ./FlexFlow/inference/prompt/chatgpt_offloading.json --max-requests-per-batch 1 --expansion-degree -1 -tensor-parallelism-degree 1 -offload --fusion -output-file ./FlexFlow/inference/output/offloading.txt > ./FlexFlow/inference/output/offloading.out

./FlexFlow/build/inference/spec_infer/spec_infer -ll:cpu 8 -ll:util 8 -ll:gpu 1 -ll:fsize 21800 -ll:zsize 80000 -llm-model JackFram/llama-68m -ssm-model JackFram/llama-68m -prompt ./FlexFlow/inference/prompt/chatgpt_offloading.json --max-requests-per-batch 1 --expansion-degree -1 -tensor-parallelism-degree 1 -offload --fusion -output-file ./FlexFlow/inference/output/offloading.txt > ./FlexFlow/inference/output/offloading.out