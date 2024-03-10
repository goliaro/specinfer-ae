#! /usr/bin/env bash
set -e
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

export UCX_DIR="$PWD/ucx-1.15.0/install"
export PATH=$UCX_DIR/bin:$PATH
export LD_LIBRARY_PATH=$UCX_DIR/lib:$LD_LIBRARY_PATH

./download_dataset.sh
./download_models.sh

batch_sizes=( 1 2 4 8 16 )

# rm -rf ./FlexFlow/inference/output || true
mkdir -p ./FlexFlow/inference/output

# single node, single GPU
ncpus=8
ngpus=1
llm_model_name="huggyllama/llama-7b"
ssm_model_name="JackFram/llama-68m"
for bs in "${batch_sizes[@]}"
do
    # Incremental decoding
    ./FlexFlow/build/inference/incr_decoding/incr_decoding -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize 20000 -ll:zsize 80000 -llm-model $llm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs -tensor-parallelism-degree $ngpus --fusion -output-file ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-incr_dec.txt > ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-incr_dec.out
    # Sequence-based speculative decoding
    ./FlexFlow/build/inference/spec_infer/spec_infer -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize 21800 -ll:zsize 80000 -llm-model $llm_model_name -ssm-model $ssm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs --expansion-degree -1 -tensor-parallelism-degree $ngpus --fusion -output-file ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-sequence_specinfer.txt > ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-sequence_specinfer.out
    # Tree-based speculative decoding
    ./FlexFlow/build/inference/spec_infer/spec_infer -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize 20000 -ll:zsize 80000 -llm-model $llm_model_name -ssm-model $ssm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs -tensor-parallelism-degree $ngpus --fusion -output-file ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-tree_specinfer.txt > ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-tree_specinfer.out
done

# single node, multiple GPU
ncpus=8
ngpus=4
llm_model_name="facebook/opt-30b"
ssm_model_name="facebook/opt-125m"
for bs in "${batch_sizes[@]}"
do
    # Incremental decoding
    ./FlexFlow/build/inference/incr_decoding/incr_decoding -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize 20000 -ll:zsize 80000 -llm-model $llm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs -tensor-parallelism-degree $ngpus --fusion -output-file ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-incr_dec.txt > ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-incr_dec.out
    # Sequence-based speculative decoding
    ./FlexFlow/build/inference/spec_infer/spec_infer -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize 20000 -ll:zsize 80000 -llm-model $llm_model_name -ssm-model $ssm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs --expansion-degree -1 -tensor-parallelism-degree $ngpus --fusion -output-file ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-sequence_specinfer.txt > ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-sequence_specinfer.out
    # Tree-based speculative decoding
    ./FlexFlow/build/inference/spec_infer/spec_infer -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize 20000 -ll:zsize 80000 -llm-model $llm_model_name -ssm-model $ssm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs -tensor-parallelism-degree $ngpus --fusion -output-file ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-tree_specinfer.txt > ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-tree_specinfer.out
done

# multiple node, multiple GPU
ncpus=8
ngpus=1
llm_model_name="huggyllama/llama-65b"
ssm_model_name="JackFram/llama-68m"
for bs in "${batch_sizes[@]}"
do
    # Incremental decoding
    ./FlexFlow/build/inference/incr_decoding/incr_decoding -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize 20000 -ll:zsize 30000 -llm-model $llm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs -tensor-parallelism-degree $ngpus --fusion -output-file ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-incr_dec.txt > ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-incr_dec.out
    # Sequence-based speculative decoding
    ./FlexFlow/build/inference/spec_infer/spec_infer -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize 20000 -ll:zsize 30000 -llm-model $llm_model_name -ssm-model $ssm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs --expansion-degree -1 -tensor-parallelism-degree $ngpus --fusion -output-file ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-sequence_specinfer.txt > 1_machine-${ngpus}_gpu-${bs}_batchsize-sequence_specinfer.out
    # Tree-based speculative decoding
    ./FlexFlow/build/inference/spec_infer/spec_infer -ll:cpu $ncpus -ll:util $ncpus -ll:gpu $ngpus -ll:fsize 20000 -ll:zsize 30000 -llm-model $llm_model_name -ssm-model $ssm_model_name -prompt ./FlexFlow/inference/prompt/chatgpt_$bs.json --max-requests-per-batch $bs -tensor-parallelism-degree $ngpus --fusion -output-file ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-tree_specinfer.txt > ./FlexFlow/inference/output/1_machine-${ngpus}_gpu-${bs}_batchsize-tree_specinfer.out
done
