# SpecInfer Artifact
*SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification* [ASPLOS'24] [Paper Link](https://arxiv.org/abs/2305.09781)

This is the artifact for SpecInfer. Follow the instructions below to install and run the tests.

## Clone the repo
To clone the repo, use the command below (don't forget the `--recursive` flag):
```
git clone --recursive https://github.com/goliaro/specinfer-ae.git
```

## Requirements
### Hardware dependencies
We run out experiments on two AWS g5.12xlarge instances, each with 4 NVIDIA A10 24GB GPUs, 48 CPU cores, and 192 GB DRAM. See below for the instructions to spin up the instances. Alternatively, we will also provide the reviewers login access to two pre-configured instances in our own AWS EC2 account. 

#### Spinning the AWS instances
Launch two AWS g5.12xlarge instances using the `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.1.0 (Ubuntu 20.04)` AMI. Make sure to place the instances in a [placement group](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/placement-groups.html) that utilizes the cluster strategy to achieve low-latency network performance. Attach the same security group to all instances and add an inbound rule in the security group to allow all incoming traffic from the same security group. For example, you can add the following rule: Type: All TCP, Source: Anywhere-IPv4.

### Software dependencies
The following software is required: CUDA 12.1, NCCL, Rust, CMake and Python3. Further, UCX and MPI are required for the multinode experiments. Additional Python dependencies are listed here: [here](https://github.com/flexflow/FlexFlow/blob/inference/requirements.txt). We recommend using the `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.1.0 (Ubuntu 20.04)` AMI, and provide scripts and a conda environment to install all the remaining dependencies.

### Models
We use the following LLM/SSM models for our experiments (for each model, we specify in parentheses the corresponding HuggingFace repository): LLaMA-68M (`JackFram/llama-68m`), LLaMA-7B (`huggyllama/llama-7b`), LLaMA-65B (`huggyllama/llama-65b`), OPT-125M (`facebook/opt-125m`), OPT-13B (`facebook/opt-13b`), OPT-125M (`facebook/opt-30b`). After installing SpecInfer, you can download all these models with the script: 
```
./download_models.sh
```
⚠️ Note that you will need to install SpecInfer BEFORE downloading the models.

## Installation
### Installing the prerequisites
After gaining access to the AWS instances, install the prerequisites by following the steps below. First, activate the conda shell support by running `conda init bash`, and then restarting the shell session.
Next, create the conda environment with all the required dependencies by running:
```
conda env create -f FlexFlow/conda/flexflow.yml
conda activate flexflow
```

### Multinode setup
Download and build UCX by running the `install_ucx.sh` script. Next, if you are running SpecInfer on two AWS instances, you will need to configure MPI so that the two instances are mutually accessible. Pick a main node, and create a SSH key pair with:
```
    ssh-keygen -t ed25519
```
Append the contents of the public key (`~/.ssh/id_ed25519.pub`) to the `~/.ssh/authorized_keys` file on BOTH the main and secondary machine. Note that if the `.ssh` folder or the `authorized_keys` file do not exist, you will need to create them manually. 
Finally, create a file at the path `~/hostfile` with the following contents:
```
<main_node_private_ip> slots=4
<secondary_node_private_ip> slots=4
```
replacing `<main_node_private_ip>` and `<secondary_node_private_ip>` with the private IP addresses of the two machines, and the number of slots with the number of GPUs available (if you are using the recommended AWS instances, you will use a value of 4). You can find each machine's private IP address by running the command (and use the first IP value that is printed):
```
    hostname -I
```

### Install SpecInfer

To install SpecInfer, run the script: 
```
./install_specinfer.sh
```

## Basic Test
To ensure that SpecInfer is installed correctly and is functional, run the 
```
./basic_test.sh
```
script. This script will test the basic incremental decoding and speculative inference functionalities, on both single and multi nodes. It will also test the support for offloading. The test passes if it prints the "Test passed!" message. 

## Experiment workflow

### Running experiments
We run the following two experiments to evaluate SpecInfer under different hardware setups. The output data will be saved to the `FlexFlow/inference/output` path.
- **Server-grade GPU evaluation.** This experiment tests the performance of SpecInfer on server-grade GPUs. The LLMs and SSMs are loaded in GPU memory, and we measure the end-to-end inference latency using 1 node, and 2 nodes. In the single node case, we measure the performance using 1 GPU, or 4 GPUs. In the multinode case, we use 4GPUs per node. The experiments use LLAMA-7B, OPT-30B and LLAMA-65B as the LLMs, and LLAMA-68M and OPT-125M as SSMs. The experiment runs SpecInfer in three different modes: incremental decoding, sequence-based speculative decoding, and tree-based speculative decoding. The former two are used to obtain data for the ablation study, and the latter is the novel inference mode proposed by SpecInfer, and will be deployed by the user. To run the server-grade GPU evaluation, run:
```
./server_gpu_experiments.sh
```
- **Offloading evaluation.** This experiment tests the performance of \sys when loading only a subset of parameters in GPU memory, while offloading the remaining ones on CPU DRAM. This technique is used to perform inference when the target model is larger than the available GPU memory. In the experiment, SpecInfer uses a single GPU and swaps the model's weights to and from the CPU. To run the offloading evaluation, run: 
```
./offloading_experiments.sh
```

**Third-party frameworks.** Please follow the vLLM, FasterTransformer, and HuggingFace TGI, and FlexGen official documentation to reproduce the performance of the third-party frameworks under the experiment scenarios.

### Parsing the output data
The scripts above will generate data at the `FlexFlow/inference/output` path. For each scenario, a `.txt` file contains the generated output for each prompt, and a `.out` file contains the stdout logs. The quality of the generated output can be evaluated visually and compared with the output from third-party inference frameworks.

We provide scripts to parse the raw output data and generate CSV files that can be used to generate the paper’s figures. The scripts are in the [data_parsing](./data_parsing) folder. Each script reads the relevant output data from the files in the `FlexFlow/inference/output` folder and produces output CSV (`.csv) file(s) with the result(s). The scripts work even with incomplete experiment data.
