# LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners

# Setup

```shell
git clone ...
cd continual_agent_bench
pip install -r requirements.txt
pip install pre-commit==4.0.1  # ensure that pre-commit hooks are installed
pre-commit install  # install pre-commit hooks
pre-commit run --all-files  # check its effect

docker pull mysql  # build images for db_bench

docker pull ubuntu  # build images for os_interaction
docker build -f scripts/dockerfile/os_interaction/default scripts/dockerfile/os_interaction --tag local-os/default
```

# Run experiments
If you want to run experiments in single machine mode, please use the following command:
```shell
export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ./src/run_experiment.py --config_path "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/standard.yaml"
```

If you want to run experiments in distributed mode, you first need to start the `ServerSideController` in the machine that can deploy the docker containers.
```shell
export PYTHONPATH=./

python src/distributed_deployment_utils/server_side_controller/main.py
```
Then, you can run the following command in HPC node.
```shell
export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/distributed_deployment_utils/run_experiment_remotely.py --config_path "configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/standard.yaml"
```
The `ServerSideController` can be reused for multiple experiments.
> [!NOTE]
> Don't forget to update the IP address in `configs/components/environment.yaml` as well as in the files under `configs/components/clients`.