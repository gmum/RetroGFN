# RetroGFN

This is the official implementation of "RetroGFN: Diverse and Flexible Retrosynthesis using GFlowNets".

## Project Structure

### API

Under `gflownet.api`, the repository provides a flexible API that clearly separates the GFlowNet components. The states, actions and action spaces can be represented as an arbitrary classes, which allows for easy implementation of GFlowNets with non-static environments (e.g. with dynamic action spaces).

- `env_base.py` Base class for environments. It provides a minimal and flexible interface that can be used to implement environments with dynamic action spaces. An action space is a set of possible actions that can be taken from a state in forward (forward action space) and backward (backward action space) direction. The reward is decoupled from the environment, so that environment should only describe the possible transitions between states. The environment can be reversed to enable backward sampling of the trajectories.
- `policy_base.py` A base class for policies. Given the current batch of states, a policy samples corresponding actions. It also computes the log probabilities when chosen actions and following states are provided.
- `sampler_base.py` A base class for samplers. A sampler samples trajectories from the environment using a policy.
- `trajectories.py`. A trajectory is a sequence of states and actions sampled by a sampler using the environment and the policy. Every state has a corresponding forward and backward action space which describe the possible actions that can be taken from that state. Trajectories are stored in a batch manner. The terminal states in the trajectories are assigned with rewards.
- `reward_base.py`. A class representing the reward function. The reward function is a function of a proxy output that takes a batch of states and computes rewards that are used to train the policy.
- `proxy_base.py`. A base class for proxies. A proxy is a function that takes a batch of states and computes values that are then used to compute the reward.
- `objective_base.py`. A base class for GFN objectives. An objective is a function that takes a batch of trajectories and computes the loss (objective)
- `replay_buffer_base.py`. A base class for replay buffers. A replay buffer stores terminal states or trajectories and can sample them
in backward direction using the provided sampler.

### Shared
Under `gflownets.shared`, the repository provides shared utilities that are used across the different GFlowNets implementations, e.g. Trajectory Balance Objective, Conditioned Trajectory Balance Objective, uniform policy, cached proxy base class, random samplers, reward_prioritized buffer, etc.

### GFNs
Uner `gflownets.gfns`, the repository provides the implementation of the GFlowNets. For the moment we support a toy HyperGrid GFlowNet and our RetroGFN.

## Setup
To create the conda environment, run the following commands:
```bash
conda create --name gflownet python=3.11.8 -y
conda activate gflownet

# If using CUDA:
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install dgl==1.1.2 -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html

pip install -e .

pip install pre-commit
pre-commit install
```

### Setup Syntheseus (and Chemformer Proxy)
To evaluate the RetroGFN using [Syntheseus API](https://github.com/microsoft/syntheseus) and standard top-k accuracy, you need to install our fork of Syntheseus repository using the following command:
```sh
sh external/setup_syntheseus.sh
```
If you want to additionally evaluate the RetroGFN with round-trip accuracy, you need to download the (eval) Chemformer Forward checkpoints from [here](https://ujchmura-my.sharepoint.com/:f:/r/personal/piotr_gainski_doctoral_uj_edu_pl/Documents/feasibility_proxies/chemformer?csf=1&web=1&e=rIjzdH) into `checkpoints` directory.

### Setup RFM Proxy
To train the RetroGFN using Reaction Feasibility Model (RFM) and evaluate it with Feasibility Thresholded Count (FTC) metric, you need to install [our RFM repository](https://github.com/panpiort8/ReactionFeasibilityModel/) with the following command:
````sh
sh external/setup_rfm.sh
````
and download the RFM checkpoints from [here](https://ujchmura-my.sharepoint.com/:f:/g/personal/piotr_gainski_doctoral_uj_edu_pl/EhHNt1xE009Eh6YI6z8b9KUBT6-2C-lsOTX5I0EWLk4lnw?e=9cPzl5) into `checkpoints` directory.


## Train
To train the RetroGFN using chemformer proxy, run:
```sh
python train.py --cfg configs/retro_chemformer_proxy.gin
```
The script will dump the results under `experiments/retro_chemformer_proxy/<timestamp>` directory. Our code uses gin-config package that allows for lightweight models configuration along with dependency injection.

## Evaluation
### With Syntheseus
To evaluate trained RetroGFN model, you need to install our fork of Syntheseus repository first (instructions above). To evaluate the model on USPTO-50k, you first need to download the data from [here](https://ujchmura-my.sharepoint.com/:f:/g/personal/piotr_gainski_doctoral_uj_edu_pl/EofVnAjcT0VGjGgHq3MmWZwBttkLM7rEU3ZyPqmRc5B5iw?e=ERBxLq) into `data` and run:
```sh
cd external/syntheseus
python -m syntheseus.cli.eval_single_step \
    data_dir=../../data/uspto_50k/converted/ \
    model_class=RetroGFN \
    model_kwargs.repeat=20 \
    model_dir=../../experiments/retro_chemformer_proxy/<timestamp> \
    results_dir=../../eval_results/uspto_50k/RetroGFN \
    num_gpus=1

cd ../..
python external/syntheseus/syntheseus/cli/evaluate_single_step_with_feasibility.py --device <device> --results_dir eval_results/uspto_50k/RetroGFN
```

### With other code
We provide a wrapper around RetroGFN that allows for easy adaptation to other codebases:
```python
from gflownet.gfns.retro.retro_gfn_single_step import RetroGFNSingleStep

model = RetroGFNSingleStep(
    model_dir='experiments/retro_chemformer_proxy/<timestamp>',
    device='cuda',
    repeat=20
)
products = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CCCC']

outputs = model.predict(products, num_results=100) # outputs is a list of lists of SMILES
```

[//]: # ()
[//]: # (### Checkpoints)

[//]: # (Here we provide two checkpoints that were used in the final evaluation in our paper: [RetroGFN with Chemformer Proxy]&#40;&#41; and [RetroGFN with RFM Proxy]&#40;&#41;. The checkpoints should be downloaded into `checkpoints` directory.)

[//]: # ()
[//]: # ()
[//]: # (## Cite)
