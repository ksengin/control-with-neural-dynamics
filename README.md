# Control with Neural Dynamics (CND)

Implementation of "Neural Optimal Control using Learned System Dynamics"

[Paper](https://arxiv.org/abs/2302.09846) | [Video](https://youtu.be/WIbEY5rs60g?si=qQWBTbnGwF12WaRm)

## Install

1. Clone this repository and navigate to `src` folder
```bash
git clone https://github.com/ksengin/control-with-neural-dynamics.git
cd src
```

2. Create a python environment

```bash
conda create -n controlenv python=3.10 -y
conda activate controlenv
```

3. Install latest pytorch using your system configuration, for example
```bash
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

4. Install the rest of the packages
```bash
python -m pip install -r requirements.txt
```

## Quick start

To use CND, you can run

```bash
python train_controller.py --learned_fk --system cartpole --sess cnd_cartpole
```

which first learns the state transitions of a system, then trains a controller network using the learned system. The checkpoints are saved under `logs/cnd_cartpole`. Currently supported systems include: 'cartpole', 'acrobot', 'quadrotor'.


To train a controller using a previously learned state transitions network, you can use 

```bash
python train_controller.py --learned_fk --system cartpole --sess cnd_cartpole --archive_fkmodel logs/cnd_cartpole/systemid_model_f_cartpole_sine.pth
```

After training, state evolutions using the controller with the neural and true forward kinematics models are saved under the `logs` directory to `result_eval_surrogatefk.pdf` and `result_eval_truefk.pdf`, respectively.

### Training on custom systems

To train CND on a custom system,

- Write down the system equations in [src/models/control_systems.py](src/models/control_systems.py) to generate samples for supervising the state transitions network. Alternatively, you can use a simulator to generate data pairs.
- Specify the system configuration in [src/system_config.py](src/system_config.py), including the state and action dimensions, desired state and action in the final timestep, cost function parameters and the distribution to sample initial states from.
- (Optional) If you would like to test the performance of the controller subject to the true state transitions, implement the system equations in [src/control/system_dynamics.py](src/control/system_dynamics.py).



## Acknowledgments

This codebase is built on top of the following repositories:

- https://github.com/donken/NeuralOC
- https://github.com/DiffEqML/torchcontrol

we thank the authors for open sourcing their great work.

## Citation

If you find our code or paper useful, please consider citing:

```bibtex
@inproceedings{engin2023neural,
  title={Neural optimal control using learned system dynamics},
  author={Engin, Selim and Isler, Volkan},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  pages={953--960},
  year={2023},
  organization={IEEE}
}
```