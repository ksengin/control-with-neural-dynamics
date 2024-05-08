# Control with Neural Dynamics (CND)

Implementation of "Neural Optimal Control using Learned System Dynamics"

[Paper](https://arxiv.org/abs/2302.09846) | [Video](https://youtu.be/WIbEY5rs60g?si=qQWBTbnGwF12WaRm)

## Install

1. Clone this repository and navigate to src folder
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

which first learns the state transitions of a system, then trains a controller network using the learned system. The checkpoints are saved under `logs/cnd_cartpole`.


To train a controller using a previously learned state transitions network, you can use 

```bash
python train_controller.py --learned_fk --system cartpole --sess cnd_cartpole --archive_fkmodel logs/cnd_cartpole/systemid_model_f_cartpole_sine.pth
```

## Acknowledgments

This codebase built on top of the following repositories:

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