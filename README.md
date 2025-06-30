# FastTD3 - Simple and Fast RL for Humanoid Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2505.22642-b31b1b.svg)](https://arxiv.org/abs/2505.22642)


FastTD3 is a high-performance variant of the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm, optimized for complex humanoid control tasks. FastTD3 can solve various humanoid control tasks with dexterous hands from HumanoidBench in just a few hours of training. Furthermore, FastTD3 achieves similar or better wall-time-efficiency to PPO in high-dimensional control tasks from popular simulations such as IsaacLab and MuJoCo Playground.

For more information, please see our [project webpage](https://younggyo.me/fast_td3)


## ✨ Features

FastTD3 offers researchers a significant speedup in training complex humanoid agents.

- Ready-to-go codebase with detailed instructions and pre-configured hyperparameters for each task
- Support popular benchmarks: HumanoidBench, MuJoCo Playground, and IsaacLab
- User-friendly features that can accelerate your research, such as rendering rollouts, torch optimizations (AMP and compile), and saving and loading checkpoints

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed:
- Conda (for environment management)
- Git LFS (Large File Storage) -- For IsaacLab
- CMake -- For IsaacLab

And the following system packages:
```bash
sudo apt install libglfw3 libgl1-mesa-glx libosmesa6 git-lfs cmake
```

## 📖 Installation

This project requires different Conda environments for different sets of experiments.

### Common Setup
First, ensure the common dependencies are installed as mentioned in the [Prerequisites](#prerequisites) section.

### Environment for IsaacLab
```bash
conda create -n fasttd3_isaaclab -y python=3.10
conda activate fasttd3_isaaclab

# Install IsaacLab (refer to official documentation for the latest steps)
# Official Quickstart: https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
cd ..

# Install project-specific requirements
pip install -r requirements/requirements.txt
```
### (Optional) Accelerate headless GPU rendering in cloud instances

In some cloud VM images the NVIDIA kernel driver is present but the user-space OpenGL/EGL/Vulkan libraries aren't, so MuJoCo falls back to CPU renderer. You can install just the NVIDIA user-space libraries (and skip rebuilding the kernel module) with:

```bash
sudo apt install -y kmod
sudo sh NVIDIA-Linux-x86_64-<your_driver_version>.run -s --no-kernel-module --ui=none --no-questions
```

As a rule-of-thumb, if you're running experiments and rendering is taking longer than 5 seconds, it is very likely that GPU renderer is not used.

## 🚀 Running Experiments

Activate the appropriate Conda environment before running experiments.

Please see `fast_td3/hyperparams.py` for information regarding hyperparameters!

### IsaacLab Experiments
```bash
conda activate fasttd3_isaaclab
# FastTD3
python fast_td3/train.py \
    --env_name Isaac-Velocity-Flat-G1-v0 \
    --exp_name FastTD3 \
    --render_interval 0 \
    --seed 1
```

**Quick note:** For boolean-based arguments, you can set them to False by adding `no_` in front each argument, for instance, if you want to disable Clipped Q Learning, you can specify `--no_use_cdq` in your command.

## 💡 Performance-Related Tips

We used a single Nvidia A100 80GB GPU for all experiments. Here are some remarks and tips for improving performances in your setup or troubleshooting in your machine configurations.

- *Sample-efficiency* tends to improve with larger `num_envs`, `num_updates`, and `batch_size`. But this comes at the cost of *Time-efficiency*. Our default settings are optimized for wall-time efficiency on a single A100 80GB GPU. If you're using a different setup, consider tuning hyperparameters accordingly.
- When FastTD3 performance is stuck at local minima at the early phase of training in your experiments
  - First consider increasing the `num_updates`. This happens usually when the agent fails to exploit value functions. We also find higher `num_updates` tends to be helpful for relatively easier tasks or tasks with low-dimensional action spaces.
  - If the agent is completely stuck or much worse than your expectation, try using `num_steps=3` or disabling `use_cdq`.
  - For tasks that have penalty reward terms (e.g., torques, energy, action_rate, ..), consider lowering them for initial experiments, and tune the values. In some cases, curriculum learning with lower penalty terms followed by fine-tuning with stronger terms is effective.
- When you encounter out-of-memory error with your GPU, our recommendation for reducing GPU usage is (i) smaller `buffer_size`, (ii) smaller `batch_size`, and then (iii) smaller `num_envs`. Because our codebase is assigning the whole replay buffer in GPU to reduce CPU-GPU transfer bottleneck, it usually has the largest GPU consumption, but usually less harmful to reduce.

## 🛝 Playing with the FastTD3 training

A Jupyter notebook (`training_notebook.ipynb`) is available to help you get started with:
- Training FastTD3 agents.
- Loading pre-trained models.
- Visualizing agent behavior.
- Potentially, re-training or fine-tuning models.

## 🤖 Sim-to-Real RL with FastTD3

We provide the [walkthrough](sim2real.md) for training deployable policies with FastTD3.

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details. Note that the repository relies on third-party libraries subject to their respective licenses.

## Acknowledgements

This codebase builds upon [LeanRL](https://github.com/pytorch-labs/LeanRL) framework. 

We would like to thank people who have helped throughout the project:

- We thank [Kevin Zakka](https://kzakka.com/) for the help in setting up MuJoCo Playground.
- We thank [Changyeon Kim](https://changyeon.site/) for testing the early version of this codebase

## Citations

### FastTD3
```bibtex
@article{seo2025fasttd3,
  title     = {FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control},
  author    = {Seo, Younggyo and Sferrazza, Carmelo and Geng, Haoran and Nauman, Michal and Yin, Zhao-Heng and Abbeel, Pieter},
  booktitle = {preprint},
  year      = {2025},
}
```

### TD3
```bibtex
@inproceedings{fujimoto2018addressing,
  title={Addressing function approximation error in actor-critic methods},
  author={Fujimoto, Scott and Hoof, Herke and Meger, David},
  booktitle={International conference on machine learning},
  pages={1587--1596},
  year={2018},
  organization={PMLR}
}
```

### LeanRL

Following the [LeanRL](https://github.com/pytorch-labs/LeanRL)'s recommendation, we put CleanRL's bibtex here:

```bibtex
@article{huang2022cleanrl,
  author  = {Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga and Dipam Chakraborty and Kinal Mehta and João G.M. Araújo},
  title   = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {274},
  pages   = {1--18},
  url     = {http://jmlr.org/papers/v23/21-1342.html}
}
```

### Parallel Q-Learning (PQL)
```bibtex
@inproceedings{li2023parallel,
  title={Parallel $ Q $-Learning: Scaling Off-policy Reinforcement Learning under Massively Parallel Simulation},
  author={Li, Zechu and Chen, Tao and Hong, Zhang-Wei and Ajay, Anurag and Agrawal, Pulkit},
  booktitle={International Conference on Machine Learning},
  pages={19440--19459},
  year={2023},
  organization={PMLR}
}
```

### IsaacLab
```bibtex
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```

### Getting SAC to Work on a Massive Parallel Simulator
```bibtex
@article{raffin2025isaacsim,
  title   = "Getting SAC to Work on a Massive Parallel Simulator: An RL Journey With Off-Policy Algorithms",
  author  = "Raffin, Antonin",
  journal = "araffin.github.io",
  year    = "2025",
  month   = "Feb",
  url     = "https://araffin.github.io/post/sac-massive-sim/"
}
```

### Speeding Up SAC with Massively Parallel Simulation
```bibtex
@article{shukla2025fastsac,
  title   = "Speeding Up SAC with Massively Parallel Simulation",
  author  = "Shukla, Arth",
  journal = "https://arthshukla.substack.com",
  year    = "2025",
  month   = "Mar",
  url     = "https://arthshukla.substack.com/p/speeding-up-sac-with-massively-parallel"
}
```
