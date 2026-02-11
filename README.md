# Reinforcement Learning for Flappy Bird
Neural Fitted Q-Iteration, DQN and Double DQN

<p align="center">
    <img src="images/dqn_best_step475000_ret18.50-episode-3.gif" 
        alt="flappy bird"/>
</p>

## Purpose of the Project
  
This project was carried out as part of an academic reinforcement learning assignment and focuses on the implementation, training, and evaluation of different approaches for learning control policies from interaction.

The main objective is to analyze how classical Deep Q-Learning methods and batch reinforcement learning techniques behave when applied to the same task, under comparable experimental conditions. In particular, the project investigates the impact of algorithmic choices on learning stability, sample efficiency, and final performance.

To this end, three algorithms are implemented and compared:
- Neural Fitted Q-Iteration (NFQ),
- Deep Q-Network (DQN),
- Double Deep Q-Network (DDQN).

All agents are trained on the same environment and share a common network architecture and reward structure, allowing for a fair and systematic comparison.  
Beyond implementation, the project emphasizes reproducibility and experimental analysis, including multi-seed evaluation, aggregated performance metrics, and visual inspection of learning dynamics.

## Problem Description and Environment

The considered task is the classic Flappy Bird game, formulated as a reinforcement learning control problem.  
The agent controls a bird that moves horizontally at constant speed and must decide when to flap in order to avoid colliding with vertical pipes and the ground.

The environment is episodic and stochastic, with each episode starting from an initial configuration and terminating upon collision with an obstacle or the environment boundaries. The objective of the agent is to survive for as long as possible by maintaining a safe trajectory through the gaps between pipes.

All experiments are conducted using the `FlappyBird-v0` environment provided by the `flappy-bird-gymnasium` package and wrapped within the Gymnasium interface. The environment exposes a low-dimensional continuous state representation and a discrete action space, making it suitable for value-based reinforcement learning methods with function approximation.

No modifications are applied to the environment dynamics. However, a mild reward shaping term is introduced to encourage longer survival and stabilize learning, while preserving the original structure of the task.

## Reinforcement Learning Formulation 

The Flappy Bird task is formulated as a Markov Decision Process (MDP) where the agent aims to maximize its cumulative reward through interaction with the environment. The problem is defined by the following components, designed to balance computational efficiency with learning stability.

| Component        | Definition                        | Rationale & Design Choices                                                                                                                      |
|------------------|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| State Space      | Low-dimensional continuous vector | Includes relative distances to the next obstacle and velocity. Captures essential dynamics while avoiding the overhead of raw pixel processing. |
| Action Space     | "Discrete: {0,1}"                 | 0: No action; 1: Upward flap. This binary control is ideal for evaluating value-based methods.                                                  |
| Reward Function  | Sparse + Survival Shaping         | +1 for passing pipes; +0.05 per step for survival. Shaping encourages exploration without distorting the primary goal.                          |
| Function Approx. | Fully Connected Neural     | Shared architecture across algorithms to ensure that performance gains are due to the learning rule, not capacity.                              |
| Optimization     | Stochastic Gradient Descent (SGD) | Minimizes Mean Squared TD Error. Chosen for its stable and interpretable dynamics in low-dimensional state spaces.                              |
| Policy           | ϵ-greedy                          | Decaying schedule for DQN/DDQN; fixed ϵ for NFQ to ensure adequate coverage during batch collection.                                            |

Design Justifications
- State Preprocessing: Observations are converted to float32 and lightly clipped to prevent numerical instabilities during backpropagation, a critical step for SGD stability.
- Reward Shaping Sensitivity: Preliminary experiments showed that "survival rewards" $> 0.05$ led to risk-averse agents that avoided pipes to stay alive longer. The current value was tuned to ensure pipe-     passing remains the dominant objective.
- Optimization Choice: SGD was preferred over adaptive optimizers (like Adam) in this specific task to maintain tighter control over the learning dynamics and to better observe the differences in stability    between DQN and Double DQN.
- Termination: Episodes terminate strictly upon collision, with no explicit negative terminal reward, letting the loss of future rewards act as the primary penalty.

## Algorithmic Approaches

The project evaluates three value-based reinforcement learning algorithms to solve the Flappy Bird task. Following the course's focus on comparative analysis, the algorithms were chosen to represent both batch and online learning paradigms.

**Selected Methods**
- Neural Fitted Q-Iteration (NFQ): Used as a batch RL baseline. It frames value estimation as a regression problem, alternating between data collection and offline optimization.
- Deep Q- (DQN): An online method that introduces a replay buffer and a target  to stabilize training in deep function approximation.
- Double Deep Q- (DDQN): An evolution of DQN designed to mitigate the overestimation of action values by decoupling action selection from evaluation.

**Implementation Strategy**
Instead of complex architectures, all agents share a fully connected neural  and are optimized via SGD. This setup ensures that differences in performance, such as DDQN’s superior stability or NFQ’s sample inefficiency, are directly attributable to the algorithmic logic rather than representational capacity. Hyperparameters like the discount factor $\gamma$ and the $\epsilon$-greedy schedule were kept consistent across runs to ensure a fair systematic comparison.

### Software Stack

The project is developed using Python and relies on the following main libraries:
- **Gymnasium** for environment interaction,
- **flappy-bird-gymnasium** for the Flappy Bird environment,
- **PyTorch** for neural  modeling and optimization,
- **NumPy** for numerical operations,
- **Matplotlib** for logging and visualization of results.

### Real-Time Training Monitoring

During the execution of the training scripts (`dqn_flappy.py`, `ddqn_flappy.py`, `nfq_flappy.py`), the progress is logged directly to the terminal to allow for real-time monitoring of the agent's evolution. 
The logs provide visibility into the following metrics:
- Episode Progress: Current episode number and total environment steps completed.
- Performance Metrics: Episode return and rolling average reward to track improvement.
- Exploration Status: The current value of $\epsilon$ (epsilon), ensuring the decay schedule is progressing as intended.
- Optimization: Loss values for each update to monitor the stability of the SGD optimization.

### Logging and Experiment Organization

Each training run is executed in isolation and logs its outputs to a dedicated directory. For every algorithm and random seed, a separate run folder is created, containing all artifacts produced during training and evaluation.

In particular, each run directory stores:
- training and evaluation metrics (e.g., episode returns and evaluation scores),
- environment step counters associated with evaluation,
- model checkpoints corresponding to the latest and best-performing policies.

This directory-based organization ensures a clear separation between runs and enables systematic post-hoc analysis, aggregation across seeds, and full reproducibility of the experiments.

## Experimental Setup and Evaluation

All experiments are conducted under a controlled and consistent setup to ensure a fair comparison between algorithms. Particular care is taken to account for the stochastic nature of reinforcement learning and to reduce the impact of random initialization.

### Training

Each agent is trained for a fixed number of episodes using the same environment configuration, reward structure, and network architecture. For DQN and DDQN, training is fully online, with the agent continuously interacting with the environment and updating the Q-network during learning. For NFQ, training follows a batch-oriented protocol, where experience is first collected and then used for fitted Q-iteration updates.

Exploration is handled through ε-greedy policies, with algorithm-specific strategies:
- a fixed ε is used for NFQ during data collection,
- a decaying ε schedule is adopted for DQN and DDQN, with an initial exploration phase followed by predominantly greedy behavior.

### Multi-Seed Evaluation

To improve robustness and reproducibility, each algorithm is trained using multiple random seeds. Specifically, three independent runs are performed for each agent, each initialized with a different seed affecting network initialization, environment dynamics, and exploration behavior.

All reported results are aggregated across these runs. Performance curves are summarized using mean values, along with variability bands computed from the minimum and maximum returns observed across seeds.

### Evaluation Metrics

Agent performance is evaluated in terms of episode return, which corresponds to the cumulative reward obtained during an episode and directly reflects the agent’s ability to successfully pass pipes and survive over time.

Two complementary evaluation perspectives are considered:
- **Training episode returns**, capturing learning dynamics and stability during training,
- **Evaluation episode returns**, obtained by periodically running the agent with a greedy policy to assess the quality of the learned behavior independently from exploration noise.

### Result Aggregation and Visualization

Learning curves are generated by aggregating results across seeds and applying a smoothing window to improve readability. Dedicated comparison scripts are used to align curves across episodes and environment steps, and to visualize both average performance and variability.

This evaluation protocol enables a systematic and quantitative comparison of learning speed, stability, and final performance across NFQ, DQN, and DDQN.


## Results and Discussion

This section presents and discusses the experimental results obtained by applying NFQ, DQN, and Double DQN to the Flappy Bird task. Results are analyzed both in terms of learning dynamics during training and evaluation performance measured independently from exploration.

### Evaluation Performance

Evaluation curves report the average return obtained by executing the learned policy greedily at fixed intervals during training. This evaluation protocol provides a clear picture of the agent’s effective behavior, decoupled from exploration noise.

Across all experiments, Double DQN consistently achieves higher and more stable evaluation returns compared to standard DQN. This behavior confirms the expected reduction of overestimation bias introduced by the Double Q-learning update, resulting in smoother learning curves and improved asymptotic performance.

While individual evaluation curves exhibit variability across random seeds, the aggregated results show a clear upward trend in performance as training progresses. Variability bands highlight that Double DQN maintains a tighter performance range, indicating improved stability.

### Impact of Reward Shaping

Preliminary experiments revealed that the magnitude of the survival reward plays a crucial role in shaping agent behavior. When the survival reward was set too high, agents learned conservative strategies that prioritized staying alive rather than successfully passing pipes.

Reducing the survival reward to a sufficiently small value restored the dominance of the environment-defined pipe-passing reward and led to more goal-oriented behavior. This highlights the importance of carefully balancing rewards.

### Evaluation Performance

Training episode returns provide insight into the learning dynamics and stability of each algorithm.

[Graph 1: NFQ vs DQN – eval] 
<p align="center">
   <img src="images/nfq_vs_dqn_eval.png" width="80%" alt="Eval nfq vs dqn"> 
</p>

**NFQ vs DQN – Evaluation Performance.**  
Mean evaluation return as a function of environment interaction steps, aggregated across three random seeds. The shaded area represents the minimum and maximum performance observed across seeds. Evaluation is performed using a greedy policy, providing a clean estimate of policy quality.

[Graph 2: DQN vs DDQN – eval]
<p align="center">
   <img src="images/dqn_vs_ddqn_eval.png" width="80%" alt="Eval dqn vs ddqn">
</p>

**DQN vs DDQN – Evaluation Performance.**  
Comparison of evaluation returns for DQN and Double DQN under identical experimental conditions. Results are aggregated across three random seeds and reported as mean with min–max variability bands. The improved stability and higher asymptotic performance of Double DQN highlight the impact of reducing overestimation bias.

In addition to aggregated results across multiple seeds, we report a representative single-run evaluation curve and rollout video to provide qualitative insight into the learned behavior of the agent.
Example evaluation curve from a single representative DDQN run:

<p align="center">
   <img src="images/eval_curve_ddqn_seed2.png" width="50%" alt="Eval dqn vs ddqn">
</p>


### Training Dynamics

[Graph 3: DQN vs NFQ – training episodes]
<p align="center">
   <img src="images/dqn_vs_nfq_episodes.png" width="80%" alt="epis nfq vs dqn">
</p>

**DQN vs NFQ – Training Episode Returns.**  
Average episode return observed during training as a function of training episodes. These curves reflect learning dynamics under ε-greedy exploration and are influenced by exploration noise. They are reported to illustrate differences in learning stability rather than final policy quality.

[Graph 4: DQN vs DDQN – training episodes]
<p align="center">
   <img src="images/dqn_vs_ddqn_episodes.png" width="80%" alt="epis dqn vs ddqn">
</p>

**DQN vs DDQN – Training Episode Returns.**  
Training episode returns for DQN and Double DQN under ε-greedy exploration. While both algorithms exhibit similar learning trends during training, evaluation-based results provide a clearer comparison of final policy performance.


### Discussion

The results highlight the advantages of online deep reinforcement learning methods over batch-based approaches in dynamic control tasks. Double DQN emerges as the most reliable algorithm in this setting, combining improved stability with higher final performance.

NFQ serves as a useful baseline for understanding the limitations of fitted value iteration in environments characterized by fast dynamics and sparse rewards.

All reported trends are consistent across multiple random seeds, strengthening the reliability of the observed conclusions.

## How to Run the Code

This section describes how to reproduce the experiments and generate the results reported in this repository.

### Limitations

While the implemented agents demonstrate successful learning, several limitations were observed during the experimental phase:
- Sensitivity to Reward Shaping: The reliance on a survival reward ($+0.05$) highlights a weakness in sparse reward handling. Without this shaping, convergence was significantly slower, suggesting that the agents might struggle in environments where "safe" intermediate rewards cannot be easily defined.
- Sample Efficiency of NFQ: As a batch method, NFQ proved less efficient than online methods in this high-frequency control task. Its performance was heavily bottlenecked by the quality and diversity of the initial data collection phase.
- Overfitting to Specific Dynamics: The agents were trained on a low-dimensional state representation. While effective, this model would likely fail if moved to an image-based input (pixels) without a significant redesign of the architecture (e.g., adding Convolutional layers) and increased computational resources.
- Stability vs. Performance Trade-off: The use of SGD provided stable updates, but at the cost of slower convergence compared to adaptive optimizers like Adam. Furthermore, even with Double DQN, some variance between seeds remains, indicating that the policy is still sensitive to random initialization.

### Requirements

The project requires Python 3.10 or later. The main dependencies are:
- `gymnasium`
- `flappy-bird-gymnasium`
- `torch`
- `numpy`
- `pandas`
- `matplotlib`

It is recommended to run the code inside a dedicated Python virtual environment to avoid dependency conflicts.
A virtual environment can be created and activated as follows (i used a Windows pc):

```bash
python -m venv ReinfLearn-env
ReinfLearn-env\Scripts\activate 
```

Dependencies can be installed using pip:
```bash
pip install gymnasium flappy-bird-gymnasium torch numpy pandas matplotlib
```
To make sure that gymnasium library is correctly installed you can run:
```bash
python test_flappy_gymnasium.py
```

### Training the agents

Each algorithm can be trained independently by running the corresponding script:
```bash
python nfq_flappy.py
python dqn_flappy.py
python ddqn_flappy.py
```
Each execution performs a single training run with a fixed random seed and saves all outputs (metrics, evaluation scores, and model checkpoints) to a dedicated directory inside the results/ folder.

To reproduce the multi-seed experiments, the training scripts should be executed multiple times with different seed values.
I trained each algorithm respectively with seed 0, 1 and 2.

Each training run saves its outputs to a dedicated directory called "algorithm_seedX" (for example ddqn_seed0), which contains all artifacts required for analysis, comparison, and reproducibility:
- plots/: Generated figures, including evaluation curves (e.g., evaluation return vs environment steps) and other run-specific visualizations;
- videos/: Optional recordings of the agent interacting with the environment using the learned policy, intended for qualitative inspection;
- Model checkpoints (*.pt): Saved network weights, including the best-performing model observed during evaluation and the final model at the end of training;
- eval_scores.npy: NumPy array storing evaluation returns collected during periodic greedy evaluations;
- eval_steps.npy: NumPy array storing the corresponding environment step counts for each evaluation;
- metrics.csv: CSV file containing logged training and evaluation metrics used for post-hoc analysis and for generating aggregated comparison plots.

```
.
└── algorithm_seedX/                      # Output directory for a single training run (specific algorithm + seed)
    ├── plots/                            # Run-specific visualizations generated during or after training
    │   └── eval_curve.png                # Evaluation return vs environment steps (greedy policy)
    │
    ├── videos/                           # Recordings of the agent during evaluation
    │   └── algorithm_best_step_retY-episodeZ
    │                                      # Naming convention:
    │                                      # Y = evaluation return achieved during the recorded run
    │                                      # Z = index of the evaluation episode used for video recording
    │
    ├── _algorithm_flappy_best.pt         # Model checkpoint with the highest evaluation return
    ├── _algorithm_flappy_latest.pt       # Model checkpoint saved at the end of training
    │
    ├── eval_scores.npy                   # Evaluation returns recorded at periodic evaluation intervals
    ├── eval_steps.npy                    # Environment steps corresponding to each evaluation point
    │
    └── metrics.csv                       # Logged training and evaluation metrics used for analysis and aggregation

```


### Generating Comparison Plots

**This analysis step must be performed after all training runs have been completed**, as it relies on the `results/` directory.
Aggregated comparison plots across multiple random seeds can be generated using the provided analysis scripts. These scripts load the metrics saved by individual runs, align evaluation curves, and compute aggregated statistics such as mean and variability across seeds.

The following scripts are provided:
- `compare_eval_pairs.py` – generates pairwise aggregated evaluation comparisons between selected algorithms.
- `compare_training_pairs.py` – generates pairwise comparisons based on training episode returns to analyze learning dynamics.

To run the analysis scripts, execute the following commands after completing the training runs:

```bash
python compare_eval_pairs.py --results results --seeds 0 1 2 --smooth 3 --outdir compare_pairs_eval
python compare_training_pairs.py --results results --seeds 0 1 2 --smooth 50 --outdir compare_pairs_episodes
```

## Repository Structure
```
.
├── results/                         # Per-run output directories (one per algorithm and seed)
│   ├── ddqn_seed0/                  # Metrics, checkpoints, plots, and videos for a single run
│   ├── ddqn_seed1/
│   ├── ddqn_seed2/
│   ├── dqn_seed0/
│   ├── dqn_seed1/
│   ├── dqn_seed2/
│   ├── nfq_seed0/
│   ├── nfq_seed1/
│   └── nfq_seed2/
│
├── compare_pairs_eval/               # Aggregated evaluation comparison outputs
│   ├── nfq_vs_dqn_eval.png          # Evaluation plots (mean ± min/max)
│   ├── dqn_vs_ddqn_eval.png         # Evaluation plots (mean ± min/max)
│   ├── nfq_mean_min_max.csv         # Aggregated statistics
│   ├── dqn_mean_min_max.csv
│   ├── ddqn_mean_min_max.csv
│   └── summary_pairs.csv            # Summary comparison table
│
├── compare_pairs_episodes/          # Aggregated training-dynamics plots (episode-based)
│   ├── dqn_vs_ddqn_episodes.png     # Episode return comparison during training 
│   └── dqn_vs_nfq_episodes.png      # Episode return comparison during training 
│
├── dqn_flappy.py                    # DQN training script
├── ddqn_flappy.py                   # Double DQN training script
├── nfq_flappy.py                    # NFQ training script
│
├── compare_eval_pairs.py            # Pairwise evaluation comparison script
├── compare_training_pairs.py        # Pairwise training-dynamics comparison script
│
├── test_flappy_gymnasium.py         # Environment sanity check script
├── README.md                        # Project documentation
└── ReinLearn-env/                   # Python virtual environment (optional)
```


## References

- R. Schiavone, *Flappy Bird Gym Environment*, GitHub repository,  
  https://github.com/robertoschiavone/flappy-bird-env
- Berta, R. (2025). *Neural Fitted Q-Iteration*. Course lecture notes, Reinforcement Learning.
- Berta, R. (2025). *Deep Q-Networks and Extensions*. Course lecture notes, Reinforcement Learning.
- Gymnasium: Farama Foundation. Gymnasium: An open source interface for reinforcement learning algorithms. https://gymnasium.farama.org/
- PyTorch: Paszke, A., et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. https://pytorch.org/
- DQN: Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
- Double DQN: van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-Learning. AAAI Conference on Artificial Intelligence.
- NFQ: Riedmiller, M. (2005). Neural Fitted Q Iteration – First Experiences with a Data Efficient Neural Reinforcement Learning Method. Machine Learning: ECML.
- Miguel Morales, Grokking Deep Reinforcement Learning, Manning
- Richard S. Sutton and Andrew G. Barto, Reinforcement Learning: An Introduction, MIT Press

## Author

Project developed by Giorgia La Torre (student id 4441614) as part of a university reinforcement learning project.





























