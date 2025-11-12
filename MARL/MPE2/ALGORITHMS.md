# Multi-Agent Reinforcement Learning Algorithms

This project implements several MARL algorithms with support for **Centralized Training with Decentralized Execution (CTDE)**.

## 📚 Available Algorithms

### 1. **MADDPG** (Multi-Agent DDPG) ⭐ Recommended for Simple Tag

**File:** `agents/maddpg_agent.py`

**Description:**

- **Centralized Training**: Critic has access to all agents' observations and actions
- **Decentralized Execution**: Each actor only uses its local observations
- Works with both continuous and discrete action spaces
- Best for mixed cooperative-competitive scenarios like Simple Tag

**Key Features:**

- Actor-Critic architecture
- Experience replay buffer
- Soft target network updates
- Exploration noise for continuous actions

**When to Use:**

- Predator-prey environments (Simple Tag)
- Mixed cooperative-competitive settings
- When you need stable learning in adversarial scenarios

**References:**

- Paper: [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
- Authors: Lowe et al., 2017

---

### 2. **MAPPO** (Multi-Agent PPO) ⭐ Most Stable

**File:** `agents/mappo_agent.py`

**Description:**

- **Centralized Training**: Centralized value function uses global state
- **Decentralized Execution**: Decentralized policy uses only local observations
- On-policy algorithm with clipped surrogate objective
- Very stable and sample-efficient

**Key Features:**

- PPO clipping for stable updates
- Generalized Advantage Estimation (GAE)
- Entropy regularization for exploration
- Mini-batch updates with multiple epochs

**When to Use:**

- When you need stable, reliable training
- Cooperative scenarios
- When sample efficiency is important
- Simple Tag with cooperative adversaries

**References:**

- Paper: [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)
- Authors: Yu et al., 2021

---

### 3. **DQN** (Deep Q-Network) - Independent Learning

**File:** `agents/dqn_agent.py`

**Description:**

- Independent learning (each agent learns separately)
- Not CTDE, but useful for baseline comparison
- Works only with discrete action spaces

**Key Features:**

- Experience replay
- Target network
- Epsilon-greedy exploration

**When to Use:**

- Baseline comparison
- Simple scenarios
- Quick prototyping

---

### 4. **Random Agent** - Baseline

**File:** `agents/random_agent.py`

**Description:**

- Selects random actions
- Useful for baseline performance comparison

---

## 🎯 Which Algorithm Should I Use?

### For Simple Tag Environment:

| Algorithm  | Best For              | Pros                                                | Cons                                              |
| ---------- | --------------------- | --------------------------------------------------- | ------------------------------------------------- |
| **MADDPG** | Adversarial scenarios | Handles mixed coop-comp well, stable in competition | More complex, needs tuning                        |
| **MAPPO**  | Stable training       | Very stable, sample efficient, easy to tune         | May be slower to converge in adversarial settings |
| **DQN**    | Baseline              | Simple, fast                                        | No coordination, may struggle in MARL             |
| **Random** | Baseline              | Shows min performance                               | No learning                                       |

### My Recommendation:

1. **Start with MAPPO** - Most stable and reliable
2. **Try MADDPG** - Better for competitive scenarios
3. **Compare with DQN** - Baseline performance

---

## 🚀 Usage Examples

### Training with MADDPG

```python
from mpe2 import simple_tag_v3
from agents import MADDPGAgent, MADDPGController

# Create environment
env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=3)
env.reset()

# Calculate total dimensions
agent_names = env.possible_agents
total_obs_dim = sum(env.observation_space(name).shape[0] for name in agent_names)
total_action_dim = sum(env.action_space(name).n for name in agent_names)

# Create MADDPG agents
config = {
    'gamma': 0.95,
    'tau': 0.01,
    'lr_actor': 1e-4,
    'lr_critic': 1e-3,
    'noise_scale': 0.1
}

agents = {}
for agent_name in agent_names:
    obs_space = env.observation_space(agent_name)
    act_space = env.action_space(agent_name)

    agents[agent_name] = MADDPGAgent(
        agent_name, obs_space, act_space,
        n_agents=len(agent_names),
        total_obs_dim=total_obs_dim,
        total_action_dim=total_action_dim,
        config=config
    )

# Create controller
controller = MADDPGController(agents, buffer_size=100000, batch_size=64)

# Training loop
for episode in range(1000):
    observations, _ = env.reset()
    done = False

    while env.agents:
        actions = {}
        for agent_name in env.agents:
            actions[agent_name] = agents[agent_name].select_action(
                observations[agent_name], training=True
            )

        next_observations, rewards, terminations, truncations, _ = env.step(actions)

        # Store experience
        dones = {name: terminations[name] or truncations[name] for name in env.agents}
        controller.store_experience(observations, actions, rewards, next_observations, dones)

        # Update agents
        controller.update_all_agents()

        observations = next_observations
```

### Training with MAPPO

```python
from mpe2 import simple_tag_v3
from agents import MAPPOAgent, MAPPOController
import numpy as np

# Create environment
env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=3)
env.reset()

# Calculate total observation dimension
agent_names = env.possible_agents
total_obs_dim = sum(env.observation_space(name).shape[0] for name in agent_names)

# Create MAPPO agents
config = {
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'lr_actor': 3e-4,
    'lr_critic': 1e-3,
    'clip_epsilon': 0.2,
    'entropy_coef': 0.01
}

agents = {}
for agent_name in agent_names:
    obs_space = env.observation_space(agent_name)
    act_space = env.action_space(agent_name)

    agents[agent_name] = MAPPOAgent(
        agent_name, obs_space, act_space,
        total_obs_dim=total_obs_dim,
        config=config
    )

# Create controller
controller = MAPPOController(agents)

# Training loop
rollout_length = 25  # Same as max_cycles
for episode in range(1000):
    observations, _ = env.reset()

    for step in range(rollout_length):
        if not env.agents:
            break

        # Get global state
        global_state = np.concatenate([observations[name] for name in agent_names])

        actions = {}
        for agent_name in env.agents:
            action, log_prob, value = agents[agent_name].select_action(
                observations[agent_name],
                training=True,
                global_state=global_state
            )
            actions[agent_name] = action

            # Store in buffer (will be used in update)
            # Note: This is simplified - see full implementation in train_mappo.py

        next_observations, rewards, terminations, truncations, _ = env.step(actions)
        observations = next_observations

    # Update all agents after rollout
    controller.update_all_agents(observations)
```

---

## 🔧 Hyperparameter Tuning Guide

### MADDPG Hyperparameters

```python
MADDPG_CONFIG = {
    'gamma': 0.95,              # Discount factor (try 0.95-0.99)
    'tau': 0.01,                # Soft update rate (try 0.001-0.01)
    'lr_actor': 1e-4,           # Actor learning rate
    'lr_critic': 1e-3,          # Critic learning rate (usually higher)
    'hidden_dim': 128,          # Hidden layer size
    'noise_scale': 0.1,         # Exploration noise
    'noise_decay': 0.9999,      # Noise decay rate
    'min_noise': 0.01,          # Minimum noise
}
```

### MAPPO Hyperparameters

```python
MAPPO_CONFIG = {
    'gamma': 0.99,              # Discount factor
    'gae_lambda': 0.95,         # GAE lambda (0.9-0.99)
    'lr_actor': 3e-4,           # Actor learning rate
    'lr_critic': 1e-3,          # Critic learning rate
    'clip_epsilon': 0.2,        # PPO clip parameter (0.1-0.3)
    'entropy_coef': 0.01,       # Entropy coefficient (0.001-0.1)
    'value_coef': 0.5,          # Value loss coefficient
    'max_grad_norm': 0.5,       # Gradient clipping
    'ppo_epochs': 10,           # Update epochs (5-15)
    'mini_batch_size': 64,      # Mini-batch size
}
```

---

## 📖 Further Reading

- [CTDE Overview](https://bair.berkeley.edu/blog/2018/12/12/rllib/)
- [Multi-Agent RL Survey](https://arxiv.org/abs/1911.10635)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [MPE2 Documentation](https://mpe2.farama.org/)

---

## 🤝 Contributing

To add a new algorithm:

1. Create `agents/your_algorithm_agent.py`
2. Inherit from `BaseAgent`
3. Implement `select_action()` and `update()` methods
4. Add to `agents/__init__.py`
5. Create example training script
6. Update this documentation
