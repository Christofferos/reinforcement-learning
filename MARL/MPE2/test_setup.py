"""
Quick test script to verify Simple Tag environment setup.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from mpe2 import simple_tag_v3
    print("✓ MPE2 library imported successfully")
except ImportError as e:
    print(f"✗ Failed to import MPE2: {e}")
    print("  Install with: pip install mpe2")
    sys.exit(1)

try:
    from config import ENV_CONFIG
    print("✓ Config loaded successfully")
except ImportError as e:
    print(f"✗ Failed to import config: {e}")
    sys.exit(1)

try:
    from agents import RandomAgent
    print("✓ Agents module loaded successfully")
except ImportError as e:
    print(f"✗ Failed to import agents: {e}")
    sys.exit(1)

# Test environment creation
try:
    env = simple_tag_v3.parallel_env(**ENV_CONFIG)
    print("✓ Environment created successfully")
    
    observations, infos = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  - Agents: {env.possible_agents}")
    print(f"  - Number of agents: {len(env.possible_agents)}")
    
    # Test one step
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print("✓ Environment step successful")
    
    env.close()
    print("✓ Environment closed successfully")
    
except Exception as e:
    print(f"✗ Environment test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test agent creation
try:
    env = simple_tag_v3.parallel_env(**ENV_CONFIG)
    env.reset()
    
    agent_name = env.possible_agents[0]
    obs_space = env.observation_space(agent_name)
    act_space = env.action_space(agent_name)
    
    agent = RandomAgent(agent_name, obs_space, act_space)
    print("✓ Random agent created successfully")
    
    observation = obs_space.sample()
    action = agent.select_action(observation)
    print(f"✓ Agent action selection successful: {action}")
    
    env.close()
    
except Exception as e:
    print(f"✗ Agent test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
print("\nYou can now:")
print("  1. Train agents: python train.py")
print("  2. Evaluate agents: python evaluate.py --checkpoint-dir <dir>")
print("="*60)
