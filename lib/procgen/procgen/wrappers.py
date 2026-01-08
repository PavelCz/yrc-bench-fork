import numpy as np


class RandomEnvSwitchWrapper:
    """
    A wrapper that randomly switches between two procgen environments on reset.
    Since the environments are vectorized, each sub-environment independently 
    chooses which underlying environment to use when it resets.
    
    Args:
        venv0: First vectorized environment (env 0)
        venv1: Second vectorized environment (env 1)
        random_percent: Probability (0-100) of choosing venv0 on reset for each sub-env.
                       For example, random_percent=70 means 70% chance of using venv0,
                       30% chance of using venv1.
    
    The wrapper adds 'selected_env' to each info dict with value 0 for venv0 or 1 for venv1.
    """
    
    def __init__(self, venv0, venv1, random_percent):
        if venv0.num_envs != venv1.num_envs:
            raise ValueError(
                f"Both environments must have the same number of envs. "
                f"Got venv0.num_envs={venv0.num_envs}, venv1.num_envs={venv1.num_envs}"
            )
        
        if not (0 <= random_percent <= 100):
            raise ValueError(
                f"random_percent must be between 0 and 100. Got {random_percent}"
            )
        
        self.venv0 = venv0
        self.venv1 = venv1
        self.random_percent = random_percent / 100.0  # Convert to probability
        
        # Set standard vectorized environment attributes
        self.num_envs = venv0.num_envs
        self.observation_space = venv0.observation_space
        self.action_space = venv0.action_space
        
        # Track which environment each sub-env is currently using (True=venv0, False=venv1)
        self.env_selector = np.random.random(venv0.num_envs) < self.random_percent
    
    def reset(self):
        # Randomly choose which environment each sub-env will use
        self.env_selector = np.random.random(self.num_envs) < self.random_percent
        
        # Reset both environments
        obs0 = self.venv0.reset()
        obs1 = self.venv1.reset()
        
        # Select observations from the chosen environment for each sub-env
        obs = np.where(self.env_selector[:, None, None, None], obs0, obs1)
        return obs
    
    def step_async(self, actions):
        # Both environments need to step
        self.venv0.step_async(actions)
        self.venv1.step_async(actions)

    def step_wait(self):
        # Get results from both environments
        obs0, rews0, dones0, infos0 = self.venv0.step_wait()
        obs1, rews1, dones1, infos1 = self.venv1.step_wait()

        # Select results from the CURRENT environment BEFORE switching
        obs = np.where(self.env_selector[:, None, None, None], obs0, obs1)
        rews = np.where(self.env_selector, rews0, rews1)
        dones = np.where(self.env_selector, dones0, dones1)

        # Select infos from the current environment and add env identifier
        infos = []
        for i in range(self.num_envs):
            info = infos0[i] if self.env_selector[i] else infos1[i]
            # Add which environment was used: 0 for venv0, 1 for venv1
            info['selected_env'] = 0 if self.env_selector[i] else 1
            infos.append(info)

        # NOW switch environments for any sub-env that is done
        for i, done in enumerate(dones):
            if done:
                self.env_selector[i] = np.random.random() < self.random_percent

        return obs, rews, dones, infos

    def step(self, actions):
        """Step the environments synchronously."""
        self.step_async(actions)
        return self.step_wait()
    
    def close(self):
        self.venv0.close()
        self.venv1.close()
    
    def render(self, mode="human"):
        # Render from venv0 by default
        return self.venv0.render(mode=mode)
    
    def get_images(self):
        # Get images from both and select based on env_selector
        imgs0 = self.venv0.get_images()
        imgs1 = self.venv1.get_images()
        return [imgs0[i] if self.env_selector[i] else imgs1[i] 
                for i in range(self.num_envs)]
    
    def keys_to_act(self, keys_list):
        # Delegate to venv0 (both should have the same key mapping)
        if hasattr(self.venv0, 'keys_to_act'):
            return self.venv0.keys_to_act(keys_list)
        # Fallback: return None for each key sequence
        return [None] * len(keys_list)
    
    def get_state(self):
        # Get state from the currently selected environment for each sub-env
        states0 = self.venv0.get_state()
        states1 = self.venv1.get_state()
        return [states0[i] if self.env_selector[i] else states1[i]
                for i in range(self.num_envs)]
    
    def set_state(self, states):
        # Set state on both environments (only the selected one matters)
        self.venv0.set_state(states)
        self.venv1.set_state(states)
