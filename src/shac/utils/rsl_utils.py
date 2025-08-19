from shac.utils.common import *


# Import rsl_rl components
try:
    # Add the rsl_rl source directory to Python path
    import sys

    rsl_rl_path = os.path.join(os.path.dirname(__file__), '../../..', 'externals', 'rsl_rl')
    sys.path.insert(0, rsl_rl_path)

    # Import RSL-RL components directly
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner
    from rsl_rl.env.vec_env import VecEnv

    RSL_RL_AVAILABLE = True

    # Patch the git repository handling to avoid errors with invalid git repos
    import rsl_rl.utils.utils as rsl_utils

    original_store_code_state = rsl_utils.store_code_state


    def safe_store_code_state(logdir, repositories):
        """Safe version of store_code_state that filters out invalid git repositories."""
        valid_repositories = []
        for repo_path in repositories:
            try:
                import git
                # Check if this is a valid git repository
                git.Repo(repo_path, search_parent_directories=True)
                valid_repositories.append(repo_path)
            except (git.exc.InvalidGitRepositoryError, Exception):
                # Skip invalid git repositories
                continue

        if valid_repositories:
            original_store_code_state(logdir, valid_repositories)


    # Replace the function
    rsl_utils.store_code_state = safe_store_code_state

except ImportError as e:
    RSL_RL_AVAILABLE = False
    print_warning(f"rsl_rl not available: {e}")
    print_warning("Make sure the rsl_rl directory exists in externals/")


# Wrapper class to make existing environments compatible with rsl_rl's VecEnv interface
class RSLRLEnvWrapper(VecEnv):
    """Wrapper to make existing environments compatible with rsl_rl's VecEnv interface."""

    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.num_obs = env.num_obs
        self.num_actions = env.num_actions
        self.max_episode_length = getattr(env, 'max_episode_length', 1000)
        self.device = getattr(env, 'device', torch.device('cpu'))

        # Initialize buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device)

        # Initialize extras with proper structure for RSL-RL
        self.extras = {"observations": {}}  # RSL-RL expects this structure

        # Get initial observations - handle both tuple and single return values
        obs = self.env.reset()
        if isinstance(obs, tuple):
            self.obs_buf = obs[0].to(self.device)
        else:
            self.obs_buf = obs.to(self.device)

    def get_observations(self):
        """Return current observations and extras."""
        return self.obs_buf, self.extras

    def reset(self):
        """Reset all environments."""
        obs = self.env.reset()
        if isinstance(obs, tuple):
            self.obs_buf = obs[0].to(self.device)
        else:
            self.obs_buf = obs.to(self.device)
        self.episode_length_buf.fill_(0)
        self.reset_buf.fill_(0)
        return self.obs_buf, self.extras

    def step(self, actions):
        """Apply actions to environments."""
        step_result = self.env.step(actions)

        # Handle different step return formats
        if len(step_result) == 4:
            obs, rewards, dones, info = step_result
        elif len(step_result) == 3:
            obs, rewards, dones = step_result
            info = {}
        else:
            raise ValueError(f"Unexpected step return format: {len(step_result)} values")

        # Update buffers
        self.obs_buf = obs.to(self.device)
        self.rew_buf = rewards.to(self.device)
        self.reset_buf = dones.to(self.device)
        self.episode_length_buf += 1

        # Handle episode resets
        if self.reset_buf.any():
            self.episode_length_buf[self.reset_buf] = 0

        # Update extras with info from the environment
        self.extras.update(info)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_attr(self, attr_name, indices=None):
        """Get environment attribute."""
        return getattr(self.env, attr_name)

    def set_attr(self, attr_name, value, indices=None):
        """Set environment attribute."""
        setattr(self.env, attr_name, value)

    def eval(self, runner, cfg):
        # Evaluation mode - run a simple evaluation loop
        print("Running evaluation...")
        runner.eval_mode()  # Switch to evaluation mode
        
        # Get the actor-critic directly from the algorithm
        actor_critic = runner.alg.actor_critic
        
        # Check if empirical normalization is used
        use_normalization = hasattr(runner, 'empirical_normalization') and runner.empirical_normalization
        if use_normalization:
            obs_normalizer = runner.obs_normalizer
            print("Using empirical observation normalization")
        else:
            obs_normalizer = None
            print("No observation normalization")
        
        # Ensure environment is in evaluation mode
        if hasattr(runner.env.env, 'set_eval'):
            runner.env.env.set_eval(True)
        if hasattr(runner.env.env, 'eval'):
            runner.env.env.eval()
        
        # Set deterministic actions for evaluation
        if hasattr(runner.alg, 'set_eval'):
            runner.alg.set_eval(True)
        
        # Run evaluation for a few episodes
        num_eval_episodes = 10
        total_reward = 0.0
        total_steps = 0
        
        # Get episode length from environment config
        max_episode_length = getattr(runner.env.env, 'episode_length', 1000)
        print(f"Max episode length: {max_episode_length}")

        for episode in range(num_eval_episodes):
            obs, extras = runner.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False
            
            # Reset episode-specific variables
            if hasattr(runner.env.env, 'reset_episode'):
                runner.env.env.reset_episode()

            while not done and episode_steps < max_episode_length:
                # Normalize observations if normalization is used
                if use_normalization:
                    normalized_obs = obs_normalizer(obs)
                else:
                    normalized_obs = obs
                
                # Get action from the policy using torch inference mode
                with torch.inference_mode():
                    actions = actor_critic.act_inference(normalized_obs)
                
                obs, rewards, dones, infos = runner.env.step(actions)
                episode_reward += rewards.sum().item()
                episode_steps += 1
                done = dones.any()

                # Render if requested
                if cfg.general.render:
                    runner.env.env.render()

            # Handle episode termination
            if episode_steps >= max_episode_length:
                print(f"Episode {episode + 1} terminated at max length {max_episode_length}")
            
            total_reward += episode_reward
            total_steps += episode_steps
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")

        print(f"Average reward over {num_eval_episodes} episodes: {total_reward / num_eval_episodes:.2f}")
        print(f"Average steps over {num_eval_episodes} episodes: {total_steps / num_eval_episodes:.1f}")
        
        # Restore training mode if needed
        if hasattr(runner.env.env, 'set_eval'):
            runner.env.env.set_eval(False)
        if hasattr(runner.alg, 'set_eval'):
            runner.alg.set_eval(False)
