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

        # Run evaluation for a few episodes
        num_eval_episodes = 10
        total_reward = 0.0
        total_steps = 0

        for episode in range(num_eval_episodes):
            obs, extras = runner.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False

            while not done:
                # Get action from the trained policy
                actions = runner.alg.act(obs, obs)  # Use obs for both actor and critic
                obs, rewards, dones, infos = runner.env.step(actions)
                episode_reward += rewards.sum().item()
                episode_steps += 1
                done = dones.any()

                # Render if requested
                if cfg.general.render:
                    runner.env.env.render()

            total_reward += episode_reward
            total_steps += episode_steps
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")

        print(f"Average reward over {num_eval_episodes} episodes: {total_reward / num_eval_episodes:.2f}")
        print(f"Average steps over {num_eval_episodes} episodes: {total_steps / num_eval_episodes:.1f}")
