import hydra, os, wandb, yaml, shutil
from IPython.core import ultratb
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from shac.utils import hydra_utils
from hydra.utils import instantiate
from shac.utils.common import *
from shac.utils.rlgames_utils import (
    RLGPUEnvAlgoObserver,
    RLGPUEnv,
)
from gym import wrappers
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv
from omegaconf import OmegaConf, open_dict

try:
    from svg.train import Workspace
except:
    print_warning("SVG not installed")

# Expose sim2mujoco envs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# enables ipdb when script crashes
# sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)


def register_envs(env_config):
    def create_dflex_env(**kwargs):
        # create env without grads since PPO doesn't need them
        env = instantiate(env_config.config, no_grad=True)

        print("num_envs = ", env.num_envs)
        print("num_actions = ", env.num_actions)
        print("num_obs = ", env.num_obs)

        frames = kwargs.pop("frames", 1)
        if frames > 1:
            env = wrappers.FrameStack(env, frames, False)

        return env

    def create_warp_env(**kwargs):
        # create env without grads since PPO doesn't need them
        env = instantiate(env_config.config, no_grad=True)

        print("num_envs = ", env.num_envs)
        print("num_actions = ", env.num_actions)
        print("num_obs = ", env.num_obs)

        frames = kwargs.pop("frames", 1)
        if frames > 1: 
            env = wrappers.FrameStack(env, frames, False)

        return env

    vecenv.register(
        "DFLEX",
        lambda config_name, num_actors, **kwargs: RLGPUEnv(
            config_name, num_actors, **kwargs
        ),
    )
    env_configurations.register(
        "dflex",
        {
            "env_creator": lambda **kwargs: create_dflex_env(**kwargs),
            "vecenv_type": "DFLEX",
        },
    )

    vecenv.register(
        "WARP",
        lambda config_name, num_actors, **kwargs: RLGPUEnv(
            config_name, num_actors, **kwargs
        ),
    )
    env_configurations.register(
        "warp",
        {
            "env_creator": lambda **kwargs: create_warp_env(**kwargs),
            "vecenv_type": "WARP",
        },
    )


def create_wandb_run(wandb_cfg, job_config, run_id=None):
    env_name = job_config["env"]["config"]["_target_"].split(".")[-1]
    try:
        alg_name = job_config["alg"]["_target_"].split(".")[-1]
    except:
        alg_name = job_config["alg"]["name"].upper()
    try:
        # Multirun config
        job_id = HydraConfig().get().job.num
        name = f"{alg_name}_{env_name}_sweep_{job_config['general']['seed']}"
        notes = wandb_cfg.get("notes", None)
    except:
        # Normal (singular) run config
        name = f"{alg_name}_{env_name}"
        notes = wandb_cfg["notes"]  # force user to make notes
    return wandb.init(
        project=wandb_cfg.project,
        config=job_config,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        name=name,
        notes=notes,
        id=run_id,
        resume=run_id is not None,
    )


cfg_path = os.path.dirname(__file__)
cfg_path = os.path.join(cfg_path, "cfg")


@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig):
    cfg_full = OmegaConf.to_container(cfg, resolve=True)

    if cfg.general.run_wandb:
        create_wandb_run(cfg.wandb, cfg_full)

    # patch code to make jobs log in the correct directory when doing multirun
    # logdir = HydraConfig.get()["runtime"]["output_dir"]
    # main_logdir = cfg.general.logdir
    # logdir = os.path.join(main_logdir, logdir, "logs")
    # print(logdir, main_logdir)

    # 1) your base “main” log-dir from your config
    main_logdir = cfg.general.logdir

    # 2) get now’s date & time strings
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")  # e.g. "2025-08-07"
    time_str = now.strftime("%H-%M-%S.%f")  # e.g. "14-30-05"

    # 3) compose <main_logdir>/<date>/<time>/logs
    logdir = os.path.join("outputs", main_logdir, date_str, time_str, "logs")

    # 4) make sure it exists
    os.makedirs(logdir, exist_ok=True)
    print(f"Writing logs to: {logdir}")

    # 5) Copy everything from Hydra's output directory to the parent of our log directory
    # This ensures that both the SHAC policies/logs and Hydra files are in the same directory structure
    # Get Hydra's output directory
    hydra_output_dir = HydraConfig.get().run.dir
    if hydra_output_dir and os.path.exists(hydra_output_dir):
        # Copy everything from Hydra's output directory to the parent of our log directory
        target_dir = os.path.dirname(logdir)  # This is logdir/..

        # Copy all files and subdirectories recursively
        for item in os.listdir(hydra_output_dir):
            src_path = os.path.join(hydra_output_dir, item)
            dst_path = os.path.join(target_dir, item)
            
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
            elif os.path.isdir(src_path):
                if os.path.exists(dst_path):
                    # If directory exists, copy contents
                    for subitem in os.listdir(src_path):
                        sub_src = os.path.join(src_path, subitem)
                        sub_dst = os.path.join(dst_path, subitem)
                        if os.path.isfile(sub_src):
                            shutil.copy2(sub_src, sub_dst)
                            print(f"Copied file: {subitem} to {dst_path}")
                        elif os.path.isdir(sub_src):
                            shutil.copytree(sub_src, sub_dst, dirs_exist_ok=True)
                            print(f"Copied directory: {subitem} to {dst_path}")
                else:
                    # If directory doesn't exist, copy the whole directory
                    shutil.copytree(src_path, dst_path)
                    print(f"Copied directory: {item} to {target_dir}")

    seeding(cfg.general.seed)

    # First check if we are doing a sim2mujoco transfer (training or normal play)
    if cfg.general.sim2mujoco:
        # If PPO or SHAC
        if cfg.alg.name == "ppo" or cfg.alg.name == "sac":
            # Change mode to play and use the sim2mujoco env
            cfg.env.config.no_grad = True

            # first shuffle around config structure
            cfg_train = cfg_full["alg"]
            cfg_train["params"]["general"] = cfg_full["general"]
            env_name = cfg_train["params"]["config"]["env_name"]
            cfg_train["params"]["diff_env"] = cfg_full["env"]["config"]
            cfg_train["params"]["general"]["logdir"] = logdir

            # boilerplate to get rl_games working
            # Set mode to play even if train is true
            cfg_train["params"]["general"]["play"] = True
            cfg_train["params"]["general"]["train"] = False

            # Set num_envs and eval games correctly based on mujoco settings
            num_envs = cfg["env"]["mujoco"]["config"]["num_envs"]
            cfg["env"]["ppo"]["num_actors"] = num_envs
            num_games = cfg["env"]["mujoco"]["config"]["num_games"]
            cfg_train["params"]["config"]["player"]["games_num"] = num_games

            # Now handle different env instantiation
            if env_name.split("_")[0] == "df":
                cfg_train["params"]["config"]["env_name"] = "dflex"
            elif env_name.split("_")[0] == "warp":
                cfg_train["params"]["config"]["env_name"] = "warp"
            env_name = cfg_train["params"]["diff_env"]["_target_"]
            cfg_train["params"]["diff_env"]["name"] = env_name.split(".")[-1]

            # save config
            if cfg_train["params"]["general"]["train"]:
                os.makedirs(logdir, exist_ok=True)
                yaml.dump(cfg_train, open(os.path.join(logdir, "cfg.yaml"), "w"))

            # register envs with the correct number of actors for PPO
            if cfg.alg.name == "ppo":
                cfg["env"]["config"]["num_envs"] = cfg["env"]["ppo"]["num_actors"]
            else:
                cfg["env"]["config"]["num_envs"] = cfg["env"]["sac"]["num_actors"]

            register_envs(cfg.env.mujoco)

            # add observer to score keys
            if cfg_train["params"]["config"].get("score_keys"):
                algo_observer = RLGPUEnvAlgoObserver()
            else:
                algo_observer = None

            runner = Runner(algo_observer)
            runner.load(cfg_train)
            runner.reset()
            runner.run(cfg_train["params"]["general"])
        else:
            cfg.env.config.no_grad = True

            # Set mode to play even if train is true
            # cfg["general"]["play"] = True
            cfg["general"]["train"] = False

            # Set num_envs and eval games correctly based on mujoco settings
            num_envs = cfg["env"]["mujoco"]["config"]["num_envs"]
            cfg["env"]["config"]["num_envs"] = num_envs
            num_games = cfg["env"]["mujoco"]["config"]["num_games"]
            cfg["env"]["player"]["games_num"] = num_games

            algo = instantiate(cfg.alg, env_config=cfg.env.mujoco.config, logdir=logdir)

            if cfg.general.checkpoint:
                algo.load(cfg.general.checkpoint)

            if cfg.general.train:
                algo.train()
            else:
                algo.run(cfg.env.player.games_num)

    elif "_target_" in cfg.alg:
        cfg.env.config.no_grad = not cfg.general.train

        algo = instantiate(cfg.alg, env_config=cfg.env.config, logdir=logdir)

        if cfg.general.checkpoint:
            algo.load(cfg.general.checkpoint)

        if cfg.general.train:
            algo.train()
        else:
            algo.run(cfg.env.player.games_num)

    elif cfg.alg.name == "ppo" or cfg.alg.name == "sac":
        # if not hydra init, then we must have PPO
        # to set up RL games we have to do a bunch of config menipulation
        # which makes it a huge mess...

        # PPO doesn't need env grads
        cfg.env.config.no_grad = True

        # first shuffle around config structure
        cfg_train = cfg_full["alg"]
        cfg_train["params"]["general"] = cfg_full["general"]
        env_name = cfg_train["params"]["config"]["env_name"]
        cfg_train["params"]["diff_env"] = cfg_full["env"]["config"]
        cfg_train["params"]["general"]["logdir"] = logdir
        cfg_train["params"]["config"]["train_dir"] = logdir

        # boilerplate to get rl_games working
        cfg_train["params"]["general"]["play"] = not cfg_train["params"]["general"][
            "train"
        ]

        # Now handle different env instantiation
        if env_name.split("_")[0] == "df":
            cfg_train["params"]["config"]["env_name"] = "dflex"
        elif env_name.split("_")[0] == "warp":
            cfg_train["params"]["config"]["env_name"] = "warp"
        env_name = cfg_train["params"]["diff_env"]["_target_"]
        cfg_train["params"]["diff_env"]["name"] = env_name.split(".")[-1]

        # save config
        if cfg_train["params"]["general"]["train"]:
            os.makedirs(logdir, exist_ok=True)
            yaml.dump(cfg_train, open(os.path.join(logdir, "cfg.yaml"), "w"))

        # register envs with the correct number of actors for PPO
        if cfg.alg.name == "ppo":
            cfg["env"]["config"]["num_envs"] = cfg["env"]["ppo"]["num_actors"]
        else:
            cfg["env"]["config"]["num_envs"] = cfg["env"]["sac"]["num_actors"]
        register_envs(cfg.env)

        # add observer to score keys
        if cfg_train["params"]["config"].get("score_keys"):
            algo_observer = RLGPUEnvAlgoObserver()
        else:
            algo_observer = None
        runner = Runner(algo_observer)
        runner.load(cfg_train)
        runner.reset()
        runner.run(cfg_train["params"]["general"])
    elif cfg.alg.name == "svg":
        cfg.env.config.no_grad = True
        with open_dict(cfg):
            cfg.alg.env = cfg.env.config
        w = Workspace(cfg.alg)
        w.run_epochs()
    else:
        raise NotImplementedError

    if cfg.general.run_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
