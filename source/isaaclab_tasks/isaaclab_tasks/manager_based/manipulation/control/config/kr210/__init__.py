import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Force-Limit-KR210-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.force_limit_cfg:KR210ForceLimitEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KR210ForceLimitPPORunnerCfg",
    },
)