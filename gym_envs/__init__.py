from gym.envs.registration import register

register(
    id='HexapodEnv-v0',
    entry_point='gym_envs.bullet:HexapodEnv',
)