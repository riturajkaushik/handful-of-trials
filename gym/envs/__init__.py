from gym.envs.registration import register

register(
    id='HexapodEnv-v0',
    entry_point='gym.envs.bullet:HexapodEnv',
)