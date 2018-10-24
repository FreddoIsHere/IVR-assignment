from gym.envs.registration import register

register(
    id='3DReacherMy-v0',
    entry_point='reacher3D.Reacher:ReacherEnv',
)
