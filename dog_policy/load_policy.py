from dog_policy.dog_policy import DogPolicy


def load_policy(env):
    policy_args = {}
    if env is not None:
        policy_args['ob_space'] = env.observation_space
        policy_args['ac_space'] = env.action_space
    policy = DogPolicy(**policy_args)
    return policy


