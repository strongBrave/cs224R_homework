"""
READ-ONLY: Behavior cloning agent definition
"""
from cs224r.infrastructure.replay_buffer import ReplayBuffer
from cs224r.policies.MLP_policy import MLPPolicySL
from .base_agent import BaseAgent

class BCAgent(BaseAgent):
    """
    Attributes
    ----------
    actor : MLPPolicySL
        An MLP that outputs an agent's actions given its observations
    replay_buffer: ReplayBuffer
        A replay buffer which stores collected trajectories

    Methods
    -------
    train:
        Calls the actor update function
    add_to_replay_buffer:
        Updates a the replay buffer with new paths
    sample
        Samples a batch of trajectories from the replay buffer
    """
    def __init__(self, env, agent_params):
        super(BCAgent, self).__init__()

        # Initialize variables
        self.env = env
        self.agent_params = agent_params

        # Create policy class as our actor
        self.actor = MLPPolicySL(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            learning_rate=self.agent_params['learning_rate'],
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            self.agent_params['max_replay_buffer_size'])

    def train(self, ob_no, ac_na):
        """
        :param ob_no: batch_size x obs_dim batch of observations
        :param ac_na: batch_size x ac_dim batch of actions
        """
        # Training a behaviour cloning agent refers to updating its actor using
        # the given observations and corresponding action labels
        log = self.actor.update(ob_no, ac_na)  # HW1: you will modify this
        return log

    def add_to_replay_buffer(self, paths):
        """
        :param paths: paths to add to the replay buffer
        """
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        """
        :param batch_size: size of batch to sample from replay buffer
        """
        # HW1: you will modify this
        return self.replay_buffer.sample_random_data(batch_size)

    def save(self, path):
        """
        :param path: path to save
        """
        return self.actor.save(path)
