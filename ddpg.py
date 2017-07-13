"""Implements the deep deterministic policy gradient algorithm from:

Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement
learning." arXiv preprint arXiv:1509.02971 (2015).

Currently setup to run on OpenAI Pendulum-v0 environment.
"""
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as nn
from lasagne.regularization import regularize_network_params, l2

from replay import ReplayBuffer

W_init = lasagne.init.GlorotUniform()
b_init = lasagne.init.Constant(0.)
W_out = lasagne.init.Uniform(3e-3)
b_out = lasagne.init.Uniform(3e-3)
relu = lasagne.nonlinearities.rectify


def get_symbolic_var(state_shape):
    """Returns theano symbolic variable according to state space dimensions."""

    if not isinstance(state_shape, tuple):
        raise Exception('state_shape must be of type <tuple>.')
    elif len(state_shape) not in [1, 2, 3]:
        raise Exception('state shape must have n_dims in [1, 2, 3]')

    # Decides whether matrix or specific tensor rank
    return [T.matrix, T.tensor3, T.tensor4][len(state_shape) - 1]


class Actor(object):
    """Defines the actor in an actor-critic architecture."""
    def __init__(self, state_shape, num_actions, action_scale, lr, tau):

        self.state_shape = state_shape
        self.num_actions = num_actions
        self.action_scale = action_scale
        self.tau = tau

        # Build networks, then initialize their weights to be equal
        sym_s0 = get_symbolic_var(state_shape)('s0')
        sym_s1 = get_symbolic_var(state_shape)('s1')

        self.network = self._build_network(sym_s0)
        self.targets = self._build_network(sym_s1)
        self.update_target(tau=1.0)

        # For making predictions via current and target networks
        a_pred = nn.get_output(self.network)
        self.predict_fn = theano.function([sym_s0], a_pred)
        self.target_fn = theano.function([sym_s1], nn.get_output(self.targets))

        # The policy is updated by following gradients from critic network.
        # In theano, this is done by specifying the 'known_gradients' parameter
        # in T.grad, without giving an explicit scalar cost
        action_grads = T.col('action_grads')
        known_grads = {a_pred: action_grads}

        params = nn.get_all_params(self.network, trainable=True)
        grads = [-T.grad(None, p, known_grads=known_grads) for p in params]
        updates = lasagne.updates.adam(grads, params, lr)
        train = theano.function([sym_s0, action_grads], grads, updates=updates)
        self.train_fn = train

    def _build_network(self, input_var):
        """Builds actor network, scaling outputs based on action bounds."""

        tanh = lasagne.nonlinearities.ScaledTanH(1, scale_out=self.action_scale)

        network = nn.InputLayer((None, ) + self.state_shape, input_var)
        network = nn.DenseLayer(network, 30, W_init, b_init, relu)
        network = nn.DenseLayer(network, 30, W_init, b_init, relu)
        return nn.DenseLayer(network, self.num_actions, W_out, b_out, tanh)

    def predict(self, state):
        """Returns a prediction from the current policy."""
        if state.ndim == 1:
            state = state[np.newaxis].astype(theano.config.floatX)
        return self.predict_fn(state).astype(theano.config.floatX)

    def predict_target(self, state):
        """Returns a prediction from the target policy."""
        if state.ndim == 1:
            state = state[np.newaxis].astype(theano.config.floatX)
        return self.target_fn(state).astype(theano.config.floatX)

    def update_target(self, tau=None):
        """Updates target network by linearily blending network weights."""

        if tau is None:
            tau = self.tau

        aw = nn.get_all_param_values(self.network)
        tw = nn.get_all_param_values(self.targets)

        # Update weights for target networks by linearly blending
        weights = [tau * an + (1. - tau) * at for an, at in zip(aw, tw)]

        nn.set_all_param_values(self.targets, weights)


class Critic(object):
    """Defines the critic in an actor-critic architecture."""
    def __init__(self, state_shape, num_actions, discount, lr, tau, l2_decay):

        self.state_shape = state_shape
        self.num_actions = num_actions
        self.discount = discount
        self.tau = tau

        # Initialize some symbolic variables to interface with graphs easier 
        sym_s0 = get_symbolic_var(state_shape)('s0')
        sym_a0 = T.col('policy_actions')
        sym_s1 = get_symbolic_var(state_shape)('s1')
        sym_a1 = T.col('target_actions')
        sym_r = T.col('rewards')
        sym_t = T.col('terminal_state')
        sym_vars = [sym_s0, sym_a0, sym_s1, sym_a1, sym_r, sym_t]

        self.network = self._build_network(sym_s0, sym_a0)
        self.targets = self._build_network(sym_s1, sym_a1)
        self.update_target(tau=1.0)

        # Functions for sampling from current and target Q-functions
        q_pred = nn.get_output(self.network)
        q_target = nn.get_output(self.targets)

        self.predict_fn = theano.function([sym_s0, sym_a0], q_pred)
        self.target_fn = theano.function([sym_s1, sym_a1], q_target)

        # Calculate action gradients for updating actor / policy
        grads = T.grad(T.mean(q_pred), sym_a0)
        self.action_grads = theano.function([sym_s0, sym_a0], grads)

        # Build critic training function; loss is similar to DQN, where
        # it's the mean squared error between Q and target Q values
        yi = sym_r + (1. - sym_t) * self.discount * q_target
        loss = T.mean(T.sqr(yi - q_pred))
        loss += regularize_network_params(self.network, l2) * l2_decay

        params = nn.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adam(loss, params, lr)
        self.train_fn = theano.function(sym_vars, loss, updates=updates)

    def _build_network(self, state_var, action_var):
        """Builds critic network:; inputs: (state, action), outputs: Q-val."""

        # States -> Hidden
        state_in = nn.InputLayer((None, ) + self.state_shape, state_var)
        states = nn.DenseLayer(state_in, 30, W_init, b_init, relu)
        states = nn.DenseLayer(states, 30, W_init, b_init, nonlinearity=None)

        # Actions -> Hidden
        action_in = nn.InputLayer((None, self.num_actions), action_var)
        actions = nn.DenseLayer(action_in, 30, W_init, b=None, nonlinearity=None)

        # States_h + Actions_h -> Output
        net = nn.ElemwiseSumLayer([states, actions])
        net = nn.NonlinearityLayer(net, relu)
        return nn.DenseLayer(net, 1, W_out, b_out, nonlinearity=None)
    
    def get_action_grads(self, state, action):
        """Returns the action gradients from current Q-network."""
        if state.ndim == 1:
            state = state[np.newaxis]
        return self.action_grads(state, action)
    
    def update_target(self, tau=None):
        """Updates target network by linearily blending network weights."""
        if tau is None:
            tau = self.tau

        # Grab the weights for the current (n) and target (t) networks
        aw = nn.get_all_param_values(self.network)
        tw = nn.get_all_param_values(self.targets)

        weights = [tau * an + (1. - tau) * at for an, at in zip(aw, tw)]

        nn.set_all_param_values(self.targets, weights)


class Agent(object):
    """Implements an agent that follows DDPG algorithm."""
    def __init__(self, state_shape, num_actions, action_scale=2.0,
                 discount=0.99, tau=0.01, actor_lrate=0.001,
                 critic_lrate=0.01, l2_decay=1e-3, batch_size=64,
                 q_update_iter=1, capacity=1000000):

        if not isinstance(state_shape, tuple):
            raise AssertionError('state_shape must be of type <tuple>.')
        elif len(state_shape) == 0:
            raise AssertionError('No state space dimensions provided.')
        elif num_actions == 0:
            raise ValueError('Number of actions must be > 0.')
        elif capacity < batch_size:
            raise ValueError('Replay capacity must be > batch_size.')

        self.batch_size = batch_size
        self.q_update_iter = q_update_iter
        self.replay_buffer = ReplayBuffer(capacity, state_shape, num_actions)
        self.actor = Actor(state_shape, num_actions, action_scale, 
                           actor_lrate, tau)
        self.critic = Critic(state_shape, num_actions, discount, critic_lrate,
                             tau, l2_decay)
        self.step = 0

    def choose_action(self, state):
        """Returns an action for the agent to perform in the environment."""
        return self.actor.predict(state).flatten()

    def update_buffer(self, s0, a, r, s1, terminal):
        """Updates memory replay buffer with new experience."""
        self.replay_buffer.update(s0, a, r, s1, terminal)

    def update_policy(self):
        """Updates Q-networks using replay memory data + performing SGD"""

        mb = self.replay_buffer.sample(self.batch_size)

        # To update the critic, we need a prediction from target policy
        target_a = self.actor.predict_target(mb[3])
        self.critic.train_fn(mb[0], mb[1], mb[3], target_a, mb[2], mb[4])

        # Updating the actor requires gradients from critic
        action = self.actor.predict(mb[0])
        grads = self.critic.get_action_grads(mb[0], action)
        self.actor.train_fn(mb[0], grads)

        # Every few steps in an episode we update target network weights
        if self.step == self.q_update_iter:
            self.actor.update_target()
            self.critic.update_target()
        self.step = self.step + 1 if self.step != self.q_update_iter else 0
