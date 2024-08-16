import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import setup
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from actor import ActorNetwork
from critic import CriticNetwork
from keras.layers import Dense
from keras.initializers import HeNormal, GlorotNormal

class Agent:
    def __init__(self, input_dims, n_actions=None, prefix=""):
        
        # Training parameters
        self.gamma         = setup.GAMMA
        self.alpha         = setup.ALPHA
        self.beta          = setup.BETA
        self.tau           = setup.TAU
        self.batch_size    = setup.N
        self.noise         = setup.XI
        self.save          = True # For ensureness
        
        # Replay buffer memory
        self.memory        = ReplayBuffer(setup.D, input_dims, n_actions)
        
        self.n_actions     = n_actions
        
        # Saving models with prefix and unicode
        self.prefix        = prefix
        self.unicode       = np.random.randint(0, 9999999)
        
        # Networks
        self.actor         = ActorNetwork(name=self.prefix + 'actor', n_actions=n_actions)
        self.target_actor  = ActorNetwork(name=self.prefix + 'target_actor', n_actions=n_actions)
        self.critic        = CriticNetwork(name=self.prefix + 'critic')
        self.target_critic = CriticNetwork(name=self.prefix + 'target_critic')
        
        # Build networks
        self.actor.compile(optimizer=Adam(learning_rate=self.alpha))
        self.critic.compile(optimizer=Adam(learning_rate=self.beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=self.alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=self.beta))
        
        self.update_network_parameters(tau=1)
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)
 
        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def set_prefix(self, prefix="", unicode=True):
        if unicode:
            prefix = "%s"%self.unicode + '_' + prefix + '_'
        else:
            prefix = prefix + '_'
        
        self.actor.model_name         = prefix + self.actor.model_name
        self.target_actor.model_name  = prefix + self.target_actor.model_name
        self.critic.model_name        = prefix + self.critic.model_name
        self.target_critic.model_name = prefix + self.target_critic.model_name
        
        self.actor.checkpoint_file        =\
            os.path.join(self.actor.checkpoint_dir, self.actor.model_name + '_ddpg.keras')
        self.target_actor.checkpoint_file  =\
            os.path.join(self.target_actor.checkpoint_dir, self.target_actor.model_name + '_ddpg.keras')
        self.critic.checkpoint_file        =\
            os.path.join(self.critic.checkpoint_dir, self.critic.model_name + '_ddpg.keras')
        self.target_critic.checkpoint_file =\
            os.path.join(self.target_critic.checkpoint_dir, self.target_critic.model_name + '_ddpg.keras')
           
    def save_models(self):
        # print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)
        
    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)
    
    def set_weights(self):
        
        # HeNormal is designed for relu activation function. 
        # it sets the weights based on the number of input neurons
        for i, layer in enumerate(self.actor.layers):
            if isinstance(layer, Dense):
                if i == len(self.actor.layers) - 1: 
                    initializer = GlorotNormal(seed=42)
                else:
                    initializer = HeNormal(seed=42)
                    
                weights = [initializer(shape=w.shape) for w in layer.get_weights()]
                layer.set_weights(weights)
        
        for i, layer in enumerate(self.target_actor.layers):
            if isinstance(layer, Dense):
                if i == len(self.target_actor.layers) - 1: 
                    initializer = GlorotNormal(seed=42)
                else:
                    initializer = HeNormal(seed=42)
                    
                weights = [initializer(shape=w.shape) for w in layer.get_weights()]
                layer.set_weights(weights)
                
        for i, layer in enumerate(self.critic.layers):
            if isinstance(layer, Dense):
                if i == len(self.target_actor.layers) - 1: 
                    initializer = GlorotNormal(seed=42)
                else:
                    initializer = HeNormal(seed=42)
                    
                weights = [initializer(shape=w.shape) for w in layer.get_weights()]
                layer.set_weights(weights)
                
        for i, layer in enumerate(self.target_critic.layers):
            if isinstance(layer, Dense):
                if i == len(self.target_actor.layers) - 1: 
                    initializer = GlorotNormal(seed=42)
                else:
                    initializer = HeNormal(seed=42)
                    
                weights = [initializer(shape=w.shape) for w in layer.get_weights()]
                layer.set_weights(weights)
                        
    def choose_action(self, observation, evaluate=False):

        state = tf.convert_to_tensor([observation], dtype=tf.float32)

        action = self.actor(state)

        # Adding a random Gaussian noise
        if not evaluate:
            action += tf.random.normal(shape=[len(self.n_actions)], mean=0.0, stddev=self.noise)
            
        return action[0]

    def learn(self):
            
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states  = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            
            target_actions = self.target_actor(states_)
            critic_value_  = tf.squeeze(self.target_critic(states_, target_actions), 1)
            critic_value   = tf.squeeze(self.critic(states, actions), 1)
            target         = rewards + self.gamma*critic_value_*(1-done)
            critic_loss    = keras.losses.MSE(target, critic_value)
            
        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:

            new_policy_actions = self.actor(states)
            actor_loss         = -1 * self.critic(states, new_policy_actions)
            actor_loss         = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()
        
        if self.save:
            self.save = False
            self.save_models()