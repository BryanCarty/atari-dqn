from tf_agents.environments import suite_atari # Suite for loading Atari Gym environments
from tf_agents.environments.atari_preprocessing import AtariPreprocessing # Does the same preprocessing as was done on Nature's 2015 DQN paper
from tf_agents.environments.atari_wrappers import FrameStack4 # Stacks previous 4 frames
import tensorflow as tf
import numpy as np

max_episode_steps = 27000 # 4 frames per step, so a max of 108k frames per episode
environment_name = "BreakoutNoFrameskip-v4" # Environment does not have frame skipping by default

class AtariPreprocessingWithAutoFire(AtariPreprocessing):
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        super().step(1) # FIRE to start
        return obs
    def step(self, action):
        lives_before_action = self.ale.lives()
        obs, rewards, done, info = super().step(action) # done is true when a life is lost and terminal_on_life_loss, or when the episode is over.
        if self.ale.lives() < lives_before_action and not done: 
            super().step(1) # FIRE to start after life lost
        return obs, rewards, done, info

env = suite_atari.load( 
    environment_name, # Load a wrapped BreakoutNoFrameskip environment, with a max of 27,000 steps per episode.
    max_episode_steps=max_episode_steps, # AtariPreprocessing wrapper implements same preprocessing done in 2015 DQN paper e.g. convert to greyscale, scale to 84 x 84 pixels, implement frame skipping (4 frames). Hover over AtariPreprocessing to see the rest.
    gym_env_wrappers=[AtariPreprocessingWithAutoFire, FrameStack4] # In addition to frame skipping, these skipped frames are stacked, one on top of another (deciphers velocity). Stack size = 4.
)

from tf_agents.environments.tf_py_environment import TFPyEnvironment
tf_env = TFPyEnvironment(env) # We wrap the environment inside a TFPyEnvironment. This will make the environment usable within a Tensorflow graph.

from tf_agents.networks.q_network import QNetwork # Q Network -> takes an observation as input and outputs 1 Q-Value per action.
preprocessing_layer = tf.keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32)/255.) # This layer casts the observations to 32 bit floats and normalizes them (values will range from 0.0 -> 1.0)
conv_layer_params = [(32, (8,8), 4), (64, (4,4), 2), (64, (3,3), 1)] # 3 convolution layers -> a layer being (num filters, (filter dimensions), stride)
fc_layer_params = [512] # a dense layer with 512 units. This layer is followed by a dense layer with 4 units -> 1 per action. Outputs Q value.
                    # input tensor spec, action tensor spec, preprocessing layer, conv layer, fully connected layer params.
q_net = QNetwork(tf_env.observation_spec(), tf_env.action_spec(), preprocessing_layers=preprocessing_layer, conv_layer_params=conv_layer_params, fc_layer_params=fc_layer_params)

from tf_agents.agents.dqn.dqn_agent import DqnAgent # Implements the DQN algorithm from "Human level control through deep reinforcement learning" Mnih et al., 2015
train_step = tf.Variable(0) # counts the number of training steps
update_period = 4 # train the model every 4 steps
optimizer = tf.keras.optimizers.RMSprop(learning_rate=2.5e-4, rho=0.95, momentum=0.0, epsilon=0.00001, centered=True) # same hyperparameters as in 2015 DQN paper.
epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay( # will compute epsilon for a given training step. Will go from 1.0 to 0.001. The value used in 2015 DQN paper is 1,000,000 frames so 250,000 steps. But since we're training every 4 steps (16 frames), we'll decay over 62,500 training steps
    initial_learning_rate=1.0,
    decay_steps= 250000 // update_period, # number of decay steps # 1,000,000 ALE frames  # // Performs floor-division on the values on either side. Then assigns it to the expression on the left.
    end_learning_rate=0.01
)
agent = DqnAgent( # Creates a DQN Agent
    tf_env.time_step_spec(), 
    tf_env.action_spec(), 
    q_network=q_net, 
    optimizer=optimizer, 
    target_update_period=2000, # Period for soft update of the target networks. 2000 x 4 = 8000, 8000 x 4 = 32,000 ALE Frames.
    td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"), # The Huber loss function can be used to balance between the Mean Absolute Error and the Mean Squared Error. It is therefore a good loss function for when you have varied data or only a few outliers. Reduction is none so it returns error per instance not the mean error.
    gamma=0.99, # discount factor
    train_step_counter=train_step, # counts number of training steps
    epsilon_greedy=lambda: epsilon_fn(train_step) # function to compute epsilon for a given training step.
)
agent.initialize() # initialize the agent


from tf_agents.replay_buffers import tf_uniform_replay_buffer # A batched replay buffer of nests of Tensors which can be sampled uniformly. Would like to implement a prioritized experience replay buffer if time permits.
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec, # specification of the data saved in the replay buffer.(trajectory spec) The DQN agent knows what the collected data looks like
    batch_size=tf_env.batch_size, # In our case it's 1.
    max_length=200000 # max size of the replay buffer -> This will require a lot of RAM.
)
''' 
from tf_agents.replay_buffers.py_hashed_replay_buffer import PyHashedReplayBuffer
replay_buffer = PyHashedReplayBuffer(
    data_spec=agent.collect_data_spec,
    capacity=1000000
)
'''


replay_buffer_observer = replay_buffer.add_batch # an observer that writes trajectorys to the replay buffer.

class ShowProgress: # custom observer to count steps. 
    def __init__(self, total):
        self.counter=0
        self.total=total
    def __call__(self, trajectory):
        if not trajectory.is_boundary:
            self.counter+=1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")

from tf_agents.metrics import tf_metrics 
train_metrics = [ # metrics
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),#counts undiscounted rewards
    tf_metrics.AverageEpisodeLengthMetric(),
]

from tf_agents.eval.metric_utils import log_metrics
import logging
logging.getLogger().setLevel(logging.INFO)
#log_metrics(train_metrics) # logs all metrics

from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver 
collect_driver = DynamicStepDriver(
    tf_env, # interact with this env
    agent.collect_policy, # using this policy
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period # The number of steps to take in the environment. Collect 4 steps for each training iteration.
)

from tf_agents.policies.random_tf_policy import RandomTFPolicy
initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
init_driver = DynamicStepDriver(
    tf_env, 
    initial_collect_policy, 
    observers=[replay_buffer.add_batch, ShowProgress(20000)],
    num_steps=20000 # 80,000 ALE frames
)
final_time_step, final_Policy_state = init_driver.run()


dataset = replay_buffer.as_dataset( # for our main loop, instead of calling the get_next() method, we use a tf.data.Dataset so we can benefit from parallelism and prefetching.
    sample_batch_size=64, # We sample 64 trajectories at each training step (2015 DQN paper), each with 2 steps, e.g. 1 full transition.
    num_steps=2,
    num_parallel_calls=3 # This dataset willprocess 3 elements in parallel, and prefetch 3 batches.
).prefetch(3) # While the model is executing training step s , the input pipeline is reading the data for step s+1 . Doing so reduces the step time to the maximum (as opposed to the sum) of the training and the time it takes to extract the data.

from tf_agents.utils.common import function
collect_driver.run = function(collect_driver.run) # convert main functions to tensorflow functions for performance benefits.
agent.train = function(agent.train)

def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size) # returns a nested object of type policy_state containing properly initialized Tensors.
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)


train_agent(n_iterations=100000)




# # # # # # # # # # # # # # # #
#   VISUALIZING PERFORMANCE   #
# # # # # # # # # # # # # # # #

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim


frames = []
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))
watch_driver = DynamicStepDriver(
    tf_env,
    agent.policy,
    observers=[save_frames, ShowProgress(1000)],
    num_steps=1000)
final_time_step, final_policy_state = watch_driver.run()

plot_animation(frames)


import PIL
import os
image_path = os.path.join("videos", "rl", "breakout.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames[:150]]
frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=30,
                     loop=0)

