print("Importing Libraries... ")
from numpy import load
from environment_wrapper import *
from udrl import GCN, ReplayBuffer, MetricLogger
from agent import Agent
from behavior import Behavior
from numpy.random import randint
import time
print("------------------------------------------------------------------------------------------------------------")
print("                                                  UDRL")
print("------------------------------------------------------------------------------------------------------------")


# ------------------------------------------------------------------------------------------------------------ #
#                                                  UDRL
# ------------------------------------------------------------------------------------------------------------ #
print("!!! Entered training file !!!")

#def udrl_train(load_path=None, episodes=31, version=0, aggregator="mean", prob=100, exp_rate=True, new_rate=None):
def udrl_train_temp(load_path=None, episodes=31, version=0, aggregator="pool", prob=85, exp_rate=True, new_rate=None): 
    print("Entered the main UDRL training loop.")

    ### Hyperparameters ###
    ### Hyperparamters
    
    # Number of iterations in the main loop
    n_main_iter = 700

    # Number of (input, target) pairs per batch used for training the behavior function
    batch_size = 768

    # Scaling factor for desired horizon input
    horizon_scale = 0.01

    # Number of episodes from the end of the replay buffer used for sampling exploratory
    # commands
    last_few = 75

    # Learning rate for the ADAM optimizer
    learning_rate = 0.0003

    # Number of exploratory episodes generated per step of UDRL training
    n_episodes_per_iter = 20

    # Number of gradient-based updates of the behavior function per step of UDRL training
    n_updates_per_iter = 100

    # Number of warm up episodes at the beginning of training
    n_warm_up_episodes = 10

    # Maximum size of the replay buffer (in episodes)
    replay_size = 500

    # Scaling factor for desired return input
    return_scale = 0.02

    # Evaluate the agent after `evaluate_every` iterations
    evaluate_every = 10

    # Target return before breaking out of the training loop
    target_return = 200

    # Maximun reward given by the environment
    max_reward = 250

    # Maximun steps allowed
    max_steps = 300

    # Reward after reaching `max_steps` (punishment, hence negative reward)
    max_steps_reward = -50

    # Hidden units
    hidden_size = 32

    # Times we evaluate the agent
    n_evals = 1

    # Will stop the training when the agent gets `target_return` `n_evals` times
    stop_on_solved = False
    #######################
    env    = GraphWrapper()
    agent  = Agent(aggregator=aggregator)
    logger = MetricLogger(version=str(version), mode="train", aggregator=aggregator)
    # buffer = []
    #behavior = None
    learning_history=[]
    print("Environment, Agent, and Metric Logger instantiated successfully.\n")

    if buffer is None:
        buffer = agent.initialize_replay_buffer(env, replay_size, 
                                          n_warm_up_episodes, 
                                          last_few)
        print("Buffer initialized successfully.")
    
    # if behavior is None:
    #     behavior = behavior.initialize_behavior_function(state_size, 
    #                                             action_size, 
    #                                             hidden_size, 
    #                                             learning_rate, 
    #                                             [return_scale, horizon_scale])
    # print("Behavior initialized successfully.")



# assist seeds for mean : [9968, 8726, 11869, 6750, 5684, 6008, 9276, 5, 3283, 203, 5808, 5596, 6043, 9571, 5579]
# uncomment below 2 lines and set decay = 0.9998 in params to train for mean (also use assist seeds to give better training results)
# dqn_train(load_path=None, episodes=15, version=0, aggregator="mean", prob=100) # decay = 0.9998
# dqn_train(load_path="sched_net_pool_0.pt", episodes=50, version=1, aggregator="mean", prob=0, exp_rate=True)

# uncomment below 2 lines and set decay = 0.999992 and burnin to 1e4 in params to train for pool
#udrl_train(load_path=None, episodes=100, version=0, aggregator="pool", prob=85) # decay = 0.999992
udrl_train_temp(load_path=None, episodes=31, version=0, aggregator="pool", prob=85, exp_rate=True, new_rate=None) # decay = 0.999992
udrl_train_temp(load_path="sched_net_pool_0.pt", episodes=100, version=1, aggregator="pool", prob=85)