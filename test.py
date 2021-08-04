from numpy import average
from environment_wrapper import *
from dqn import Agent, MetricLogger
from numpy.random import randint
import time

def fair(env:GraphWrapper, seed:int):
    total_reward = 0
    done = False
    actions = 0

    # set env to specific seed
    env.reset(seed)

    # Run the simulation
    while True:

        # perform action
        # next_state, reward, done = env.step((env.leaf_nodes[0], 1), False)
        reward, done = env.auto_step(step=True)
        total_reward += reward
        actions += 1

        if done:
            break
    
    print("---------------------------------------------------------------------------------------------")
    print("Fair Scheduling Baseline -", " Reward :", total_reward, " Actions :", actions, " Seed : ", seed)
    print("---------------------------------------------------------------------------------------------")


def agent_action(env:GraphWrapper, seed:int, agent:Agent, e:int, logger:MetricLogger):
    total_reward = 0
    done = False
    env.reset(seed)
    state = env.observe()
    # Run the simulation
    while True:

        # Run agent on the state
        action = agent.act(state)

        # Agent performs action
        next_state, reward, done = env.step(action, False)
        total_reward += reward

        logger.log_step(reward, None, None)

        # Update state
        state = next_state

        if done:
            break

    logger.log_episode()
    print(total_reward)
    logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)


def dqn_test(load_path="sched_net_0.pt", episodes=10):
    env = GraphWrapper()
    agent = Agent()
    logger = MetricLogger()

    # only exploit 
    agent.exploration_rate_min = 0.0
    agent.load(load_path, set=True, new_rate=0.0)

    for e in range(episodes):
        seed = randint(12345)
        fair(env, seed)
        agent_action(env, seed, agent, e, logger)

dqn_test(episodes=10)

# ------------------------------------------------------------------------------------------------------------ #
#                                                Env Test
# ------------------------------------------------------------------------------------------------------------ #
# def test_env():
#     episodes = 1
#     env = GraphWrapper()
#     total_reward = 0
#     total_actions = 0
#     for e in range(episodes):

#         env.reset()
#         state = env.observe()
#         done = False
#         episode_reward = 0
#         actions = 0
#         start = time.time()

#         # Run the simulation
#         while True:

#             state, node_inputs, leaf_nodes = state
            
#             # Agent performs action
#             next_state, reward, done = env.step((leaf_nodes[0], 1), False)

#             # Update state
#             state = next_state
#             episode_reward += reward
#             actions += 1
#             if done:
#                 print("episode : ",e, "episode reward", episode_reward, "actions", actions, "time", time.time()-start)
#                 break
        
#         total_reward += episode_reward
#         total_actions += actions
#         print("Average Reward :", total_reward/(e+1), "Average Length :", total_actions/(e+1))

# test_env()