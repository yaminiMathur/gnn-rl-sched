from environment_wrapper import *
from dqn import Agent, MetricLogger
from numpy.random import randint
import time

# def save_graph(graph, pos, file_name):
#     plt.figure(num=None, figsize=(20, 20), dpi=80)
#     plt.axis('off')
#     fig = plt.figure(1)
#     nx.draw_networkx_nodes(graph,pos)
#     nx.draw_networkx_edges(graph,pos)
#     nx.draw_networkx_labels(graph,pos)

#     plt.savefig(file_name,bbox_inches="tight")
#     pylab.close()
#     del fig


# ------------------------------------------------------------------------------------------------------------ #
#                                                  DQN
# ------------------------------------------------------------------------------------------------------------ #
env = GraphWrapper()
agent = Agent()
logger = MetricLogger()

episodes = 10
for e in range(episodes):

    env.reset()
    state = env.observe()
    done = False
    start = time.time()

    # Run the simulation
    while True:

        # Run agent on the state
        action = agent.act(state)

        # Agent performs action
        next_state, reward, done = env.step((action, 1), False)

        # Remember
        agent.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = agent.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        if done:
            break

    print("Completed episode :", e, "Time Taken:", time.time()-start)
    logger.log_episode()

    if e % 5 == 0:
        logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)

# ------------------------------------------------------------------------------------------------------------ #
#                                                  Test
# ------------------------------------------------------------------------------------------------------------ #
# episodes = 10
# env = GraphWrapper()
# for e in range(episodes):

#     env.reset()
#     state = env.observe()
#     done = False
#     episode_reward = 0
#     actions = 0
#     start = time.time()

#     # Run the simulation
#     while True:

#         G, node_inputs, leaf_nodes = state
#         if len(leaf_nodes) > 0:
#             action = leaf_nodes[randint(len(leaf_nodes))]
#         else :
#             action = -1
#         # Agent performs action
#         next_state, reward, done = env.step((action, randint(1, 10)), True)

#         # Update state
#         state = next_state
#         episode_reward += reward
#         actions += 1

#         if done:
#             print("episode : ",e, "episode reward", episode_reward, "actions", actions, "time", time.time()-start)
#             break