from environment_wrapper import *
from dqn import Agent, MetricLogger
from numpy.random import randint

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

    logger.log_episode()

    if e % 5 == 0:
        logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)