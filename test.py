from numpy import average
from environment_wrapper import *
from dqn import Agent, MetricLogger
from numpy.random import randint
import time
from matplotlib import pyplot as plt
from spark_env.canvas import *

def dynamic(env:GraphWrapper, seed:int):
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
    
    visualize_dag_time_save_pdf(env.env.finished_job_dags, env.env.executors, "./results/test/dag_time_dynamic_makespan.png")
    visualize_executor_usage(env.env.finished_job_dags, "./results/test/exec_time_dynamic_makespan.png")
    
    print("---------------------------------------------------------------------------------------------")
    print("Dynamic Scheduling Baseline -", " Reward :", total_reward, " Actions :", actions, " Seed : ", seed)
    print("---------------------------------------------------------------------------------------------")
    return total_reward


def agent_action(env:GraphWrapper, seed:int, agent:Agent, e:int):
    total_reward = 0
    done = False
    env.reset(seed)
    state = env.observe()
    action_time = 0
    actions = 0
    # Run the simulation
    while True:

        start = time.time()
        # Run agent on the state
        action = agent.act(state)
        action_time += time.time() - start
        actions += 1

        # Agent performs action
        next_state, reward, done = env.step(action, False)
        total_reward += reward

        # Update state
        state = next_state

        if done:
            break

    
    visualize_dag_time_save_pdf(env.env.finished_job_dags, env.env.executors, "./results/test/dag_time_makespan.png")
    visualize_executor_usage(env.env.finished_job_dags, "./results/test/exec_time_makespan.png")

    print("episode reward : ", total_reward, "total action time : ", action_time, "total actions : ", actions)
    return total_reward


def dqn_test(load_path="sched_net_mean_3.pt", episodes=25, aggregator="mean"):
    env = GraphWrapper()
    agent = Agent()
    # logger = MetricLogger(version="0_old", mode="test", aggregator=aggregator)

    # only exploit 
    agent.exploration_rate_min = 0.0
    agent.load(load_path, set=True, new_rate=0.0)

    # for calculating cdf for dynamic and agent actions
    dynamic_rewards = []
    agent_rewards = []

    for e in range(episodes):
        seed = randint(12345)
        dynamic_rewards.append(dynamic(env, seed))
        agent_rewards.append(agent_action(env, seed, agent, e))

    return dynamic_rewards, agent_rewards


r1, r2 = dqn_test(episodes=1)
count_1, bins_1 = np.histogram(np.array(r1), bins=10)
pdf_1 = count_1 / sum(count_1)
cdf_1 = np.cumsum(pdf_1)

count_2, bins_2 = np.histogram(np.array(r2), bins=10)
pdf_2 = count_2 / sum(count_2)
cdf_2 = np.cumsum(pdf_2)

plt.plot(bins_1[1:], cdf_1, color="red", label="Dynamic Sched")
plt.plot(bins_2[1:], cdf_2, label="GNN sched")
plt.legend()
plt.savefig('./results/cdf_combined_11.png')

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