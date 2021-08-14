from numpy import average
from environment_wrapper import *
from dqn import Agent, MetricLogger
from numpy.random import randint
import time
from matplotlib import pyplot as plt
from spark_env.canvas import *

def dynamic(env:GraphWrapper, seed:int, aggregator:str, e:int):
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
    
    visualize_dag_time_save_pdf(env.env.finished_job_dags, env.env.executors, "./results/test/"+str(e)+"_dag_time_dynamic_"+aggregator+".png")
    visualize_executor_usage(env.env.finished_job_dags, "./results/test/"+str(e)+"exec_time_dynamic_"+aggregator+".png")
    
    print("---------------------------------------------------------------------------------------------")
    print("Dynamic Scheduling Baseline -", " Reward :", total_reward, " Actions :", actions, " Seed : ", seed)
    print("---------------------------------------------------------------------------------------------")
    return total_reward

def agent_action(env:GraphWrapper, seed:int, agent:Agent, e:int, aggregator:str):
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

    
    visualize_dag_time_save_pdf(env.env.finished_job_dags, env.env.executors, "./results/test/"+str(e)+"dag_time_"+aggregator+".png")
    visualize_executor_usage(env.env.finished_job_dags, "./results/test/"+str(e)+"exec_time_"+aggregator+".png")

    print("episode reward : ", total_reward, "total action time : ", action_time, "total actions : ", actions)
    return total_reward

def dqn_test(load_path="final_models_mean/sched_net_2_mean.pt", episodes=25, aggregator="mean"):
    env = GraphWrapper()
    agent = Agent(aggregator=aggregator)
    # logger = MetricLogger(version="0_old", mode="test", aggregator=aggregator)

    # only exploit 
    agent.exploration_rate_min = 0.0
    agent.load(load_path, exploration_rate=False, new_rate=0.0)

    # for calculating cdf for dynamic and agent actions
    dynamic_rewards = []
    agent_rewards = []
    seeds = [8738,  9029, 182, 9832, 9335, 3162, 10212, 10523, 12083, 1380, 887, 1304, 6905, 7318, 7634, 4422, 5597, 8190, 10023, 11435, 7639, 3308, 12014, 906, 6027]
    for e in range(episodes):
        seed = seeds[e]
        dynamic_rewards.append(dynamic(env, seed, aggregator, e))
        agent_rewards.append(agent_action(env, seed, agent, e, aggregator))

    return dynamic_rewards, agent_rewards

# comment this line to stop testing with mean aggregator
r1, r2 = dqn_test(load_path="final_models_mean/sched_net_2_mean.pt", episodes=25, aggregator="mean")

# uncomment this line to test with pool aggregator 
# r1, r2 = dqn_test(load_path="final_models_pool/sched_net_pool_1.pt", episodes=25, aggregator="pool")


fig = plt.figure()
ax = fig.add_subplot(111)
x, y = compute_CDF(r1)
ax.plot(x, y, color="red", label="Dynamic Sched")
x, y = compute_CDF(r2)
ax.plot(x, y, label="GNN Sched")
plt.xlabel('Total reward')
plt.ylabel('CDF')
plt.legend(["Dynamic Sched", "GNN Sched"])
fig.savefig('./results/cdf_combined_pool_1.png')
