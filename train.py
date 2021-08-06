from numpy import load
from environment_wrapper import *
from dqn import Agent, MetricLogger
from numpy.random import randint
import time

# ------------------------------------------------------------------------------------------------------------ #
#                                                  DQN
# ------------------------------------------------------------------------------------------------------------ #
def dqn_train(load_path=None, episodes=31, version=0, aggregator="mean"):
    env    = GraphWrapper()
    agent  = Agent(aggregator=aggregator)
    logger = MetricLogger(version=str(version), mode="train", aggregator=aggregator)

    # load the presaved model
    if load_path:
        agent.load(load_path, exploration_rate=True)

    prob = 100; index = 0

    for e in range(episodes):
        
        seed = randint(12345)
        env.reset(seed)
        state = env.observe()
        done = False
        start = time.time()
        total_reward = 0; total_actions = 0

        # Run the simulation
        while True:

            # Assist based on probability value
            if randint(0, 100) < prob:
                index = env.auto_step()
                action = agent.act(state, (index, 1))
            else:
                action = agent.act(state)
 
            # Agent performs action
            next_state, reward, done = env.step(action, False)
            total_reward += reward
            total_actions += 1

            # Remember
            agent.cache(state, next_state, action[0], reward, done)

            # Learn
            q, loss = agent.learn_batch()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            if done:
                break
        
        agent.save(version)
        torch.cuda.empty_cache()
        logger.log_episode()

        print('--------------------------------------------------------------------------------------------------------------------------------------------')
        print("Completed episode :", e, "Time Taken:", time.time()-start, "reward", total_reward, "seed :", seed, "actions :", total_actions)
        logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)
        print('--------------------------------------------------------------------------------------------------------------------------------------------')

dqn_train(episodes=70, version=2, aggregator="mean")
# ------------------------------------------------------------------------------------------------------------ #
#                                                Actor Critic
# ------------------------------------------------------------------------------------------------------------ #

# def actor_critic_train():
#     agent = Agent_AC()