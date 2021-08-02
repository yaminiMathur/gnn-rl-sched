from environment_wrapper import *
from dqn import Agent, MetricLogger
from numpy.random import randint
import time

# ------------------------------------------------------------------------------------------------------------ #
#                                                  DQN
# ------------------------------------------------------------------------------------------------------------ #
def dqn_train(load_path=None, episodes=31):
    env = GraphWrapper()
    agent = Agent()
    logger = MetricLogger()

    # load the presaved model
    if load_path:
        agent.load(load_path, exploration_rate=True)

    for e in range(episodes):
        
        seed = randint(12345)
        env.reset(seed)
        state = env.observe()
        done = False
        start = time.time()
        total_reward = 0
        total_actions = 0

        # Run the simulation
        while True:

            # Run agent on the state
            action = agent.act(state)

            # Agent performs action
            next_state, reward, done = env.step((action, 1), False)
            total_reward += reward
            total_actions += 1

            # Remember
            agent.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = agent.learn_batch()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            if done:
                break
        
        agent.save(3)
        torch.cuda.empty_cache()
        logger.log_episode()

        print('--------------------------------------------------------------------------------------------------------------------------------------------')
        print("Completed episode :", e, "Time Taken:", time.time()-start, "reward", total_reward, "seed :", seed, "actions :", total_actions)
        logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)
        print('--------------------------------------------------------------------------------------------------------------------------------------------')

dqn_train(load_path="sched_net_2.pt", episodes=30)