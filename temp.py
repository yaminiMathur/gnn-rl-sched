
    def initialize_behavior_function(state_size, 
                                    action_size, 
                                    hidden_size, 
                                    learning_rate, 
                                    command_scale):
            '''
            Initialize the behaviour function. See section 2.3.2
            
            Params:
                state_size (int)
                action_size (int)
                hidden_size (int) -- NOTE: not used at the moment
                learning_rate (float)
                command_scale (List of float)
            
            Returns:
                Behavior instance
            
            '''
            
            behavior = Behavior(state_size, 
                                action_size, 
                                hidden_size, 
                                command_scale)
            
            behavior.init_optimizer(lr=learning_rate)
            print("\n\nInitialized Behavior Function...")
            return behavior

        ### Generate Episode
    def generate_episode(env, policy, init_command=[1, 1]):
            '''
            Generate an episode using the Behaviour function.
            
            Params:
                env (OpenAI Gym Environment)
                policy (func)
                init_command (List of float) -- default [1, 1]
            
            Returns:
                Namedtuple (states, actions, rewards, init_command, total_return, length)
            '''
            
            command = init_command.copy()
            desired_return = command[0]
            desired_horizon = command[1]
            
            states = []
            actions = []
            rewards = []
            
            time_steps = 0
            done = False
            total_rewards = 0
            state = env.reset().tolist()
            
            while not done:
                state_input = torch.FloatTensor(state).to(device)
                command_input = torch.FloatTensor(command).to(device)
                action = policy(state_input, command_input)
                next_state, reward, done, _ = env.step(action)
                
                # Modifying a bit the reward function punishing the agent, -100, 
                # if it reaches hyperparam max_steps. The reason I'm doing this 
                # is because I noticed that the agent tends to gather points by 
                # landing the spaceshipt and getting out and back in the landing 
                # area over and over again, never switching off the engines. 
                # The longer it does that the more reward it gathers. Later on in 
                # the training it realizes that it can get more points by turning 
                # off the engines, but takes more epochs to get to that conclusion.
                if not done and time_steps > max_steps:
                    done = True
                    reward = max_steps_reward
                
                # Sparse rewards. Cumulative reward is delayed until the end of each episode
        #         total_rewards += reward
        #         reward = total_rewards if done else 0.0
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state.tolist()
                
                # Clipped such that it's upper-bounded by the maximum return achievable in the env
                desired_return = min(desired_return - reward, max_reward)
                
                # Make sure it's always a valid horizon
                desired_horizon = max(desired_horizon - 1, 1)
            
                command = [desired_return, desired_horizon]
                time_steps += 1
                
            return make_episode(states, actions, rewards, init_command, sum(rewards), time_steps)

    # def generate_episodes(env, behavior, buffer, n_episodes, last_few):
    #     '''
    #         1. Sample exploratory commands based on replay buffer
    #         2. Generate episodes using Algorithm 2 and add to replay buffer
            
    #         Params:
    #             env (OpenAI Gym Environment)
    #             behavior (Behavior)
    #             buffer (ReplayBuffer)
    #             n_episodes (int)
    #             last_few (int):
    #                 how many episodes we use to calculate the desired return and horizon
    #         '''
            
    #      stochastic_policy = lambda state, command: behavior.action(state, command)
                
    #     for i in range(n_episodes_per_iter):
    #         command = sample_command(buffer, last_few)
    #         episode = generate_episode(env, stochastic_policy, command) # See Algorithm 2
    #         buffer.add(episode)
                
    #         # Let's keep this buffer sorted
    #         buffer.sort()


    ### Main Training Loop
    def learn_batch(selfenv, buffer=None, behavior=None, learning_history=[]):
        '''
    Upside-Down Reinforcement Learning main algrithm
            
    Params:
        env (OpenAI Gym Environment)
        buffer (ReplayBuffer):
            if not passed in, new buffer is created
        behavior (Behavior):
            if not passed in, new behavior is created
            learning_history (List of dict) -- default []
    '''
        
    if buffer is None:
        buffer = initialize_replay_buffer(replay_size, 
                                            n_warm_up_episodes, 
                                            last_few)
        
    if behavior is None:
        behavior = initialize_behavior_function(state_size, 
                                                    action_size, 
                                                    hidden_size, 
                                                    learning_rate, 
                                                    [return_scale, horizon_scale])
        
    for i in range(1, n_main_iter+1):
        mean_loss = train_behavior(behavior, buffer, n_updates_per_iter, batch_size)
            
        print('Iter: {}, Loss: {:.4f}'.format(i, mean_loss), end='\r')
            
        # Sample exploratory commands and generate episodes
        generate_episodes(env, 
                            behavior, 
                            buffer, 
                            n_episodes_per_iter,
                            last_few)
            
        if i % evaluate_every == 0:
            command = sample_command(buffer, last_few)
            mean_return = evaluate_agent(env, behavior, command)
                
            learning_history.append({
                    'training_loss': mean_loss,
                    'desired_return': command[0],
                    'desired_horizon': command[1],
                    'actual_return': mean_return,
                })
                
        if stop_on_solved and mean_return >= target_return: 
            break
        
        #return behavior, buffer, learning_history

    def save(self, episode=0):
        save_path = (self.save_dir + f"/sched_net_{self.aggregator}_{episode}.pt")
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"Sched_net saved to {save_path} at step {self.curr_step}")
