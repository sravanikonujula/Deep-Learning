import gym
env = gym.make('MountainCar-v0')
for i_episode in range(225):
    obs = env.reset()
    for v in range(100):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            print(" Score obtained {} timesteps".format(v+1))
        else:
            print(" Failed to give Score..!! ")
            break