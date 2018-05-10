import gym
envirnd = gym.make('CartPole-v0')
for i_episode in range(100):
    obs = envirnd.reset()
    for v in range(100):
        envirnd.render()
        #print(observation)
        oprtn = envirnd.action_space.sample()
        obs, reward, done, info = envirnd.step(oprtn)
        if done:
            print("score obtained :{}".format(v + 1))
            break