from ddpg import DDPGAgent
import numpy as np
from memory import MemoryBuffer
import gc
from environment import create_env


def test(device, args):

    env = create_env(args.env, args)
    ram = MemoryBuffer(1)
    player = DDPGAgent(env.observation_space, env.action_space, ram, None, device, args)
    if args.model_dir is not None:
        player.load_models(args.model_dir, test=True)
    steps_done = 0
    count_eps = 0
    count_success = 0
    while True:
        episode_rewards = []
        episode_lenghts = []
        for _ep in range(1, args.eval_eps):
            observation = env.reset()
            total_reward = 0
            episode_action = []
            for steps in range(1000):
                if 'img' in args.obs:
                    state = np.expand_dims(observation, axis=0)
                else:
                    state = np.float32(observation)

                action, action_rescale = player.get_exploitation_action(state)
                episode_action.append(action)
                new_observation, reward, done, info = env.step(action_rescale)
                observation = new_observation
                total_reward += reward
                steps_done += 1

                if args.render:
                    env.render()
                if done:
                    episode_rewards.append(total_reward)
                    count_eps += 1
                    episode_lenghts.append(steps)
                    if reward > 1:
                        count_success += 1.0
                    break
            # check memory consumption and clear memory
            gc.collect()

        reward_ave = np.array(episode_rewards).mean()
        length_ave = np.array(episode_lenghts).mean()
        print('Test, episode %d, steps: %d, Success_rate: %.3f ave_reward: %.3f, ave_length: %.3f' %
              (count_eps, steps_done, count_success/count_eps, reward_ave, length_ave))

    env.close()
