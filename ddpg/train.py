from ddpg import DDPGAgent
import numpy as np
import os
from memory import MemoryBuffer, PrioMemoryBuffer
from tensorboardX import SummaryWriter
from datetime import datetime
import gc
from environment import create_env


def train(rank, device, args):
    current_time = datetime.now().strftime('%b%d_%H-%M')
    LOGGER_DIR = os.path.join(args.log_dir, args.env, current_time, 'Agent:{}'.format(rank))
    writer = SummaryWriter(LOGGER_DIR)
    MODEL_DIR = os.path.join(LOGGER_DIR, 'models')
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    env = create_env(args.env, args)

    if args.pri:
        ram = PrioMemoryBuffer(args.buffer_size)
    else:
        ram = MemoryBuffer(args.buffer_size)

    player = DDPGAgent(env.observation_space, env.action_space, ram, writer, device, args)
    if args.model_dir is not None:
        player.load_models(args.model_dir)
    steps_done = 0
    episode_rewards = []
    max_score = -9999
    count_eps = 0
    for _ep in range(1, args.max_eps):
        observation = env.reset()
        total_reward = 0
        count_eps += 1
        for r in range(10000):
            if 'img' in args.obs:
                state = np.expand_dims(observation, axis=0)
            else:
                state = np.float32(observation)
            action, action_rescale = player.get_exploration_action(state)
            new_observation, reward, done, info = env.step(action_rescale)
            steps_done += 1
            total_reward += reward
            ram.add(observation, np.expand_dims(action, axis=0), reward, new_observation)
            observation = new_observation
            # perform optimization
            if steps_done > args.start_learning:
                player.optimize()
            if done:
                break

        # logger
        writer.add_scalar('episode/reward', total_reward, steps_done)
        writer.add_scalar('episode/length', r, steps_done)
        episode_rewards.append(total_reward)
        if _ep % args.eval_eps == 0:
            reward_ave = np.array(episode_rewards).mean()
            print('Train, episode %d, steps: %d reward: %.3f,ave_reward: %.3f' % (count_eps, steps_done, episode_rewards[-1], reward_ave))
            if reward_ave > max_score:
                player.save_models(os.path.join(MODEL_DIR, 'best'))
                max_score = reward_ave
                print('Save Best!')
            else:
                player.save_models(os.path.join(MODEL_DIR, 'new'))
            episode_rewards = []
        # check memory consumption and clear memory
        gc.collect()
