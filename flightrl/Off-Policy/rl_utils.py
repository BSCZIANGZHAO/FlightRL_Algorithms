import sys
import time

from tqdm import tqdm
import numpy as np
import torch
import collections
import random


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, total_iteration=10,
                           writer=None):
    return_list = []
    # if agent is SAC algorithm     save_path = "saved_sac" else save_path = "saved_DDPG"
    if agent.__class__.__name__ == "SAC":
        save_path = "saved_sac"
    elif agent.__class__.__name__ == "DDPG":
        save_path = "saved_DDPG"
    elif agent.__class__.__name__ == "TD3":
        save_path = "saved_TD3"
    env_name = env.__class__.__name__

    for i in range(total_iteration+1):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                # step_start_time = time.time()
                try:
                    episode_return = 0
                    state = env.reset()
                    done = False
                    start = time.time()
                    while not done:
                        # accelerate the training process
                        # if time.time() - start > 20:
                        #     print("Time out")
                        #     break
                        # print(state.shape,"state")
                        action = agent.take_action(state)
                        # print(action.shape,"action")
                        next_state, reward, done, _ = env.step(action)
                        replay_buffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += reward
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            if env_name == "FlightEnvVec":
                                b_ns = b_ns.squeeze(1)  # reshape next_states
                                b_s = b_s.squeeze(1)  # reshape states
                            # print(b_ns.shape,"next_states shape")
                            # print(b_s.shape,"states shape")
                            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                               'dones': b_d}
                            agent.update(transition_dict)
                            if writer is not None:
                                if agent.__class__.__name__ == "SAC":
                                    actor_loss, critic_1_loss, critic_2_loss = agent.get_last_losses()
                                    writer.add_scalar("Loss/Actor", actor_loss,
                                                      i_episode + i * num_episodes / total_iteration)
                                    writer.add_scalar("Loss/Critic_1", critic_1_loss,
                                                      i_episode + i * num_episodes / total_iteration)
                                    writer.add_scalar("Loss/Critic_2", critic_2_loss,
                                                      i_episode + i * num_episodes / total_iteration)
                                    writer.add_scalar("Reward/Episode", reward,
                                                      i_episode + i * num_episodes / total_iteration)
                                elif agent.__class__.__name__ == "DDPG":
                                    action_loss, critic_loss = agent.get_last_losses()
                                    writer.add_scalar("Loss/Actor", action_loss,
                                                      i_episode + i * num_episodes / total_iteration)
                                    writer.add_scalar("Loss/Critic", critic_loss,
                                                      i_episode + i * num_episodes / total_iteration)
                                    writer.add_scalar("Reward/Episode", reward,
                                                      i_episode + i * num_episodes / total_iteration)
                                elif agent.__class__.__name__ == "TD3":
                                    action_loss, critic_1_loss, critic_2_loss = agent.get_last_losses()
                                    if action_loss is not None:
                                        writer.add_scalar("Loss/Actor", action_loss,
                                                          i_episode + i * num_episodes / total_iteration)
                                    if critic_1_loss is not None:
                                        writer.add_scalar("Loss/Critic_1", critic_1_loss,
                                                          i_episode + i * num_episodes / total_iteration)
                                    if critic_2_loss is not None:
                                        writer.add_scalar("Loss/Critic_2", critic_2_loss,
                                                          i_episode + i * num_episodes / total_iteration)
                                    writer.add_scalar("Reward/Episode", reward,
                                                      i_episode + i * num_episodes / total_iteration)
                    if writer is not None:
                        writer.add_scalar("Return", episode_return, i_episode)

                except KeyboardInterrupt:
                    # save actor and critic
                    num_episodes /= 10
                    if agent.__class__.__name__ == "SAC":
                        save_model(agent.actor,
                                   save_path + "/actor.pth" + str(i_episode + i * num_episodes / total_iteration))
                        # save_model(agent.critic_1, save_path + "/critic_1.pth"+ str(i*num_episodes+i_episode))
                        # save_model(agent.critic_2, save_path + "/critic_2.pth"+ str(i*num_episodes+i_episode))
                    elif agent.__class__.__name__ == "DDPG":
                        save_model(agent.actor, save_path + "/actor_DDPG.pth" + str(i * num_episodes + i_episode))
                        # save_model(agent.critic, save_path + "/critic_DDPG.pth"+ str(i*num_episodes+i_episode))
                    elif agent.__class__.__name__ == "TD3":
                        save_model(agent.actor, save_path + "/actor_TD3.pth" + str(i * num_episodes + i_episode))
                        # save_model(agent.critic_1, save_path + "/critic_1_TD3.pth"+ str(i*num_episodes+i_episode))
                        # save_model(agent.critic_2, save_path + "/critic_2_TD3.pth"+ str(i*num_episodes+i_episode))
                    print("Model saved" + agent.__class__.__name__ + "due to KeyboardInterruption.")
                    sys.exit(0)

                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
