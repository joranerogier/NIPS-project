from VisualModule import AgentEnvironment
from DQN_Agent_extra import NeurosmashAgent

import numpy as np
import os
import random
from matplotlib import pyplot as plt
from stopwatch import Stopwatch
from check_dir import check_dir


class EpisodeLoop:
    def __init__(self):
        pass

    def __init__(self):
        # set directory where plos of rewards will be saved
        self.reward_plot_dir = "output/reward_plots/"
        check_dir(self.reward_plot_dir)
        self.reward_plot_path = f"{self.reward_plot_dir}test_plots_r10_l1_b128.png"

        # set variables about the states / action / features
        self.state_size = 5
        self.action_size = 3
        self.nr_action_executions = 3
        self.max_distance = 600

        # set variables for all episodes
        self.bin_size = 25 # for the reward plots
        self.skip_frames = 1 # faster and remember less similar states # was 5
        self.show_images = False
        self.episode_count = 300
        self.batch_size = 128

        # set values of rewards
        self.extra_win_reward = 8 # reward for winning is 10 + extra_win_reward
        self.negative_value = 1

        # initialize agent and environment
        self.agent = NeurosmashAgent(state_size=self.state_size,
                                     action_size=self.action_size, batch_size=self.batch_size)
        self.env = AgentEnvironment(size=768, timescale=1)

        # initialize variables used over all episode loops
        self.games_won = 0
        self.games_lost = 0

        # initialize rewards variables
        self.total_reward = 0
        self.total_rewards = []


        self.model_output_dir = "output/model_output/"
        check_dir(self.model_output_dir)

        # initialize variables used per episode-loop
        # now only for the very first episode
        self.won_now = False
        self.enemy_direction = [-1, 0]
        self.agent_direction = [1, 0]
        self.agent_trajectories = []
        self.enemy_trajectories = []
        self.relative_pos_enemy = [0, 0]
        self.distances = []
        self.done = 0
        self.agent_coord_x = 0
        self.agent_coord_y = 0
        self.enemy_coord_x = 0
        self.enemy_coord_y = 0
        self.agent_dir_x = 0
        self.agent_dir_y = 0
        self.rel_pos_enemy_x = 0
        self.rel_pos_enemy_y = 0
        self.enemy_dir_x = 0
        self.enemy_dir_y = 0
        self.init_features = np.reshape([[0]*self.state_size],
                                        [1, self.state_size])
        self.used_features = np.reshape([[self.agent_coord_x,
                                          self.agent_coord_y,
                                          self.enemy_coord_x,
                                          self.enemy_coord_y,
                                          #self.agent_dir_x,
                                          #self.agent_dir_y,
                                          #self.rel_pos_enemy_x,
                                          #self.rel_pos_enemy_y,
                                          #self.enemy_dir_x,
                                          #self.enemy_dir_y,
                                          self.done]],
                                          [1, self.state_size])



    # is not called anymore, we don't want to use location as reward,
    # since the enemy always follows the agent on its own.
    def compute_reward(self, standard_reward, distance):
        distance_reward = (self.max_distance - distance) / self.max_distance
        complete_reward = (distance_reward + standard_reward) / 20
        self.total_reward += complete_reward


    def direction(self, agent_path, enemy_path):
        A_X = (agent_path[-1] - np.array(agent_path[-2]))[0]
        A_Y = -(agent_path[-1] - np.array(agent_path[-2]))[1]
        E_X = (enemy_path[-1] - np.array(enemy_path[-2]))[0]
        E_Y = -(enemy_path[-1] - np.array(enemy_path[-2]))[1]
        return [A_X,A_Y], [E_X,E_Y]


    def do_action(self, action):
        info, reward, agent_coord, enemy_coord, following_state = self.env.actionLoop(action, 0, self.show_images)
        self.agent_trajectories.append(list(agent_coord))
        self.enemy_trajectories.append(list(enemy_coord))

        if reward == 10: # our agent won
            self.games_won += 1
            self.won_now = True
        elif (info == 1) & (reward == 0):
            self.games_lost += 1


        if len(self.env.agent_path) < 2:
            distance = 500 # Initial distance, only for initialisation
            self.agent_direction = [1, 0] # By definition of facing each other
            self.enemy_direction = [-1, 0]
        else:
            self.distance = np.sqrt(np.square(np.array(list(np.array(agent_coord)- np.array(enemy_coord))).sum(axis=0)))
            self.agent_direction, self.enemy_direction  = self.direction(self.env.agent_path, self.env.enemy_path)

        #self.compute_reward(reward, distance)
        self.total_reward += reward
        self.rel_pos_enemy = np.array(enemy_coord) - np.array(agent_coord)

        return info, following_state


    def init_environment(self, agent_here):
        info, reward, state = self.env.reset()
        action = agent_here.act(self.init_features) # get next action
        self.done = 0
        info, next_state = self.do_action(action)

        return info, next_state

    def get_small_state(self):
        self.agent_coord_x = self.agent_trajectories[-1][0]
        self.agent_coord_y = self.agent_trajectories[-1][1]
        self.enemy_coord_x = self.agent_trajectories[-1][0]
        self.enemy_coord_y = self.agent_trajectories[-1][1]
        self.agent_dir_x = self.agent_direction[0]
        self.agent_dir_y = self.agent_direction[1]
        self.rel_pos_enemy_x = self.relative_pos_enemy[0]
        self.rel_pos_enemy_y = self.relative_pos_enemy[1]
        self.enemy_dir_x = self.enemy_direction[0]
        self.enemy_dir_y = self.enemy_direction[1]
        self.done = self.done
        return self.used_features


    def main_loop(self):
        stopwatch_main = Stopwatch()
        stopwatch_main.start()
        self.games_lost = 0
        self.games_won = 0

        for e in range(self.episode_count):
            status, next_state = self.init_environment(self.agent)
            small_state = self.get_small_state()
            small_state = np.reshape(small_state, [1, self.state_size])

            total_timesteps = 0
            action = 0
            execute_action = 0
            self.total_reward = 0
            self.won_now = False


            while self.done == 0:
                if (total_timesteps % self.skip_frames == 0) or (total_timesteps % self.skip_frames == self.skip_frames - 1):
                    evaluate_frame = True
                else:
                    evaluate_frame = False

                if execute_action == 0:
                    execute_action = self.nr_action_executions #random.randrange(1, 10) # execute the action a random amount of times
                    action = self.agent.act(small_state) # instantiate new action


                status, next_state = self.do_action(action)
                execute_action -= 1

                if status == 1:
                    self.done = 1
                    if self.won_now:
                         # maybe only if you won
                        string_game_result = "you won!"
                        if total_timesteps <= 450:
                            self.total_reward += self.extra_win_reward
                        elif total_timesteps >= 1000:
                            self.total_reward -= 2
                        elif total_timesteps >= 1500:
                            self.total_reward -= 7
                    else:
                        self.total_reward -= self.negative_value
                        #self.total_reward += (50/total_timesteps)
                        string_game_result = "you lost"
                        #if total_timesteps <= 400:
                        #    self.total_reward += 0.1


                    print(f"Game nr. {e} is finished, \n {string_game_result} - your final reward is: {self.total_reward}, duration was {total_timesteps} timesteps")


                done_list = [self.done]
                next_small_state = self.get_small_state()
                next_small_state = np.reshape(next_small_state, [1, self.state_size]) # why?

                small_state = np.reshape(small_state, [1, self.state_size])

                if (total_timesteps % self.skip_frames == 0):
                    self.agent.remember(small_state, action, self.total_reward, next_small_state, list(done_list))

                small_state = next_small_state # new small state
                total_timesteps += 1

            self.total_rewards.append(self.total_reward)

            if len(self.agent.memory) > self.batch_size:
                self.agent.train()

            if e % 50 == 0:
                self.agent.save(self.model_output_dir + "weights_"+ '{:04d}'.format(e) + ".hdf5")

            # plot and save rewards until now
            if (e % self.bin_size == 0) & (e != 0):
                x_data_per_episode = list(range(e + 1))
                y_data_per_episode = self.total_rewards
                print(x_data_per_episode)
                print(y_data_per_episode)
                y_sums = []
                y_averages = []
                x_sums_averages = list(range(int(e/self.bin_size)))

                bins = int(e/self.bin_size)

                for i in range(bins):
                    part = y_data_per_episode[(i*self.bin_size): ((i*self.bin_size)+self.bin_size)]
                    sum_part = sum(part)
                    y_sums.append(sum_part)
                    average_part = sum_part / len(part)
                    y_averages.append(average_part)

                print(x_sums_averages)
                print(y_sums)
                print(y_averages)

                fig = plt.figure()
                ax1 = fig.add_subplot(1, 3, 1)
                ax2 = fig.add_subplot(1, 3, 2)
                ax3 = fig.add_subplot(1, 3, 3)

                ax1.scatter(x_data_per_episode, y_data_per_episode, label='data')
                ax1.set_xlabel("episode nr.")
                ax1.set_ylabel("total reward")
                ax1.set_title('total reward per episode')

                ax2.bar(x_sums_averages, y_sums, label='data')
                ax2.set_xlabel("batch nr.")
                ax2.set_ylabel("sum reward")
                ax2.set_title('total reward per batch of size 4')

                ax3.bar(x_sums_averages, y_averages, label='data')
                ax3.set_xlabel("batch nr.")
                ax3.set_ylabel("average reward")
                ax3.set_title('total reward per batch of size 4')

                fig.tight_layout()

                fig.savefig(self.reward_plot_path)



        stopwatch_main.stop()
        print(f"finished all episodes (in {int(stopwatch_main.duration)}). \nTotal games won: {self.games_won} \nTotal games lost: {self.games_lost}")


if __name__ == '__main__':
    agent_learn = EpisodeLoop()
    agent_learn.main_loop()
