from VisualModule import AgentEnvironment
from DQN_Agent import NeurosmashAgent

import numpy as np
import os
import random

#from stopwatch import Stopwatch

class EpisodeLoop:
    def __init__(self):
        pass

    def __init__(self):
        self.state_size = 5
        self.max_distance = 600
        self.show_images = False
        self.skip_frames = 15 # faster and remember less similar states
        self.action_size = 3
        self.episode_count = 1000
        self.batch_size = 32
        self.agent = NeurosmashAgent(state_size=self.state_size,
                                     action_size=self.action_size)
        self.env = AgentEnvironment(size=768, timescale=10)

        self.games_won = 0
        self.games_lost = 0
        self.won_now = False
        self.agent_trajectories = []
        self.enemy_trajectories = []
        self.enemy_direction = [-1, 0]
        self.agent_direction = [1, 0]
        self.total_reward = 0
        self.total_rewards = []
        self.relative_pos_enemy = [0, 0]
        self.distances = []
        self.done = 0
        self.negative_value = 10

        self.agent_coord_x = 0
        self.agent_coord_y = 0
        self.enemy_coord_x = 0
        self.enemy_coord_y = 0
        self.agent_dir_x  = 0
        self.agent_dir_y  = 0
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

        self.model_output_dir = "output/model_output/"
        if not os.path.exists(self.model_output_dir):
            os.makedirs(self.model_output_dir)

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
        self.agent_dir_x  = self.agent_direction[0]
        self.agent_dir_y  = self.agent_direction[1]
        self.rel_pos_enemy_x = self.relative_pos_enemy[0]
        self.rel_pos_enemy_y = self.relative_pos_enemy[1]
        self.enemy_dir_x = self.enemy_direction[0]
        self.enemy_dir_y = self.enemy_direction[1]
        self.done = self.done
        return self.used_features


    def main_loop(self):
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
                    execute_action = 3#random.randrange(1, 10) # execute the action a random amount of times
                    action = self.agent.act(small_state) # instantiate new action


                status, next_state = self.do_action(action)
                execute_action -= 1

                if status == 1:
                    self.done = 1

                    print(self.won_now)

                    if self.won_now == False:
                        print("comes here")
                        self.total_reward -= 10
                        self.total_reward += (100/total_timesteps)
                    else:
                        self.total_reward += (100/total_timesteps) # maybe only if you won

                    print(f"Game nr. {e} is finished, \n you won: {self.won_now} - your final reward is: {self.total_reward}, duration was {total_timesteps} timesteps")


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
                self.agent.train(self.batch_size)


            if e % 50 == 0:
                self.agent.save(self.model_output_dir + "weights_"+ '{:04d}'.format(e) + ".hdf5")


        print(f"finished all episodes. \nTotal games won: {self.games_won} \nTotal games lost: {self.games_lost}")


if __name__ == '__main__':
    agent_learn = EpisodeLoop()
    agent_learn.main_loop()