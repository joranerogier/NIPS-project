# %%
import pandas as pd
from matplotlib import pyplot as PLT
import cv2
import numpy as np
import socket
from PIL import Image
from stopwatch import Stopwatch
import random

# %%
class AgentEnvironment:
    def __init__(self, ip="127.0.0.1", port=13000, size=768, timescale=1):
        self.initial_state_image = None
        self.image_p = []
        self.info_p = []
        self.reward_p = []
        self.enemy_path = []
        self.agent_path = []

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = ip
        self.port = port
        self.size = size
        self.timescale = timescale

        self.client.connect((ip, port))


    def reset(self):
        self._send(1, 0)
        self.image_p = []
        self.info_p = []
        self.reward_p = []
        self.enemy_path = []
        self.agent_path = []
        return self._receive()

    def step(self, action):
        self._send(2, action)
        return self._receive()

    def state2image(self, state):
        return Image.fromarray(np.array(state, "uint8").reshape(self.size, self.size, 3))

    def _receive(self):
        # Kudos to Jan for the socket.MSG_WAITALL fix!
        data = self.client.recv(2 + 3 * self.size ** 2, socket.MSG_WAITALL)
        end = data[0]
        reward = data[1]
        state = [data[i] for i in range(2, len(data))]
        return end, reward, state

    def _send(self, action, command):
        self.client.send(bytes([action, command]))

    # Background functions
    def firstcoord(self, initial_state, action, image):
        state_image = np.array(initial_state).reshape(768, 768, 3)

        info, reward, next_state = self.step(action)  # do action (new "timestep")

        # only get all extra information every 'skip_frames' times
        state_image_plus = np.array(next_state).reshape(768, 768, 3)
        stopwatch_coord = Stopwatch()
        stopwatch_coord.start()
        '''
        # too slow!
        diff = np.subtract(state_image_plus, state_image)
        diff[:, :, 0] = np.array((pd.DataFrame(np.abs(diff[:, :, 0])) > 120) * diff[:, :, 0] * 10)  # Our agent
        diff[:, :, 2] = np.array((pd.DataFrame(np.abs(diff[:, :, 2])) > 120) * diff[:, :, 2] * 10)  # Enemy agent
        agent = diff[:, :, 0]  # Only use the chanel that reflect the agent colour
        enemy = diff[:, :, 2]
        indices = np.where(agent != [0])  # Just find where the values are not zero, and we have the position of our agent
        indices_e = np.where(enemy != [0])  # For each agent
        agent_coord = (int(np.median(indices[1])), int(np.median(indices[0])))
        enemy_coord = (int(np.median(indices_e[1])), int(np.median(indices_e[0])))
        stopwatch_coord.stop()
        print(f"Total time to get coords: {stopwatch_coord.duration}")
        '''
        ## find coordinates with help of cv2
        colour_boundaries = [([45, 35, 120], [80, 80, 175]), # blue (actually red, but colours are flipped when image is saved)
                            ([120, 60, 50], [170, 130, 90])] # red (actually blue, but colours are flipped when image is saved)
        cv2.imwrite('temp.jpg', state_image_plus)
        img = cv2.imread('temp.jpg')
        i = 0
        for (lower, upper) in colour_boundaries:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            mask = cv2.inRange(img, lower, upper)
            if i == 1:
                (agent_x, agent_y, _, _) = cv2.boundingRect(mask)
                agent_coord = (int(agent_x+30), int(agent_y+30))
                #print(f"agent coordinates: {agent_x, agent_y, agent_w, agent_h}")
                # print mask
                #output = cv2.bitwise_and(img, img, mask = mask)
                #cv2.imwrite(f"red_img_{random.randrange(50)}.jpg", output)

            else:
                (enemy_x, enemy_y, _, _) = cv2.boundingRect(mask)
                #print(enemy_x, enemy_y, enemy_w, enemy_h)
                enemy_coord = (int(enemy_x+30), int(enemy_y+30))
                #print(f"enemy coordinates: {enemy_coord}")
                # print mask
                #output = cv2.bitwise_and(img, img, mask = mask)
                #cv2.imwrite(f"blue_img_{random.randrange(50)}.jpg", output)


            i += 1

        stopwatch_coord.stop()
        #print(f"Total time to get coords: {stopwatch_coord.duration}")

        # Save image with circles to check correct center of agent
        red_circle_img = cv2.circle(img, agent_coord, 20, (0, 0, 255), 2)
        blue_red_circle_img = cv2.circle(red_circle_img, enemy_coord, 20, (255, 0, 0), 2)
        cv2.imwrite(f"circles_img_{random.randrange(50)}.jpg", blue_red_circle_img)


        if image == True:
            final = cv2.resize(state_image_plus, (self.size, self.size))
            # Plot to see where the agents are in the coordenades
            cv2.rectangle(final,
                          tuple(np.array(list(agent_coord)) + [35, 60]),
                          tuple(np.array(list(agent_coord)) - [35, 60]),
                          (255, 0, 0), 2)
            cv2.rectangle(final,
                          tuple(np.array(list(enemy_coord)) + [35, 60]),
                          tuple(np.array(list(enemy_coord)) - [35, 60]),
                          (0, 0, 255), 2)
            # Displaying the image

            PLT.imshow(final)
            PLT.rcParams["figure.figsize"] = (5, 5)
            PLT.show()
        return info, reward, state_image_plus, agent_coord, enemy_coord, next_state


    def coord(self, initial_state_image, action, image):
        info, reward, state_image_plus, agent_coord, enemy_coord, following_state = self.firstcoord(initial_state_image, action, image)
        return info, reward, state_image_plus, agent_coord, enemy_coord, following_state


    def simpleCoord(self, ACTION, view_image, analyse_frame):
        stopwatch = Stopwatch()
        if len(self.image_p) == 0:
            # start with empty lists and add first elements
            info, reward, state = self.reset()
            self.initial_state_image = np.array(state).reshape(self.size, self.size, 3)

            stopwatch.start()
            info, reward, state_image_plus, agent_coord, enemy_coord, nxt_state = self.coord(self.initial_state_image, ACTION, view_image)
            stopwatch.stop()
            #print(f"Total time for coord init: {stopwatch.duration}")

            self.info_p.append(info)
            self.reward_p.append(reward)
            self.agent_path.append(list(agent_coord))
            self.enemy_path.append(list(enemy_coord))
            self.image_p.append(state_image_plus)
            # By using image_p I can make a loop from consecutive images
        else:
            stopwatch.start()
            if analyse_frame:
                state_image_plus = self.image_p[-1] # take last image
                stopwatch.start()
                info, reward, state_image_plus, agent_coord, enemy_coord, nxt_state = self.coord(state_image_plus, ACTION, view_image)
                stopwatch.stop()
                #print(f"Total time for coord: {stopwatch.duration}")
                self.image_p[-1] = state_image_plus
                self.info_p.append(info)
                self.reward_p.append(reward)
                self.agent_path.append(list(agent_coord))
                self.enemy_path.append(list(enemy_coord))
                stopwatch.stop()
                print(f"total time for current frame (with analysis): {stopwatch.duration}")
            else:
                info, reward, nxt_state = self.step(ACTION)
                stopwatch.stop()
                print(f"total time for current frame (no analysis): {stopwatch.duration}")
                return info, reward, (0,0), (0,0), nxt_state # agent and enemy coordinates were not calculated, so return default random value
        return info, reward, agent_coord, enemy_coord, nxt_state
