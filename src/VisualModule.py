# %%
import pandas as pd
from matplotlib import pyplot as PLT
import cv2
import numpy as np
import socket
from PIL import Image
from stopwatch import Stopwatch

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
    def firstcoord(self, initial_state, action, image, skip_frames, frames_had):
        state_image = np.array(initial_state).reshape(768, 768, 3)

        info, reward, next_state = self.step(action)  # do action (new "timestep")

        #if frames_had%skip_frames == 0:
        #    print("will be analysed")
            # only get all extra information every 'skip_frames' times
        state_image_plus = np.array(next_state).reshape(768, 768, 3)
        stopwatch = Stopwatch()
        stopwatch.start()
        diff = np.subtract(state_image_plus, state_image)
        diff[:, :, 0] = np.array((pd.DataFrame(np.abs(diff[:, :, 0])) > 120) * diff[:, :, 0] * 10)  # Our agent
        diff[:, :, 2] = np.array((pd.DataFrame(np.abs(diff[:, :, 2])) > 120) * diff[:, :, 2] * 10)  # Enemy agent
        agent = diff[:, :, 0]  # Only use the chanel that reflect the agent colour
        enemy = diff[:, :, 2]
        indices = np.where(agent != [0])  # Just find where the values are not zero, and we have the position of our agent
        indices_e = np.where(enemy != [0])  # For each agent
        agent_coord = (int(np.median(indices[1])), int(np.median(indices[0])))
        enemy_coord = (int(np.median(indices_e[1])), int(np.median(indices_e[0])))
        stopwatch.stop()
        print(f"Total time to get coords: {stopwatch.duration}")
        if image == True:
            final = cv2.resize(state_image_plus, (768, 768))
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

        #else:
            # other values will not be used, and were not calculated
        #    return info, reward, np.array(0), (0, 0), (0, 0), next_state

    def coord(self, initial_state_image, action, image, to_skip, nr_steps):
        info, reward, state_image_plus, agent_coord, enemy_coord, following_state = self.firstcoord(initial_state_image, action,
                                                                                   image, to_skip, nr_steps)
        return info, reward, state_image_plus, agent_coord, enemy_coord, following_state

    def simpleCoord(self, ACTION, view_image, skip_steps, steps_taken):
        stopwatch = Stopwatch()
        if len(self.image_p) == 0:
            # start with empty lists and add first elements
            info, reward, state = self.reset()
            self.initial_state_image = np.array(state).reshape(self.size, self.size, 3)
            stopwatch.start()
            info, reward, state_image_plus, agent_coord, enemy_coord, nxt_state = self.coord(self.initial_state_image, ACTION, view_image, skip_steps, steps_taken)
            stopwatch.stop()
            print(f"Total time for coord init: {stopwatch.duration}")
            self.info_p.append(info)
            self.reward_p.append(reward)
            self.agent_path.append(list(agent_coord))
            self.enemy_path.append(list(enemy_coord))
            self.image_p.append(state_image_plus)
            # By using image_p I can make a loop from consecutive images
        else:
            state_image_plus = self.image_p[-1] # take last image
            stopwatch.start()
            info, reward, state_image_plus, agent_coord, enemy_coord, nxt_state = self.coord(state_image_plus, ACTION, view_image, skip_steps, steps_taken)
            stopwatch.stop()
            print(f"Total time for coord: {stopwatch.duration}")
            self.image_p[-1] = state_image_plus
            self.info_p.append(info)
            self.reward_p.append(reward)
            self.agent_path.append(list(agent_coord))
            self.enemy_path.append(list(enemy_coord))
        return info, reward, agent_coord, enemy_coord, nxt_state
