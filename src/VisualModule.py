
# %%
import pandas as pd
from matplotlib import pyplot as PLT
import cv2
import numpy as np
import socket
from PIL import Image
# %%
class Environment:
    def __init__(self, ip="127.0.0.1", port=13000, size=768, timescale=1):
        self.initial_state_image = None
        self.image_p = []
        self.into_p = []
        self.reward_p = []
        self.Enemy_path = []
        self.Agent_path = []

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = ip
        self.port = port
        self.size = size
        self.timescale = timescale

        self.client.connect((ip, port))

    def reset(self):
        self._send(1, 0)
        self.image_p = []
        self.into_p = []
        self.reward_p = []
        self.Enemy_path = []
        self.Agent_path = []

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
    def firstcoord(self, initial_state, action, image=False):
        state_image = np.array(initial_state).reshape(768, 768, 3)

        info, reward, state = self.step(action)  # Action to get angle

        state_image_plus = np.array(state).reshape(768, 768, 3)
        diff = np.subtract(state_image_plus, state_image)
        diff[:, :, 0] = np.array((pd.DataFrame(np.abs(diff[:, :, 0])) > 120) * diff[:, :, 0] * 10)  # Our agent

        diff[:, :, 2] = np.array((pd.DataFrame(np.abs(diff[:, :, 2])) > 120) * diff[:, :, 2] * 10)  # Enemy agent
        Agent = diff[:, :, 0]  # Only use the chanel that reflect the agent colour
        Enemy = diff[:, :, 2]
        indices = np.where(
            Agent != [0])  # Just find where the values are not zero, and we have the position of our agent
        indices_e = np.where(Enemy != [0])  # For each agent
        Agent_coord = int(np.median(indices[1])), int(np.median(indices[0]))
        Enemy_coord = int(np.median(indices_e[1])), int(np.median(indices_e[0]))
        if image == True:
            final = cv2.resize(state_image_plus, (768, 768))
            # Plot to see where the agents are in the coordenades
            cv2.rectangle(final,
                          tuple(np.array(list(Agent_coord)) + [35, 60]),
                          tuple(np.array(list(Agent_coord)) - [35, 60]),
                          (255, 0, 0), 2)
            cv2.rectangle(final,
                          tuple(np.array(list(Enemy_coord)) + [35, 60]),
                          tuple(np.array(list(Enemy_coord)) - [35, 60]),
                          (0, 0, 255), 2)
            # Displaying the image

            PLT.imshow(final)
            PLT.rcParams["figure.figsize"] = (5, 5)
            PLT.show()
        return info, reward, state_image_plus, Agent_coord, Enemy_coord

    def coord(self, initial_state_image, action, image=False):
        info, reward, state_image_plus, Agent_coord, Enemy_coord = self.firstcoord(initial_state_image, action,
                                                                                   image=image)
        return info, reward, state_image_plus, Agent_coord, Enemy_coord

    def simpleCoord(self, ACTION, vew_image=0):
        if len(self.image_p) == 0:
            info, reward, state = self.reset()
            self.initial_state_image = np.array(state).reshape(768, 768, 3)
            info, reward, state_image_plus, Agent_coord, Enemy_coord = self.coord(self.initial_state_image, ACTION,
                                                                                  vew_image)
            self.into_p.append(info), self.reward_p.append(reward), \
            self.Agent_path.append(list(Agent_coord)), self.Enemy_path.append(
                list(Enemy_coord))
            self.image_p.append(state_image_plus)
            # By using image_p I can make a loop form consecutive images
        else:
            state_image_plus = self.image_p[-1]
            info, reward, state_image_plus, Agent_coord, Enemy_coord = self.coord(state_image_plus, ACTION, vew_image)
            self.image_p[-1] = state_image_plus
            self.into_p.append(info), self.reward_p.append(reward), self.Agent_path.append(
                list(Agent_coord)), self.Enemy_path.append(
                list(Enemy_coord))
        return info, reward, Agent_coord, Enemy_coord
