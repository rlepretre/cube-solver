"""
Extract of utilities module containing helper functions for the Deep Q-Learning - Lunar Lander
Jupyter notebook (C3_W3_A1_Assignment) from DeepLearning.AI's "Unsupervised Learning,
Recommenders, Reinforcement Learning" course on Coursera.
"""

import numpy as np
import torch
import random
import imageio

def extract_experiences(memory_buffer, batch_size):
    experiences = random.sample(memory_buffer, k=batch_size)
    states = torch.tensor(
        np.array([e.state for e in experiences if e is not None]), dtype=torch.float
    )
    actions = torch.tensor(
        np.array([e.action for e in experiences if e is not None]), dtype=torch.float
    )
    rewards = torch.tensor(
        np.array([e.reward for e in experiences if e is not None]), dtype=torch.float
    )
    next_states = torch.tensor(
        np.array([e.next_state for e in experiences if e is not None]), dtype=torch.float
    )
    done_vals = torch.tensor(
        np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
        dtype=torch.float,
    )
    return (states, actions, rewards, next_states, done_vals)


def create_video(filename, env, q_network, fps=30):
    """
    Creates a video of an agent interacting with a Gym environment.

    The agent will interact with the given env environment using the q_network to map
    states to Q values and using a greedy policy to choose its actions (i.e it will
    choose the actions that yield the maximum Q values).
    
    The video will be saved to a file with the given filename. The video format must be
    specified in the filename by providing a file extension (.mp4, .gif, etc..). If you 
    want to embed the video in a Jupyter notebook using the embed_mp4 function, then the
    video must be saved as an MP4 file. 
    
    Args:
        filename (string):
            The path to the file to which the video will be saved. The video format will
            be selected based on the filename. Therefore, the video format must be
            specified in the filename by providing a file extension (i.e.
            "./videos/lunar_lander.mp4"). To see a list of supported formats see the
            imageio documentation: https://imageio.readthedocs.io/en/v2.8.0/formats.html
        env (Gym Environment): 
            The Gym environment the agent will interact with.
        q_network (tf.keras.Sequential):
            A TensorFlow Keras Sequential model that maps states to Q values.
        fps (int):
            The number of frames per second. Specifies the frame rate of the output
            video. The default frame rate is 30 frames per second.  
    """

    with imageio.get_writer(filename, fps=fps) as video:
        done = False
        state = env.reset()
        frame = env.render()
        video.append_data(frame)
        while not done:
            state = np.expand_dims(state, axis=0)
            q_values = q_network(state)
            action = np.argmax(q_values.numpy()[0])
            state, _, done, _, _ = env.step(action)
            frame = env.render()
            video.append_data(frame)