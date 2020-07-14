"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing


def get_args():

    # The various arguments thart can be passed to train the model 
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=2000000)
    parser.add_argument("--replay_memory_size", type=int, default=50000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    model = DeepQNetwork()
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    replay_memory = []
    iter = 0
    #######
    state_values = np.zeros(opt.num_iters) # initial guess = 0 value
    eligibility = np.zeros(opt.num_iters)
    lamb = 0.95 # the lambda weighting factor
    # state = env.reset() # start the environment, get the initial state
    #######
    # Run the algorithm for some episodes
    while iter < opt.num_iters:
        prediction = model(state)[0]
        ########
        
        ###############
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (
                (opt.num_iters - iter) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_iters)
        u = random()
        random_action = u <= epsilon
        if random_action:
            print("Perform a random action")
            action = randint(0, 1)
        else:

            action = torch.argmax(prediction)
        
        # act according to policy
        # action = policy(state)
        next_image, reward, terminal = game_state.next_frame(action)
        # Update eligibilities
        eligibility *= lamb * opt.gamma
        eligibility[iter] += 1.0

        # next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        ####### 
        if iter > 0:
            td_error = reward + opt.gamma * state_values[iter] - state_values[iter - 1]
            state_values = state_values + 1e-6 * td_error * eligibility
            print('td_error is' + str(td_error))

        # if done:
        #     state = env.reset()
        # else:
        #     state = new_state
        ############
        replay_memory.append([state, action, reward, next_state, terminal])

        print('------')
        print('state is ' + str(state))
        print('action is ' + str(action))
        print('reward is ' + str(reward))
        print('next_state is ' + str(next_state))
        print('terminal is ' + str(terminal))
        print('------')

        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        y_batch = torch.cat(
            tuple(reward if terminal else reward + opt.gamma * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()
        # y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        state = next_state
        iter += 1
        print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
            iter + 1,
            opt.num_iters,
            action,
            loss,
            epsilon, reward, torch.max(prediction)))
        writer.add_scalar('Train/Loss', loss, iter)
        writer.add_scalar('Train/Epsilon', epsilon, iter)
        writer.add_scalar('Train/Reward', reward, iter)
        writer.add_scalar('Train/Q-value', torch.max(prediction), iter)
        if (iter+1) % 1000000 == 0:
            torch.save(model, "{}/flappy_bird_{}".format(opt.saved_path, iter+1))
    torch.save(model, "{}/flappy_bird".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
