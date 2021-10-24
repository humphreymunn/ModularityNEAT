# todo 1: q-score in fitness function
# todo 2: plot performance / confidence bars etc
# todo 3: plot networks

from warnings import catch_warnings
import neat
import run_neat_base
import gym
import time
import neat_graphs
import numpy as np

env_name = "BipedalWalker-v3" # BipedalWalker-v3, Acrobot-v1, LunarLanderContinuous-v2
env = gym.make(env_name)
q_importance = 0.175 # None if using original

# Get output from network (bipedal walker)
def eval_network_bipedal(net, net_input):
    return net.activate(net_input)

# Get output from network (bipedal walker)
def eval_network_acrobot(net, net_input):
    return np.argmax(net.activate(net_input))

def eval_network_lander(net, net_input):
    return np.array(net.activate(net_input))

def eval_single_genome_bipedal(genome, genome_config):

    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)
    total_reward = 0.0
    #action_totals = [0,0,0,0]

    # for each episode
    for i in range(run_neat_base.n):

        observation = env.reset()
        action = eval_network_bipedal(net, observation)
        #action_totals = [sum(x) for x in zip(action_totals, [abs(y) for y in action])]
        done = False
        t = 0

        while not done:
            #env.render()
            observation, reward, done, info = env.step(action)
            action = eval_network_bipedal(net, observation)
            #action_totals = [sum(x) for x in zip(action_totals, [abs(y) for y in action])]
            total_reward += reward
            t += 1
            if done:
                break
        
    fitness = total_reward / run_neat_base.n
    '''try:
        action_totals = 1 - abs((action_totals[0] + action_totals[1]) - (action_totals[2] + action_totals[3])) / \
        (action_totals[0] + action_totals[1] + action_totals[2] + action_totals[3])# [0,1], 0 if if single leg dominated, 1 for equal torque for both legs
    except ZeroDivisionError:
        action_totals = 0
    '''
    if q_importance != None:
        capped_fitness = min(max(fitness, -100), 300) # cap within [-100, 300]
        graph = neat_graphs.NeatGraph(genome)
        fitness += graph.q_score*q_importance*(capped_fitness + 100)

    return fitness

def eval_single_genome_lander(genome, genome_config):

    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)
    total_reward = 0.0

    # for each episode
    for i in range(run_neat_base.n):

        observation = env.reset()
        action = eval_network_lander(net, observation)
        done = False
        t = 0

        while not done:
            #env.render()
            observation, reward, done, info = env.step(action)
            action = eval_network_lander(net, observation)
            total_reward += reward
            t += 1
            if done:
                break
        
    fitness = total_reward / run_neat_base.n
    return fitness

def eval_single_genome_acrobot(genome, genome_config):

    net = neat.nn.RecurrentNetwork.create(genome, genome_config)
    total_reward = 0.0

    # for each episode
    for i in range(run_neat_base.n):

        observation = env.reset()
        action = eval_network_acrobot(net, observation)
        done = False
        t = 0

        while not done:
            #env.render()
            observation, reward, done, info = env.step(action)
            action = eval_network_acrobot(net, observation)
            total_reward += reward
            t += 1
            if done:
                break

    return total_reward / run_neat_base.n


def main():

    global env
    params = [None]*3
    if env_name == "BipedalWalker-v3":
        params = [eval_network_bipedal, eval_single_genome_bipedal, "config-bipedal.txt"]
    elif env_name == "Acrobot-v1":
        params = [eval_network_acrobot, eval_single_genome_acrobot, "config-acrobot.txt"]
    elif env_name == "LunarLanderContinuous-v2":
        params = [eval_network_lander, eval_single_genome_lander, "config-lander.txt"]

    run_neat_base.run(params[0],
                      params[1],
                      environment_name=env_name, environment=env, config_file=params[2])


if __name__ == '__main__':
    main()
