import neat
import gym
import neat_graphs
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datetime
from multiprocessing import Process, Pool, cpu_count
import pickle
from functools import partial

def plot_grid(grid, GRID_SIZE):
    img = np.ones((GRID_SIZE,GRID_SIZE)) * -200
    #fitness_ranges = (min([x.fitness for x in grid.values()]), max([x.fitness for x in grid.values()]))
    min_fitness = 400
    for k in grid.keys():
        genome = grid[k]
        fitness = genome.fitness#(genome.fitness - fitness_ranges[0]) / (fitness_ranges[1] - fitness_ranges[0])
        if fitness < min_fitness:
            min_fitness = fitness
        img[k[1],k[0]] = fitness
    img[img == -200] = min_fitness - 5
    mapelites_colours = np.vstack((np.array([0,0,0,0]), matplotlib.cm.magma(np.arange(256))))
    mapelites_colours = matplotlib.colors.ListedColormap(mapelites_colours, name='mapelites', N = mapelites_colours.shape[0])
    # 0 should be no deviation:
    img = np.fliplr(img)
    plt.imshow(img, cmap=mapelites_colours)
    plt.gca().invert_yaxis()
    plt.xlabel("Deviation From Uniform Torque Output Distribution")
    plt.ylabel("Modularity")
    cb = plt.colorbar()
    cb.set_label('Fitness', rotation=270)
    plt.xticks(np.arange(21), np.round(np.linspace(0,1,21),2))
    plt.xticks(rotation=20)
    plt.yticks(np.arange(21), np.round(np.linspace(0,1,21),2))
    plt.margins(x=0)
    plt.margins(y=0)
    plt.title("MAP-ELITES Grid of BipedalWalker Networks")
    plt.show()

def evaluate_genome(genome, config, episodes, env, render, grid_size):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    total_reward = 0.0
    action_totals = [0,0,0,0]

    # for each episode
    for i in range(episodes):

        observation = env.reset()
        action = net.activate(observation)
        action_totals = [sum(x) for x in zip(action_totals, [abs(y) for y in action])]
        done = False
        t = 0

        while not done:
            if render: env.render()
            observation, reward, done, info = env.step(action)
            action = net.activate(observation)
            action_totals = [sum(x) for x in zip(action_totals, [abs(y) for y in action])]
            total_reward += reward
            t += 1
            if done:
                break
        
    fitness = total_reward / episodes
    
    # calculate leg domination (range rule) and q-score
    try:
        action_totals = 1 - ((max(action_totals) - min(action_totals)) / 4) / (sum(action_totals) / 4)
    except ZeroDivisionError:
        action_totals = 0
    graph = neat_graphs.NeatGraph(genome)
    grid_loc = (round(action_totals * (grid_size-1)), round(graph.q_score * (grid_size-1))) 
    return fitness, grid_loc, genome

def run_experiments(config, log_path, grid_size, genomes_per_gen, 
    gens, crossover_prob, mutate_prob, render, episodes, grid, checkpoint_path_prefix, checkpoint_save_freq, generation):
    gens = 100000
    env = gym.make("BipedalWalker-v3")
    
    # get initial population
    p = neat.Population(config)
    if generation == 0:
        for id, genome in enumerate(p.population.values()):
            fitness, grid_loc, _ = evaluate_genome(genome, config, episodes, env, render, grid_size)
            genome.fitness = fitness
            genome.id = id
            if grid_loc not in grid or grid[grid_loc].fitness < genome.fitness:
                grid[grid_loc] = genome
    #print([(key, grid[key].fitness) for key in grid.keys()])
    #plot_grid(grid, GRID_SIZE)

    # GA
    for gen in range(generation, gens):
        #print("Generation " + str(gen) + "\n")
        pop = list(grid.values())
        children = []
        for child in range(genomes_per_gen):
            if random.random() <= crossover_prob:
                p1, p2 = random.choice(pop), random.choice(pop)
            else:
                p1 = random.choice(pop)
                p2 = p1
            
            child_genome = config.genome_type(0)
            child_genome.configure_crossover(p1, p2, config.genome_config)
            child_genome.id = len(p.population) + gen*genomes_per_gen + child
            if random.random() <= mutate_prob:
                child_genome.mutate(config.genome_config)
            children.append((child_genome, config, episodes, env, render, grid_size))
        with Pool(processes=genomes_per_gen) as pool:
            process_results = pool.starmap(evaluate_genome, children)

        for (fitness, grid_loc, genome) in process_results:
            genome.fitness = fitness
            if grid_loc not in grid or grid[grid_loc].fitness < fitness:
                grid[grid_loc] = genome

        best_genome = grid[max(grid.keys(), key=lambda x: grid[x].fitness)]
        with open(log_path, "a") as f:
            f.write("Generation " + str(gen) + ": ID " + str(best_genome.id) + ", fitness = " + str(best_genome.fitness) + "\n")
        if gen % 500 == 0 and gen != 0:
            plot_grid(grid, grid_size)
        if gen % checkpoint_save_freq == 0 and gen != 0:
            cp = Checkpoint(config, log_path, grid_size, genomes_per_gen, gens, crossover_prob, mutate_prob,
                render, episodes, grid, checkpoint_path_prefix, checkpoint_save_freq, gen)
            with open(checkpoint_path_prefix+str(gen)+".pickle", 'wb') as handle:
                pickle.dump(cp, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Checkpoint object to contain all info
class Checkpoint():
    def __init__(self, config, log_path, grid_size, genomes_per_gen, gens,
        crossover_prob, mutate_prob, render, episodes, grid, checkpoint_path_prefix, checkpoint_save_freq, gen):
        self.config = config
        self.log_path = log_path
        self.grid_size = grid_size
        self.genomes_per_gen = genomes_per_gen
        self.gens = gens
        self.crossover_prob = crossover_prob
        self.mutate_prob = mutate_prob
        self.render = render
        self.episodes = episodes
        self.grid = grid
        self.checkpoint_path_prefix = checkpoint_path_prefix
        self.checkpoint_save_freq = checkpoint_save_freq
        self.gen = gen

    def get_partial(self):
        return partial(run_experiments, self.config, self.log_path, self.grid_size, self.genomes_per_gen,
            self.gens, self.crossover_prob, self.mutate_prob, self.render, self.episodes, self.grid, self.checkpoint_path_prefix, self.checkpoint_save_freq, self.gen)

if __name__ == "__main__":
    CHECKPOINT_LOAD_PATH = r"Checkpoints\checkpoint_2021_09_26_14_37_40_gen28400.pickle"

    if CHECKPOINT_LOAD_PATH != None:
        with open(CHECKPOINT_LOAD_PATH, 'rb') as handle:
            cp = pickle.load(handle)
            tmp = cp.get_partial()
            tmp()
                
    else:
        GRID_SIZE = 20
        grid = {} # (leg_dom, q-score) : genome
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation, "config.txt")
        genomes_per_gen = 12
        gens = 10000
        crossover_prob = 0.3
        mutate_prob = 0.75
        render = False
        episodes = 6
        timestamp = datetime.datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
        LOGS_PATH = "Logs/log_" + timestamp + ".txt"
        CHECKPOINT_PATH_PREFIX = "Checkpoints/checkpoint_" + timestamp + "_gen"
        checkpoint_save_freq = 20
        run_experiments(config, LOGS_PATH, GRID_SIZE, genomes_per_gen, gens, crossover_prob, 
            mutate_prob, render, episodes, grid, CHECKPOINT_PATH_PREFIX, checkpoint_save_freq, 0)

