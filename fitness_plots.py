import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import mod
import os

reward_amounts = [0,0.1,0.125,0.15,0.175,0.2]
LOG_DIR_PATHS = [r"C:\Users\hmunn\OneDrive\Desktop\ENGG1600\Project\Data\BipedalWalker\Run 1 - " + str(i) + " Q\Logs\\" for i in reward_amounts]
LOG_DIR_PATHS = [x + os.listdir(x)[0] for x in LOG_DIR_PATHS]

with open(LOG_DIR_PATHS[0], "r") as f:
    lines = f.readlines()

gens = len(lines)
fitnesses = np.zeros((len(LOG_DIR_PATHS), gens))
modularities = np.zeros((len(LOG_DIR_PATHS), gens))
generations = np.arange(gens)

for idx, run in enumerate(LOG_DIR_PATHS):
    with open(run, "r") as f:
        lines = f.readlines()
    for idx2, line in enumerate(lines):
        if idx2 > gens - 1: break
        try:
            fitness = float(line.split("actual fitness: ")[1].split(",")[0])
        except ValueError: # old format
            fitness = float(line.split("fitness: ")[1].split(" Q-Score")[0].split(",")[0])
        except IndexError:
            fitness = float(line.split("fitness: ")[1].split(" Q-Score")[0].split(",")[0])
        
        modularity = float(line.split("Q-Score: ")[1].split(",")[0])
        fitnesses[idx, idx2] = fitness
        modularities[idx, idx2] = modularity

def plot_confidences(fitnesses, modularities, env_name):
    #some confidence interval (mean +- std)
    ci1 = np.std(fitnesses, axis=0)
    ci2 = np.std(modularities, axis=0)

    fig, axis1 = plt.subplots()
    axis2 = plt.twinx()

    fitnesses, modularities = np.average(fitnesses, axis=0), np.average(modularities, axis=0)
    axis1.plot(generations, fitnesses, color='mediumblue')
    axis1.fill_between(generations, (fitnesses-ci1), (fitnesses+ci1), color='mediumblue', alpha=.15)
    axis1.set_ylim([-100,300])
    axis1.set_ylabel("Fitness", color='mediumblue')
    axis1.set_xlabel("Generation")
    axis2.plot(generations, modularities, color='orangered')
    axis2.fill_between(generations, (modularities-ci2), (modularities+ci2), color='orangered', alpha=.15)
    axis2.set_ylim([0,1])
    #axis2.set_xlim([0,19])
    axis2.set_ylabel("Modularity (Q-Score)", color='orangered')
    plt.title("Best NEAT Solutions for " + env_name)
    plt.show()

def plot_mod_scores(fitnesses, modularities, reward_amounts, env_name):
    colours = ["red", "darkorange", "gold", "forestgreen", "deepskyblue", "navy"]
    for i, f in enumerate(fitnesses):
        plt.plot(generations, f, color=colours[i], linestyle="-")
    plt.ylim([-100,300])
    plt.ylabel("Fitness")
    plt.xlabel("Generation")
    plt.title("Best NEAT Solutions for " + env_name + " with Varying Q-Importance")
    plt.legend(["Q-Importance: " + str(i) for i in reward_amounts])
    plt.show()

plot_mod_scores(fitnesses, modularities, reward_amounts, "BipedalWalker")
