import time
from tqdm import tqdm
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool

seed = 67
random.seed(seed)
np.random.seed(seed)
data = pd.read_csv("test/data.csv")

# ACO Parameters
init_p = 0.2
alpha = 1
beta = 2
added_p = 10
evap_rate = 0.1
ants = 100
courier_speed = 108 #km/h

# ACO Functions
def get_dist_matrix(cities: int) -> pd.DataFrame:
    city_names = data[data["Num Cities"]==float(cities)]["Start"].to_list()
    city_names = list(sorted(set(city_names)))
    dist_matrix = pd.DataFrame(index=city_names, columns=city_names)

    for city1 in city_names:
        for city2 in city_names:
            dist = 0
            if city1 != city2:
                dist = float(data[(data["Num Cities"]==float(cities)) & (data["Start"]==city1) & (data["End"]==city2)]["Distance"].item().replace(" km", ""))
            dist_matrix.loc[city1, city2] = dist

    return dist_matrix

def ant_colony_optimization(dist_matrix, eta, alpha, beta, p, q, num_ants, courier_speed) -> float:
    #create pheromone table
    pheromones = dist_matrix.copy()
    pheromones[pheromones!=0] = eta

    cities = dist_matrix.columns.to_list()
    best_dist = float('inf')

    start_time = time.time()

    #run iterations
    for i in range(num_ants):

        #refresh variables
        tread = []
        dist = 0

        choices = cities.copy()
        current = choices[0]
        choices.remove(current)

        #ant moves and gets route data
        for i in range(len(cities)-1):
            probabilities = [] #get probabilities
            denominator = sum([(pheromones.loc[current, choice2]**alpha) / (dist_matrix.loc[current, choice2]**beta) for choice2 in choices])
            for choice in choices:
                numerator = (pheromones.loc[current, choice]**alpha) / (dist_matrix.loc[current, choice]**beta)
                probabilities.append(numerator/denominator)
            
            next = np.random.choice(a=choices, p=probabilities) #list next destination and move on
            tread.append([current,next])
            dist += dist_matrix.loc[current, next]
            current = next
            choices.remove(current)
        dist += dist_matrix.loc[current, cities[0]]

        #update pheromones
        pheromones[pheromones!=0] *= (1-p) #evaporate
        for route in tread:
            pheromones.loc[route[0], route[1]] += (q/dist)
            pheromones.loc[route[1], route[0]] += (q/dist)
        
        if dist < best_dist: best_dist = dist
    
    end_time = time.time()

    hours = best_dist / courier_speed
    
    return hours

# GA Parameters
num_couriers = 2
chromosome_length = 15
population_size = 6
weighting_factor = 1
tournament_size = 2
crossover_rate = 1.0
mutation_rate = 0.2
generations = 100

df = get_dist_matrix(chromosome_length+1)

def get_time(x, val):
    y = np.where(x == val)[0]+1
    if len(y) > 1:
        y = [0] + y.tolist()
        t = ant_colony_optimization(df.iloc[y, y], init_p, alpha, beta, evap_rate, added_p, ants, courier_speed)
    else:
        t = 1e6 #punishment cuz it cant do aco
    return t

def get_costs(chromosome):
    list_of_t = [get_time(chromosome, i+1) for i in range(num_couriers)]
    return sum(list_of_t) + (weighting_factor*np.std(list_of_t))

def eval(chromosomes):
    #decoding chromosomes
    costs = []
    
    with Pool() as p:
        costs = p.map(get_costs, chromosomes)

    costs = np.array(costs)
    #evaluation
    fitness_list = 1/(costs + 1)
    return fitness_list


def genetic_algorithm(chromosomes) -> np.ndarray:
    fitness_list = eval(chromosomes)

    #tournament
    tournament_winners = []
    for i in range(population_size):
        indices = np.random.choice(len(fitness_list), tournament_size, replace=False)
        choices = fitness_list[indices]
        index = indices[np.argmax(choices).item()]

        tournament_winners.append(chromosomes[index].tolist())
    
    #crossover
    new_chromosomes = []
    for i in range(0, len(tournament_winners), 2):
        if i+1 < len(tournament_winners): 
            pair = [tournament_winners[i], tournament_winners[i+1]]
            c1, c2 = [], []
            for z in range(chromosome_length):
                num = random.randint(0,1)
                c1.append(pair[num][z])
                c2.append(pair[(num+1)%2][z])
            new_chromosomes.append(c1)
            new_chromosomes.append(c2)
        else: 
            new_chromosomes.append(tournament_winners[i])
    
    #mutation
    num_of_mutations = math.floor(mutation_rate*population_size)
    mutates = np.random.choice(population_size, num_of_mutations, replace=False).tolist()
    for i in range(num_of_mutations):
        changed = random.randint(0, chromosome_length-1)
        num = random.randint(1, num_couriers)
        while  new_chromosomes[mutates[i]][changed] == num:
            num = random.randint(1, num_couriers)
        new_chromosomes[mutates[i]][changed] = num

    return np.array(new_chromosomes)

#initial population
if __name__ == "__main__":
    print(df)
    print("Birmingham is the origin of both couriers")

    graph_fitness = []

    chromosomes = np.random.choice([1, num_couriers], size=(population_size, chromosome_length))
    for i in tqdm(range(generations)):
        chromosomes = genetic_algorithm(chromosomes)
        fitness_list = eval(chromosomes)
        graph_fitness.append(np.max(fitness_list).item())
    
    fitness_list = eval(chromosomes)
    print(f"Best of GA: {chromosomes[np.argmax(fitness_list).item()]}")

    # vals = np.arange(1, num_couriers + 1)
    # grids = np.indices((num_couriers,) * chromosome_length)
    # indexes = grids.reshape(chromosome_length, -1).T
    # all_possibilities = vals[indexes]
    # fitness_list = eval(all_possibilities)
    # print(f"Best of BF: {all_possibilities[np.argmax(fitness_list).item()]}")
    # print(f"Fitness: {np.max(fitness_list).item()}")

    # ranks = np.argsort(fitness_list)[::-1]
    # for i in range(len(ranks)):
    #     print(f"{i+1}. {all_possibilities[ranks[i]]} {fitness_list[ranks[i]]}")
    
    list_o_numbers = [1+i for i in range(generations)]

    plt.figure()
    plt.plot(list_o_numbers, graph_fitness)
    plt.xlabel('Generation')
    plt.ylabel('Fitnesss')
    plt.title("Fitness v Iteration")
    plt.legend()
    plt.grid(True)
    plt.show()

    pass
