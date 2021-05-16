from sys import maxsize
from itertools import permutations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from py2opt.routefinder import RouteFinder
import copy

class Population():
    def __init__(self, bag, adjacency_mat):
        self.bag = bag
        self.parents = []
        self.score = 0
        self.best = None
        self.adjacency_mat = adjacency_mat
        
    def select(self, k=4):
        fit = self.evaluate()
        while len(self.parents) < k:
            idx = np.random.randint(0, len(fit))
            if fit[idx] > np.random.rand():
                self.parents.append(self.bag[idx])
        self.parents = np.asarray(self.parents)
    
    def mutate(self, p_cross=0.1, p_mut=0.1):
        next_bag = []
        children = self.crossover(p_cross)
        for child in children:
            if np.random.rand() < p_mut:
                next_bag.append(swap(child))
            else:
                next_bag.append(child)
        return next_bag

    def evaluate(self):
        distances = np.asarray(
            [self.fitness(chromosome) for chromosome in self.bag]
        )
        self.score = np.min(distances)
        self.best = self.bag[distances.tolist().index(self.score)]
        self.parents.append(self.best)
        if False in (distances[0] == distances):
            distances = np.max(distances) - distances
        return distances / np.sum(distances)
    
    def fitness(self, chromosome):
        return sum(
            [
                self.adjacency_mat[chromosome[i], chromosome[i + 1]]
                for i in range(len(chromosome) - 1)
            ]
        )
    def crossover(self, p_cross=0.1):
        children = []
        count, size = self.parents.shape
        for _ in range(len(self.bag)):
            if np.random.rand() > p_cross:
                children.append(
                    list(self.parents[np.random.randint(count, size=1)[0]])
                )
            else:
                parent1, parent2 = self.parents[
                    np.random.randint(count, size=2), :
                ]
                idx = np.random.choice(range(size), size=2, replace=False)
                start, end = min(idx), max(idx)
                child = [None] * size
                for i in range(start, end + 1, 1):
                    child[i] = parent1[i]
                pointer = 0
                for i in range(size):
                    if child[i] is None:
                        while parent2[pointer] in child:
                            pointer += 1
                        child[i] = parent2[pointer]
                children.append(child)
        return children
    


def init_population(cities, adjacency_mat, n_population):
    return Population(
        np.asarray([np.random.permutation(cities) for _ in range(n_population)]), 
        adjacency_mat
    )

def swap(chromosome):
    a, b = np.random.choice(len(chromosome), 2)
    chromosome[a], chromosome[b] = (
        chromosome[b],
        chromosome[a],
    )
    return chromosome


 

def TSPdynamicPrograming(V,matrix):
    score = 0
    for x in range(1, V):
        g[x + 1, ()] = matrix[x][0]
    Vertex = [i for i in range(2,V+1)]

    get_minimum(1, (tuple(Vertex)))

    print('\n\nSolution to TSP: {1, ', end='')
    solution = p.pop()
    score += matrix[0][solution[1][0]-1]
    current_matrix = solution[1][0]
    #print(current_matrix)
    print(solution[1][0], end=', ')
    for x in range(V - 2):
        for new_solution in p:
            if tuple(solution[1]) == new_solution[0]:
                solution = new_solution
                score += matrix[current_matrix-1][solution[1][0]-1]
                #print(solution[1][0],'a')
                current_matrix = solution[1][0]
                print(solution[1][0], end=', ')
                break
    print('1}')
    print(matrix[current_matrix-1][0])
    score += matrix[current_matrix-1][0]
    print(score,'score')
    return score


def get_minimum(k, a):
    if (k, a) in g:
        # Already calculated Set g[%d, (%s)]=%d' % (k, str(a), g[k, a]))
        return g[k, a]

    values = []
    all_min = []
    for j in a:
        set_a = copy.deepcopy(list(a))
        set_a.remove(j)
        all_min.append([j, tuple(set_a)])
        result = get_minimum(j, tuple(set_a))
        values.append(graph[k-1][j-1] + result)

    # get minimun value from set as optimal solution for
    g[k, a] = min(values)
    p.append(((k, a), all_min[values.index(g[k, a])]))

    return g[k, a]

def factorial(num): 
    if num < 0: 
        print("Factorial of negative num does not exist")

    elif num == 0: 
        return 1
        
    else: 
        fact = 1
        while(num > 1): 
            fact *= num 
            num -= 1
        return fact 
 
def genetic_algorithm(
    cities,
    adjacency_mat,
    n_population=5,
    n_iter=20000,
    selectivity=0.15,
    p_cross=0.5,
    p_mut=0.1,
    print_interval=100,
    return_history=False,
    verbose=False,
):
    pop = init_population(cities, adjacency_mat, n_population)
    print(pop.bag,'pop.bag')
    best = pop.best
    score = float("inf")
    history = []
    for i in range(n_iter):
        pop.select(n_population * selectivity)
        history.append(pop.score)
        if verbose:
            print(f"Generation {i}: {pop.score}")
        elif i % print_interval == 0:
            print(f"Generation {i}: {pop.score}")
        if pop.score < score:
            best = pop.best
            score = pop.score
        children = pop.mutate(p_cross, p_mut)
        pop = Population(children, pop.adjacency_mat)
    print(best,'best')
    score += adjacency_mat[0, best[0]]
    score += adjacency_mat[best[V-2], 0]
    if return_history:
        return best, history
    return best, score


 
# Main Code
if __name__ == "__main__":
    dynamic_times=[[] for i in range(17) ]
    genetic_times=[[] for i in range(17) ]
    dynamic_average_times = []
    genetic_average_times = []
    dynamic_scores=[[] for i in range(17) ]
    genetic_scores=[[] for i in range(17) ]
    errors = [[] for i in range(17) ]
    average_errors = []
    Vertex = []
    for V in range(4,21):
        for i in range(5):
            p = []
            g = {}
            standard = 0
            vertex=[]
            for i in range(1,V):
                vertex.append(i)
            graph_count = factorial(V)/(factorial(V-2)*factorial(2))
            first_graph = []
            for i in range(int(graph_count)):
                first_graph.append(random.randrange(1,30))
            graph=[[0 for i in range(V)] for i in range(V)]
            for i in range(V):
                for j in range(V):
                    if i<j:
                        graph[i][j]=random.randrange(1,30)
            for i in range(V):
                for j in range(V):
                    if j<i:
                        graph[i][j]=graph[j][i]
            
            print(graph)
            start = time.time()
            dynamic_score = TSPdynamicPrograming(V, graph)
            print(dynamic_score)
            dynamic_scores[V-4].append(dynamic_score)
            end = time.time()
            dynamic_time = end-start
            dynamic_times[V-4].append(dynamic_time)
            print(dynamic_time)
            graph=np.asarray(graph)
            start = time.time()
            genetic_ans, genetic_score = genetic_algorithm(vertex, graph, V)
            genetic_scores[V-4].append(genetic_score)
            end = time.time()
            print(genetic_ans,'ans')
            print(genetic_score,'score')
            genetic_time = end-start
            genetic_times[V-4].append(genetic_time)
            print(genetic_time)
            error = genetic_score - dynamic_score
            errors[V-4].append(error)
        dynamic_average_times.append(sum(dynamic_times[V-4])/len(dynamic_times[V-4]))
        genetic_average_times.append(sum(genetic_times[V-4])/len(genetic_times[V-4]))
        average_errors.append(sum(errors[V-4])/len(errors[V-4]))
        Vertex.append(V)
    print(dynamic_average_times)
    print(genetic_average_times)
    print(average_errors)
    
    plt.figure()
    plt.style.use("ggplot")
    plt.plot(Vertex, dynamic_average_times,c = "r")
    plt.plot(Vertex, genetic_average_times,c = "b")
    plt.legend(labels=["Average times using Dynamic Programing", "Average times using Genetic Algorithm"], loc = 'best')
    plt.xlabel("Vertex", fontweight = "bold")
    plt.ylabel("Time", fontweight = "bold")
    plt.title("Time Cost", fontsize = 15, fontweight = "bold", y = 1.1)
    plt.show()
    
    plt.figure()
    plt.style.use("ggplot")
    plt.plot(Vertex, average_errors,c = "r")
    plt.legend(labels=["Average errors"], loc = 'best')
    plt.xlabel("Vertex", fontweight = "bold")
    plt.ylabel("Error", fontweight = "bold")
    plt.title("Average Error", fontsize = 15, fontweight = "bold", y = 1.1)
    
    plt.show()


    
        



        
    
    