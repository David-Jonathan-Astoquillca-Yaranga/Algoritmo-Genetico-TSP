import random
import pandas as pd
from deap import base, creator, tools, algorithms
import time

# Leer el archivo Excel con las distancias entre nodos
def leer_distancias(archivo):
    df = pd.read_excel(archivo, index_col=0)
    return df.values

# Función de aptitud: calcular la distancia total recorrida en la secuencia de nodos
def calcular_distancia(individual, distancias):
    distancia_total = 0
    for i in range(len(individual) - 1):
        distancia_total += distancias[individual[i]][individual[i+1]]
    distancia_total += distancias[individual[-1]][individual[0]]  
    return distancia_total,

# Configuración de DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# Parámetros y ejecución del algoritmo genético
def main():
    start_time = time.time()  
    random.seed(1)  
    distancias = leer_distancias("C:\\Users\\Jonathan\\Desktop\\Genetico.xlsx") // Ubicación del archivo
    num_nodos = len(distancias)
    
    def create_individual():
        ind = [0]+random.sample(range(1,num_nodos-1),num_nodos-2)+[num_nodos-1]  
        return ind
    toolbox.register("indices",random.sample,range(1,num_nodos-1),num_nodos-2)
    toolbox.register("individual",tools.initIterate,creator.Individual,create_individual)
    toolbox.register("population",tools.initRepeat,list,toolbox.individual)
    toolbox.register("mate", tools.cxPartialyMatched)
    
    def custom_mutate(individual):
        mutGen = 0.14
        size = len(individual)
        for i in range(1, size - 1):  # Exclude the last node from mutation
            if random.random() < mutGen:
                swap_idx = random.randint(1, size - 2)
                individual[i], individual[swap_idx] = individual[swap_idx], individual[i]
        return individual,
    
    toolbox.register("mutate", custom_mutate)
    toolbox.register("select_best", tools.selBest)
    toolbox.register("evaluate", calcular_distancia, distancias=distancias)
    population = toolbox.population(n=2000)
    
    for gen in range(1000):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.9, mutpb=0.93)
        for ind in offspring:
            ind[-1] = num_nodos - 1
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = tools.selBest(population + offspring, k=len(population))
        best_individual = tools.selBest(population, k=1)[0]
        best_individual[0] = 0
        mejor_distancia = calcular_distancia(best_individual, distancias)[0]
        print(f"Generación {gen+1} - Mejor distancia encontrada:", mejor_distancia)       
    best_individual = tools.selBest(population, k=1)[0]
    best_individual[0] = 0
    mejor_distancia = calcular_distancia(best_individual, distancias)[0]
    print("Mejor distancia final:", mejor_distancia)
    print("Secuencia de nodos:", best_individual) 
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")
    
if __name__ == "__main__":
    main()
 
