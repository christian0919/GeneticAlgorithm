import sys
import tkinter
from math import e
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Definir la función objetivo
def objective_function(x, y):
    return e**((-(y+1)**2)-x**2)*(x-1)**2 - ((e**(-(x+1)**2))/3) + (e**(-(x**2)-(y**2)))*((10*x**3) - (10*x) + (10*y**3))

# Definir parámetros del algoritmo genético
population_size = 10  # Tamaño de la población en cada ciclo de cruza
mutation_rate = 0.1
num_generations = int(input('Introduce el número de generaciones: '))

# Definir límites para las coordenadas x y y
x_lower_bound = -4
x_upper_bound = 4
y_lower_bound = -4
y_upper_bound = 4

# Función para generar individuos aleatorios dentro de los límites
def generate_random_individual():
    x = random.uniform(x_lower_bound, x_upper_bound)
    y = random.uniform(y_lower_bound, y_upper_bound)
    return x, y

# Función para evaluar la aptitud de un individuo
def evaluate_fitness(individual):
    x, y = individual
    return objective_function(x, y)

# Función para aplicar la selección de la ruleta
def roulette_wheel_selection(population, fitness_scores, num_parents):
    total_fitness = sum(fitness_scores)
    probabilities = [fitness/total_fitness if fitness > 0 else 0 for fitness in fitness_scores]
    probabilities = [prob/sum(probabilities) for prob in probabilities]  # Normalizar las probabilidades
    selected = np.random.choice(len(population), size=num_parents, p=probabilities)
    return [population[idx] for idx in selected]

# Función para realizar el cruce de un punto
def one_point_crossover(parent1, parent2):
    x1, y1 = parent1
    x2, y2 = parent2
    crossover_point = random.randint(0, 1)  # Seleccionar un punto de cruce aleatorio
    if crossover_point == 0:
        child = x1, y2
    else:
        child = x2, y1
    return child

# Función para realizar el cruce uniforme
def uniform_crossover(parent1, parent2):
    x1, y1 = parent1
    x2, y2 = parent2
    mask = random.getrandbits(2)  # Generar una máscara de 2 bits para determinar qué atributos se heredan
    child_x = x1 if mask & 1 else x2
    child_y = y1 if mask & 2 else y2
    return child_x, child_y

# Función para aplicar mutación a un individuo
def mutate(individual):
    x, y = individual
    if random.random() < mutation_rate:
        # Generar nuevos valores mutados dentro de los límites
        new_x = max(x_lower_bound, min(x_upper_bound, x + random.uniform(-1, 1)))
        new_y = max(y_lower_bound, min(y_upper_bound, y + random.uniform(-1, 1)))
        individual = new_x, new_y
    return individual

# Algoritmo genético principal
def genetic_algorithm():
    # Generar la población inicial
    population = [generate_random_individual() for _ in range(population_size)]

    # Crear una figura 3D para visualizar los puntos generados en cada generación
    fig_all = plt.figure()
    ax_all = fig_all.add_subplot(111, projection='3d')

    # Definir los rangos de los ejes x y y
    x_range = np.linspace(x_lower_bound, x_upper_bound, 100)
    y_range = np.linspace(y_lower_bound, y_upper_bound, 100)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    z_mesh = objective_function(x_mesh, y_mesh)

    # Representar la función objetivo en 3D
    ax_all.plot_surface(x_mesh, y_mesh, z_mesh, cmap='viridis', alpha=0.5)

    # Coordenadas del punto más grande encontrado hasta el momento
    best_x, best_y = None, None
    best_fitness = float('-inf')

    for generation in range(num_generations):
        # Evaluar la aptitud de cada individuo en la población
        fitness_scores = [evaluate_fitness(individual) for individual in population]

        # Ordenar la población y los puntajes de aptitud en orden descendente
        fitness_scores, population = zip(*sorted(zip(fitness_scores, population), reverse=True))

        # Seleccionar los 5 mejores individuos de la población
        top_individuals = population[:5]

        # Imprimir los puntos de la población actual
        print(f"Generación {generation + 1}:")
        for i, individual in enumerate(population):
            print(f"Punto {i + 1}: {individual}")

        # Seleccionar a los padres mediante la selección de la ruleta utilizando los mejores individuos
        parents = roulette_wheel_selection(top_individuals, fitness_scores[:5], 2)

        # Generar descendencia mediante cruza y mutación
        offspring = []
        while len(offspring) < population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            # Seleccionar uno de los métodos de cruce disponibles aleatoriamente
            crossover_method = random.choice([one_point_crossover, uniform_crossover])
            child = crossover_method(parent1, parent2)

            child = mutate(child)

            # Verificar si el punto está dentro de los límites
            x_child, y_child = child
            if x_lower_bound <= x_child <= x_upper_bound and y_lower_bound <= y_child <= y_upper_bound:
                offspring.append(child)

        # Reemplazar la población anterior con la nueva generación
        population = offspring

        # Obtener coordenadas de la población y los nuevos individuos
        x_population = [individual[0] for individual in population]
        y_population = [individual[1] for individual in population]
        z_population = [objective_function(x, y) for x, y in population]
		

		# Calcular colores degradados para los puntos generados
        colors = [i/num_generations for i in range(len(population))]

        # Representar los puntos generados en la gráfica de todos los puntos
        ax_all.scatter(x_population, y_population, z_population, c=colors, cmap='plasma')

        # Buscar el punto más grande en la población actual
        max_index = np.argmax(z_population)
        max_x, max_y, max_fitness = x_population[max_index], y_population[max_index], z_population[max_index]

        # Verificar si el punto más grande de la población supera al mejor punto encontrado hasta el momento
        if max_fitness > best_fitness:
            best_x, best_y = max_x, max_y
            best_fitness = max_fitness

    # Crear una figura adicional para mostrar el mejor punto
    fig_best = plt.figure()
    ax_best = fig_best.add_subplot(111, projection='3d')
    ax_best.plot_surface(x_mesh, y_mesh, z_mesh, cmap='viridis', alpha=0.5)
    ax_best.scatter(best_x, best_y, best_fitness, color='red', label='Mejor Punto')
    ax_best.legend()

    # Mostrar las figuras con los puntos generados y el mejor punto encontrado
    plt.show()

# Ejecutar el algoritmo genético
genetic_algorithm()

