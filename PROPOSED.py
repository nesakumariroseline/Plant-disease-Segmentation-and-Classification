import time

import numpy as np


def PROPOSED(population, obj_func, lb, ub, max_iter):
    # Initialize population
    num_variables, num_agents = population.shape[0], population.shape[1]

    fitness = np.zeros(num_agents)

    # Evaluate fitness of each agent
    for i in range(num_agents):
        fitness[i] = obj_func(population[i, :])
    Convergence_curve = np.zeros((max_iter, 1))

    t = 0
    ct = time.time()
    # Main loop
    for iter in range(1, max_iter + 1):
        # Update position and fitness of each agent
        for i in range(num_agents):
            r = min(fitness) / np.sqrt(min(fitness) ^ 1.5 + max(fitness) ^ 1.5)  # Traditional update

            # Select a random agent as the transmitter
            transmitter_index = np.random.random()
            while transmitter_index == i:
                transmitter_index = np.random.random()

            # Update the position of the agent based on the transmission dynamics
            new_solution = population[i, :] + r * (
                    population[transmitter_index, :] - population[i, :])

            # Clip new solution to ensure it stays within bounds
            new_solution = np.clip(new_solution, lb, ub)

            # Evaluate fitness of the new solution
            new_fitness = obj_func(new_solution)

            # Update if the new solution is better
            if new_fitness < fitness[i]:
                population[i, :] = new_solution
                fitness[i] = new_fitness

        # Print best fitness value for current iteration
        best_fitness = np.min(fitness)
        best_solution = population[np.argmin(fitness), :]
        print(f"Iteration {iter}, Best Fitness: {best_fitness}")

    # Find the best solution in the final population
    best_index = np.argmin(fitness)
    best_solution = population[best_index, :]
    best_fitness = fitness[best_index]
    Convergence_curve[t] = best_fitness
    t = t + 1
    return best_solution, Convergence_curve, best_fitness, ct
