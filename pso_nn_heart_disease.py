#libraries

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings

# suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# creating output directory
os.makedirs("output", exist_ok=True)


# loading dataset and scaling features
data = pd.read_csv("heart_cleveland_upload.csv")
X = data.drop("condition", axis=1)
y = data["condition"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# training baseline model with grid search
print("===== BASELINE NEURAL NETWORK MODEL =====")
param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (100,)],
    'learning_rate_init': [0.001, 0.01, 0.1]
}


baseline_model = GridSearchCV(MLPClassifier(max_iter=300, random_state=42), param_grid, cv=3)
baseline_model.fit(X_train, y_train)
baseline_predictions = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)


print("best hyperparameters:", baseline_model.best_params_)
print("validation accuracy:", baseline_accuracy)


# initializing PSO parameters
print("\n===== PSO-OPTIMIZED NEURAL NETWORK MODEL =====")

population_size = 10
num_dimensions = 2
informant_count = 3
num_generations = 20

inertia_weight, cognitive_coeff, social_coeff = 0.729, 1.49, 1.49
lower_bounds = [0.0001, 5]
upper_bounds = [0.1, 100]

accuracy_history = []



# defining fitness function
def compute_fitness(hyperparameters):
    learning_rate = hyperparameters[0]
    neurons = max(1, int(hyperparameters[1]))
    model = MLPClassifier(hidden_layer_sizes=(neurons,), learning_rate_init=learning_rate, max_iter=300, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return 1 - accuracy_score(y_test, predictions)



# creating particle class for PSO
class Particle:
    def __init__(self):
        self.position = [random.uniform(lower_bounds[i], upper_bounds[i]) for i in range(num_dimensions)]
        self.velocity = [random.uniform(-1, 1) for _ in range(num_dimensions)]
        self.fitness = compute_fitness(self.position)
        self.best_position = list(self.position)
        self.best_fitness = self.fitness
        self.informants = random.sample(range(population_size), informant_count)
        self.informant_best_position = list(self.position)
        self.informant_best_fitness = self.fitness


    def update_velocity(self):
        for i in range(num_dimensions):
            r1, r2 = random.random(), random.random()
            cognitive = cognitive_coeff * r1 * (self.best_position[i] - self.position[i])
            social = social_coeff * r2 * (self.informant_best_position[i] - self.position[i])
            self.velocity[i] = inertia_weight * self.velocity[i] + cognitive + social


    def update_position(self):
        for i in range(num_dimensions):
            self.position[i] += self.velocity[i]
            self.position[i] = max(lower_bounds[i], min(upper_bounds[i], self.position[i]))
        self.fitness = compute_fitness(self.position)


    def update_informant_best(self, swarm):
        top_informant = min(self.informants, key=lambda i: swarm[i].best_fitness)
        if swarm[top_informant].best_fitness < self.informant_best_fitness:
            self.informant_best_fitness = swarm[top_informant].best_fitness
            self.informant_best_position = list(swarm[top_informant].best_position)

# creating swarm
swarm = [Particle() for _ in range(population_size)]
global_best_particle = min(swarm, key=lambda p: p.best_fitness)
best_solution_position = list(global_best_particle.best_position)
best_solution_fitness = global_best_particle.best_fitness


# running PSO optimization
for gen in range(num_generations):
    for particle in swarm:
        particle.update_informant_best(swarm)
        particle.update_velocity()
        particle.update_position()

        if particle.fitness < particle.best_fitness:
            particle.best_fitness = particle.fitness
            particle.best_position = list(particle.position)

    best_particle = min(swarm, key=lambda p: p.best_fitness)

    if best_particle.best_fitness < best_solution_fitness:
        best_solution_fitness = best_particle.best_fitness
        best_solution_position = list(best_particle.best_position)

    current_accuracy = 1 - best_solution_fitness
    accuracy_history.append(current_accuracy)

    print(f"Generation {gen + 1}: Best Accuracy = {current_accuracy:.4f}")



# printing best results
print("\nOptimization Complete!")
print(f"Best Learning Rate: {best_solution_position[0]:.5f}")
print(f"Best Neurons in Hidden Layer: {int(best_solution_position[1])}")
print(f"Final PSO-NN Validation Accuracy: {1 - best_solution_fitness:.4f}")

print("\n===== COMPARISON RESULTS =====")
print(f"Baseline Model Accuracy: {baseline_accuracy:.4f}")
print(f"Optimized Model Accuracy: {1 - best_solution_fitness:.4f}")



# plotting PSO accuracy history
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_generations + 1), accuracy_history, marker='o')
plt.title('PSO Optimization Accuracy Across Generations')
plt.xlabel('Generation')
plt.ylabel('Validation Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig("output/pso_optimization_progress.png")
plt.show()



# plotting bar chart comparison
plt.figure(figsize=(6, 4))
plt.bar(['Baseline NN', 'PSO-Optimized NN'], [baseline_accuracy, 1 - best_solution_fitness], color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig("output/model_comparison.png")
plt.show()
