import numpy as np
import random

# Define the memory structure
class Memory:
    def __init__(self, scalar_count=10, vector_size=10):
        self.scalars = np.zeros(scalar_count)  # Scalar values
        self.vectors = [np.zeros(vector_size) for _ in range(scalar_count)]  # Vector values

# Define operations available in the search space
OPERATIONS = [
    "add_scalar", "subtract_scalar", "multiply_scalar", 
    "add_vector", "dot_product", "normalize_vector"
]

def random_operation():
    """Randomly select an operation."""
    return random.choice(OPERATIONS)

class Algorithm:
    """Representation of an algorithm with setup, predict, and learn components."""
    def __init__(self):
        self.setup_instructions = []
        self.predict_instructions = []
        self.learn_instructions = []

    def mutate(self):
        """Introduce random changes to the algorithm."""
        for component in [self.setup_instructions, self.predict_instructions, self.learn_instructions]:
            mutation_type = random.choice(["add", "remove", "modify"])
            if mutation_type == "add":
                component.append(random_operation())
            elif mutation_type == "remove" and component:
                component.pop(random.randint(0, len(component) - 1))
            elif mutation_type == "modify" and component:
                component[random.randint(0, len(component) - 1)] = random_operation()

def evaluate_algorithm(algorithm, train_data, validation_data):
    """Evaluate an algorithm's performance."""
    memory = Memory()
    # Execute setup phase
    for instruction in algorithm.setup_instructions:
        execute_instruction(memory, instruction)

    # Training phase
    for x, y in train_data:
        memory.vectors[0] = x
        execute_instructions(memory, algorithm.predict_instructions)
        memory.scalars[1] = y
        execute_instructions(memory, algorithm.learn_instructions)

    # Validation phase
    total_loss = 0
    for x, y in validation_data:
        memory.vectors[0] = x
        execute_instructions(memory, algorithm.predict_instructions)
        prediction = memory.scalars[1]
        total_loss += (prediction - y) ** 2  # Mean squared error

    return total_loss / len(validation_data)

def execute_instruction(memory, instruction):
    """Simulate the execution of an instruction."""
    if instruction == "add_scalar":
        memory.scalars[0] += memory.scalars[1]
    elif instruction == "subtract_scalar":
        memory.scalars[0] -= memory.scalars[1]
    elif instruction == "multiply_scalar":
        memory.scalars[0] *= memory.scalars[1]
    elif instruction == "add_vector":
        memory.vectors[0] += memory.vectors[1]
    elif instruction == "dot_product":
        memory.scalars[0] = np.dot(memory.vectors[0], memory.vectors[1])
    elif instruction == "normalize_vector":
        memory.vectors[0] /= np.linalg.norm(memory.vectors[0]) + 1e-9

def execute_instructions(memory, instructions):
    """Execute a list of instructions."""
    for instruction in instructions:
        execute_instruction(memory, instruction)

# New mutate_algorithm function
def mutate_algorithm(algorithm):
    """Creates a mutated copy of an existing algorithm."""
    new_algorithm = Algorithm()
    new_algorithm.setup_instructions = algorithm.setup_instructions[:]
    new_algorithm.predict_instructions = algorithm.predict_instructions[:]
    new_algorithm.learn_instructions = algorithm.learn_instructions[:]
    new_algorithm.mutate()
    return new_algorithm

# Evolutionary loop
def evolutionary_search(population_size, generations, train_data, validation_data):
    """Perform evolutionary search to discover algorithms."""
    population = [Algorithm() for _ in range(population_size)]
    for gen in range(generations):
        # Evaluate population
        scores = [evaluate_algorithm(algo, train_data, validation_data) for algo in population]
        best_index = np.argmin(scores)
        best_algorithm = population[best_index]
        print(f"Generation {gen + 1}: Best Loss = {scores[best_index]:.4f}")
        # Mutate to create a new generation
        population = [mutate_algorithm(best_algorithm) for _ in range(population_size)]
    return best_algorithm

# Example data
train_data = [(np.random.randn(10), np.random.randn(1)[0]) for _ in range(100)]
validation_data = [(np.random.randn(10), np.random.randn(1)[0]) for _ in range(20)]

# Run the evolutionary search
best_algorithm = evolutionary_search(population_size=5, generations=10, train_data=train_data, 
                                     validation_data=validation_data)
print("Best Algorithm Found:")
print("Setup:", best_algorithm.setup_instructions)
print("Predict:", best_algorithm.predict_instructions)
print("Learn:", best_algorithm.learn_instructions)