from typing import List
import random
import pandas as pd
from sklearn.metrics import classification_report, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Individual(object):
    GENES = "0123456789"

    def __init__(self, chromosome: list, fitness_func=None, **kwargs):
        self.chromosome = chromosome
        self.fitness_func = fitness_func
        self.kwargs = kwargs
        self.fitness = self.cal_fitness()

    @staticmethod
    def rand_choice_gene():
        return random.choice(Individual.GENES)

    @staticmethod
    def create_random(genes_size, fitness_func=None, **kwargs) -> 'Individual':
        genes = [Individual.rand_choice_gene() for i in range(genes_size)]
        return Individual(genes, fitness_func, **kwargs)

    def cal_fitness(self):
        fitness = 0
        self.kwargs['chromosome'] = self.chromosome
        if self.fitness_func is not None:
            fitness = self.fitness_func(**self.kwargs)
        else:
            for x in self.chromosome:
                fitness += x
        self.fitness = fitness
        return self.fitness


class GeneticAlgorithm:
    def __init__(self, population_size, max_generation=1000, genes_size=10, mutation_factor=0.2, rand_factor=0.3,
                 fitness_func=None, **kwargs):
        self.POPULATION_SIZE = population_size
        self.MAX_GENERATION = max_generation
        self.GENES_SIZE = genes_size
        self.MUTATION_FACTOR = mutation_factor
        self.RANDOM_FACTOR = rand_factor
        self.fitness_func = fitness_func
        self.kwargs = kwargs
        self.__reset()
        pass

    def __generate_random_population(self):
        for i in range(self.POPULATION_SIZE):
            self.population.append(Individual.create_random(self.GENES_SIZE, self.fitness_func, **self.kwargs))

    def __reset(self):
        self.population: List[Individual] = []  # populasi saat ini
        self.best_individual: Individual = None

    def __crossover(self, a: 'Individual', b: 'Individual') -> 'Individual':
        child_chromosome = []
        for genA, genB in zip(a.chromosome, b.chromosome):
            prob = random.random()
            if prob < self.RANDOM_FACTOR:
                child_chromosome.append(genA)
            else:
                child_chromosome.append(genB)
        return Individual(child_chromosome, fitness_func=self.fitness_func, **self.kwargs)

    def __mutation(self, individual: 'Individual') -> 'Individual':
        index_to_mutate = []
        index = [i for i in range(self.GENES_SIZE)]
        for i in range(int(self.MUTATION_FACTOR * self.GENES_SIZE)):
            tmp = random.choice(index)
            index.remove(tmp)
            index_to_mutate.append(tmp)
        for idx in index_to_mutate:
            individual.chromosome[idx] = Individual.rand_choice_gene()
        individual.cal_fitness()
        return individual

    def __mate(self, parA: 'Individual', parB: 'Individual') -> 'Individual':
        child = self.__crossover(parA, parB)
        mutated_child = self.__mutation(child)
        return mutated_child

    def __check_population_is_same(self) -> bool:
        for individual in self.population:
            if not all([a == b for a, b in zip(self.population[0].chromosome, individual.chromosome)]):
                return False
        return True

    def print_population(self):
        for individual in self.population:
            print(individual.chromosome, individual.fitness)
        print('\n')

    def run(self):
        self.__reset()

        self.__generate_random_population()

        self.print_population()

        for id_generation in range(1, self.MAX_GENERATION + 1):
            self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

            if self.best_individual is None or self.best_individual.fitness < self.population[0].fitness:
                self.best_individual = self.population[0]

            if self.__check_population_is_same():
                break

            new_generations = []

            # 10% fittest population diambil utk generasi berikutnya
            size = (10 * self.POPULATION_SIZE) // 100
            new_generations.extend(self.population[: size])

            # 50% fittest population dikawinkan, child diambil utk generasi berikutnya
            size = self.POPULATION_SIZE - size
            for i in range(size):
                parentA = random.choice(self.population[: self.POPULATION_SIZE // 2])
                parentB = random.choice(self.population[: self.POPULATION_SIZE // 2])
                new_generations.append(self.__mate(parentA, parentB))

            self.population = new_generations

            self.print_population()

        print(id_generation)


def fitness_function_knn(**kwargs) -> float:
    try:
        k = int("".join(x for x in kwargs['chromosome']))
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(kwargs['train_x'], kwargs['train_y'])
        pred_y = model.predict(kwargs['test_x'])
        return precision_score(kwargs['test_y'], pred_y, average='micro')
    except Exception as e:
        # exit(e.args[0])
        return 0


if __name__ == '__main__':

    # Preparing dataset

    headers = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    df = pd.read_csv('dataset/car.data', header=None, names=headers)
    classes = []

    for header in headers:
        le = LabelEncoder()
        df[header + '_le'] = le.fit_transform(df[header])
        if header is 'class':
            classes = le.classes_

    df = df.drop(headers, axis='columns')

    # Splitting dataset

    train_x, test_x, train_y, test_y = train_test_split(df[[item + '_le' for item in headers[:-1]]],
                                                        df['class_le'], random_state=3, test_size=0.1)

    # Running with optimization

    ga = GeneticAlgorithm(50, 10, 3, fitness_func=fitness_function_knn, train_x=train_x, train_y=train_y,
                          test_x=test_x, test_y=test_y)
    ga.run()
