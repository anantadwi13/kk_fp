from typing import List
import random


class Individual(object):
    GENES = "0123456789"

    def __init__(self, chromosome: list, fitness_func=None, **kwargs):
        self.chromosome = chromosome
        self.fitness_func = fitness_func
        self.kwargs = kwargs
        self.fitness = self.cal_fitness()

    @staticmethod
    def rand_choice_gene(**kwargs):
        """
        Args:
            gene_type: ['digit', 'array_range_float', 'range_int', 'range_float']
        Returns:
            list
        """
        try:
            if kwargs['gene_type'] == 'digit':
                return random.choice(Individual.GENES)
            elif kwargs['gene_type'] == 'array_range_float':
                gene = []
                for i in range(kwargs['num_features']):
                    gene.append(random.random() * (kwargs['range_end'] - kwargs['range_start']) + kwargs['range_start'])
                return gene
            elif kwargs['gene_type'] == 'range_int':
                return random.randrange(kwargs['range_start'], kwargs['range_end'])
            elif kwargs['gene_type'] == 'range_float':
                return random.random() * (kwargs['range_end'] - kwargs['range_start']) + kwargs['range_start']
            else:
                exit('Gene type unknown!')
        except Exception as e:
            exit("Param{} undeclared!".format(''.join([' '+i for i in e.args])))

    @staticmethod
    def create_random(genes_size, fitness_func=None, **kwargs) -> 'Individual':
        genes = [Individual.rand_choice_gene(**kwargs) for i in range(genes_size)]
        return Individual(genes, fitness_func, **kwargs)

    def print(self):
        print(self.chromosome, self.fitness, sep='\n')

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
                 fitness_func=None, fitness_sort='desc', **kwargs):
        self.POPULATION_SIZE = population_size
        self.MAX_GENERATION = max_generation
        self.GENES_SIZE = genes_size
        self.MUTATION_FACTOR = mutation_factor
        self.RANDOM_FACTOR = rand_factor
        self.fitness_func = fitness_func
        self.fitness_sort = True if fitness_sort == 'desc' else False  # True => high to low
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
            individual.chromosome[idx] = Individual.rand_choice_gene(**self.kwargs)
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
            individual.print()
        print('\n')

    def run(self):
        self.__reset()

        self.__generate_random_population()

        self.print_population()

        for id_generation in range(1, self.MAX_GENERATION + 1):
            self.population = sorted(self.population, key=lambda x: x.fitness, reverse=self.fitness_sort)

            if self.best_individual is None or (self.fitness_sort and self.best_individual.fitness < self.population[0].fitness)\
                    or (not self.fitness_sort and self.best_individual.fitness > self.population[0].fitness):
                self.best_individual = self.population[0]

            new_generations = []

            # 10% fittest population diambil utk generasi berikutnya
            size = (10 * self.POPULATION_SIZE) // 100
            new_generations.extend(self.population[: size])

            # 100% fittest population dikawinkan, child diambil utk generasi berikutnya
            size = self.POPULATION_SIZE - size
            for i in range(size):
                parentA = random.choice(self.population[: self.POPULATION_SIZE])
                parentB = random.choice(self.population[: self.POPULATION_SIZE])
                new_generations.append(self.__mate(parentA, parentB))

            self.population = new_generations

            print("Gen #{}".format(id_generation))
            self.best_individual.print()
            print()
