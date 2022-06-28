"""
GA optimizer.
by Hayaho Sato, May, 2021.
"""

import copy
import random

import matplotlib.pyplot as plt
import numpy as np

from single_objective_optimizer.model.settings import *
from single_objective_optimizer.model import optimizer


class GA_JGG(optimizer.OPTIMIZER):
    """
    GA optimizer class (with JGG strategy, REXstar crossover)
    """
    def __init__(self, obj_func, population_size=None, n_p=n+1, n_c=4*n, step_size=1.0,
                 pop_range_upper=1.0, pop_range_lower=-1.0,
                 plot_evolution_or_not=False):
        """
        init function where all parameters are initialized.
        :param obj_func: objective function to be minimized
        :param population_size: population size n_pop
        :param n_p: parents size n_p for multi-parent crossover (REXstar)
        :param n_c: children size n_c
        :param step_size: step size t for REXstar
        :param pop_range_upper: upper limit of population generation
        :param pop_range_upper: lower limit of population generation
        :param plot_evolution_or_not: if you want to plot evolution, please set True (only valid when n=2)
        """
        super().__init__(obj_func, plot_evolution_or_not)

        # parameters
        if population_size is None:
            self.population_size = 10*n
        else:
            self.population_size = population_size
        self.n_p = n_p
        self.n_c = n_c
        self.step_size = step_size
        self.pop_range_upper = pop_range_upper
        self.pop_range_lower = pop_range_lower

        # groups, vecs
        self.population = np.zeros((self.population_size, n))  # population
        self.parents = np.zeros((self.n_p, n))
        self.children = np.zeros((self.n_c, n))
        self.evaluation_vec = np.zeros(self.population_size)
        self.evaluation_vec_of_parents = np.zeros(self.n_p)
        self.x_g = np.zeros(n)
        self.x_b = np.zeros(n)

        self.generate_initial_population()

    def proceed_generation(self):
        """
        proceed generation calling all operation
        :return:
        """
        if self.plot_evolution_or_not and n == 2:
            self.plot_evolution()
        self.parent_selection()
        self.compute_descending_direction()
        self.REXstar()
        self.evaluation_and_selection()
        self.generation += 1

    def generate_initial_population(self):
        """
        Generate initial population and evaluate them.
        :return:
        """
        # generate initial population
        self.population = (self.pop_range_lower + (self.pop_range_upper - self.pop_range_lower)
                           * np.random.rand(self.population_size, n))
        self.generation = 99999  # special value representing initial generation
        with open('Generation.csv', 'w') as f:
            print(str(self.generation), file=f)
        # evaluate
        self.evaluation_vec = self.obj_func(self.population)
        self.num_of_evaluation += self.population.shape[0]
        # elite preservation
        idx_min = np.argmin(self.evaluation_vec)
        if self.best_eval > self.evaluation_vec[idx_min]:
            self.best_eval = self.evaluation_vec[idx_min]
            self.best_solution = copy.deepcopy(self.population[idx_min, :])
        self.generation = 0

    def parent_selection(self):
        """
        Randomly select parent. This is non-restoring extraction.
        :return:
        """
        self.parents = np.zeros((self.n_p, n))
        self.children = np.zeros((self.n_c, n))
        self.evaluation_vec_of_parents = np.zeros(self.n_p)
        # non-restoring extraction
        for i in range(0, self.n_p):
            index = random.randint(0, self.population_size - 1 - i)
            self.parents[i, :] = copy.deepcopy(self.population[index, :])
            self.evaluation_vec_of_parents[i] = copy.deepcopy(self.evaluation_vec[index])
            self.population = np.delete(self.population, index, 0)
            self.evaluation_vec = np.delete(self.evaluation_vec, index)

    def compute_descending_direction(self):
        """
        Compute (universal) descending direction x_g - x_b.
        Additional evaluation of mirrored parents is needed.
        :return:
        """
        self.x_g = np.sum(self.parents, axis=0) / self.n_p
        xig_vectors = self.parents - self.x_g
        mirror_parents = self.parents - 2.0 * xig_vectors
        two_parents = np.r_[self.parents, mirror_parents]
        mirror_evaluation_vec = self.obj_func(mirror_parents)
        self.num_of_evaluation += mirror_parents.shape[0]
        two_evaluation_vec = np.r_[self.evaluation_vec_of_parents, mirror_evaluation_vec]
        ranks = np.argsort(two_evaluation_vec)
        sorted_two_parents = two_parents[ranks, :]
        upper_individuals = sorted_two_parents[:self.n_p, :]
        self.x_b = np.sum(upper_individuals, axis=0) / self.n_p

    def REXstar(self):
        """
        REXstar crossover method to generate children.
        :return:
        """
        for i in range(0, self.n_c):
            k_tj = np.eye(n)
            for j in range(0, n):
                k_tj[j, j] = np.random.rand() * self.step_size

            k_i = np.zeros(self.n_p)
            for k in range(0, self.n_p):
                k_i[k] = (np.random.rand() * 2.0 - 1.0) * np.sqrt(3.0 / self.n_p)

            sum_of_weighted_diff_parents = 0
            for l in range(0, self.n_p):
                sum_of_weighted_diff_parents += k_i[l] * (self.parents[l, :] - self.x_g)
            self.children[i, :] = self.x_g + (k_tj @ ((self.x_b - self.x_g).reshape((n, 1)))).reshape(n) + sum_of_weighted_diff_parents

    def evaluation_and_selection(self):
        """
        Evaluate children and select ones to next generation.
        :return:
        """
        # selection of n_p individuals
        evaluation_vec_of_children = self.obj_func(self.children)
        self.num_of_evaluation += self.children.shape[0]
        ranks = np.argsort(evaluation_vec_of_children)
        sorted_children = self.children[ranks, :]
        sorted_evaluation_vec_of_children = evaluation_vec_of_children[ranks]
        selected_children = sorted_children[:self.n_p, :]
        evaluation_vec_of_selected_children = sorted_evaluation_vec_of_children[:self.n_p]

        # elite preservation
        if self.best_eval > sorted_evaluation_vec_of_children[0]:
            self.best_eval = sorted_evaluation_vec_of_children[0]
            self.best_solution = copy.deepcopy(sorted_children[0, :])

        # add selected n_p individuals to population
        self.population = np.r_[self.population, selected_children]
        self.evaluation_vec = np.r_[self.evaluation_vec, evaluation_vec_of_selected_children]

    def plot_evolution(self):
        """
        Plot evolution of population on the contour color map of obj function.
        :return:
        """
        ax = plt.subplot()

        # obj_funcのコンター図
        cont = ax.contour(self.X_PLOT, self.Y_PLOT, self.obj_func_profit, levels=10)
        cont.clabel(fmt='%1.1f', fontsize=12)

        # 集団，および最良解
        ax.scatter(self.population[:, 0], self.population[:, 1], color='black')
        ax.scatter(self.best_solution[0], self.best_solution[1], color='red')

        # label
        plt.xlabel('x_1')
        plt.ylabel('x_2')

        ax.set_xlim(plot_range)
        ax.set_ylim(plot_range)
        ax.set_aspect('equal')

        plt.savefig(str(self.generation + 1).zfill(2) + '.png')

        plt.clf()
