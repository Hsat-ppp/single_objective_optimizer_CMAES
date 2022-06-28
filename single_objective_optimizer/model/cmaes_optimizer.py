"""
CMA-ES optimizer.
by Hayaho Sato, May, 2021.
"""

import copy

import matplotlib.pyplot as plt
import numpy as np

from single_objective_optimizer.model.settings import *
from single_objective_optimizer.model import optimizer

approximated_E_normal = (np.sqrt(n) * (1.0 - (1.0 / (4.0 * n)) + (1.0 / (21.0 * n * n))))


class CMAES(optimizer.OPTIMIZER):
    """
    CMA-ES optimizer class
    """
    def __init__(self, obj_func, population_size=None,
                 mu=None, mean=None, cov_mat_coef=1.0, step_size=1.00,
                 plot_evolution_or_not=False):
        """
        init function where all parameters are initialized.
        :param obj_func: objective function to be minimized (input: data (pop_size*n mat), output: value (pop_size vec))
        :param population_size: population size (default value will be set)
        :param mu: up to mu-th population will be used for update with positive weight coefficient
        :param cov_mat_coef: represent range of initial variation. That is, initial C = cov_mat_coef * I
        :param mean: distribution mean (if None, be set to np.zeros(n))
        :param step_size: step size (if None, be set to 1.00)
        :param plot_evolution_or_not: if you want to plot evolution, please set True (only valid when n=2)
        """
        super().__init__(obj_func, plot_evolution_or_not)

        # compute preliminary parameters
        if population_size is None:
            self.population_size = int(4 + np.floor(3 * np.log(n)))
        else:
            self.population_size = population_size
        self.weight_pre = np.array([np.log((self.population_size + 1.0) / 2.0) - np.log(i)
                                    for i in range(1, self.population_size+1, 1)])
        if mu is None:
            self.mu = np.floor(self.population_size / 2.0).astype(np.int64)
        else:
            self.mu = mu
        self.mu_eff = (np.sum(self.weight_pre[:self.mu])**2) / (np.sum(self.weight_pre[:self.mu]**2))
        self.mu_eff_minus = (np.sum(self.weight_pre[self.mu:])**2) / (np.sum(self.weight_pre[self.mu:]**2))

        # compute covariance matrix adaption coefficients
        self.alpha_cov = 2.0
        self.c_c = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n)
        self.c_1 = self.alpha_cov / (((n + 1.3)**2) + self.mu_eff)
        self.c_mu = np.min([1 - self.c_1,
                            self.alpha_cov * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / (
                                        ((n + 2.0) ** 2) + self.alpha_cov * self.mu_eff / 2.0)])

        # compute weight coefficients
        self.sum_of_weight_pre_plus = np.sum(np.abs(self.weight_pre[self.weight_pre >= 0]))
        self.sum_of_weight_pre_minus = np.sum(np.abs(self.weight_pre[self.weight_pre < 0]))
        self.alpha_mu_minus = 1 + self.c_1 / self.c_mu
        self.alpha_mueff_minus = 1 + (2.0 * self.mu_eff_minus) / (self.mu_eff + 2.0)
        self.alpha_posdef_minus = (1 - self.c_1 - self.c_mu) / (n * self.c_mu)
        self.weight = np.zeros_like(self.weight_pre)
        for i in range(len(self.weight)):
            if self.weight_pre[i] >= 0:
                self.weight[i] = self.weight_pre[i] * 1.0 / self.sum_of_weight_pre_plus
            else:
                self.weight[i] = self.weight_pre[i] * np.min([self.alpha_mu_minus, self.alpha_mueff_minus, self.alpha_posdef_minus]) / self.sum_of_weight_pre_minus
        self.c_m = 1.0

        # step_size control
        self.c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * np.max([0, np.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0]) + self.c_sigma

        # evolution paths
        self.evol_path_sigma = np.zeros(n)
        self.evol_path_c = np.zeros(n)

        # cov matrix and eigen matrix
        self.cov_matrix = cov_mat_coef * np.eye(n)  # default: C=I
        self.eigen_values = None
        self.eigen_vec_matrix = None
        self.eigen_val_matrix = None
        self.cov_mat_inv_sqrt = None
        self.eigen_value_decomposition()

        # x, y, z and evaluation by obj_func
        self.x = np.zeros((self.population_size, n))
        self.y = np.zeros((self.population_size, n))
        self.y_sum_w = np.zeros(n)
        self.z = np.zeros((self.population_size, n))
        self.y_rescaled = np.zeros((self.population_size, n))  # rescaled vec to control positive definiteness
        self.z_rescaled = np.zeros((self.population_size, n))  # rescaled vec to control positive definiteness
        self.evaluation_vec = np.zeros(self.population_size)

        # mean and step size
        if mean is None:
            self.mean = np.zeros(n)
        else:
            self.mean = copy.deepcopy(mean)
        self.step_size = step_size

    def proceed_generation(self):
        """
        Proceed generation calling all operation
        :return:
        """
        self.sample_new_population()
        self.evaluation_and_sort()
        if self.plot_evolution_or_not and n == 2:
            self.plot_evolution()
        self.selection_and_recombination()
        self.step_size_control()
        self.cov_matrix_adaption()
        self.eigen_value_decomposition()
        self.generation += 1

    def sample_new_population(self):
        """
        Sample new population from N(mean, (step_size**2) * cov_matrix)
        :return:
        """
        # generate sampling points
        self.z = np.random.normal(loc=0.0, scale=1.0, size=(self.population_size, n))
        self.z = self.z.T  # transpose to compute for each individuals
        self.y = self.eigen_vec_matrix @ self.eigen_val_matrix @ self.z
        self.z = self.z.T  # recover transpose
        self.y = self.y.T  # recover transpose
        self.x = self.mean + self.step_size * self.y

    def evaluation_and_sort(self):
        """
        Evaluate individuals x and sort x, y, z in increasing order.
        As a result, f(x[1]) < f(x[2]) < ... < f(x[population_size]).
        :return:
        """
        self.evaluation_vec = self.obj_func(self.x)
        self.num_of_evaluation += self.x.shape[0]
        ranks = np.argsort(self.evaluation_vec)
        # sort
        self.evaluation_vec = self.evaluation_vec[ranks]
        self.x = self.x[ranks, :]
        self.y = self.y[ranks, :]
        self.z = self.z[ranks, :]
        # elite preservation
        if self.best_eval > self.evaluation_vec[0]:
            self.best_eval = self.evaluation_vec[0]
            self.best_solution = copy.deepcopy(self.x[0, :])

    def selection_and_recombination(self):
        """
        Update mean vector, computing sum of w * y, using best mu individuals
        :return:
        """
        # sum up from 0 to mu-1
        self.y_sum_w = np.zeros(n)
        for i in range(self.mu):
            self.y_sum_w += self.weight[i] * self.y[i, :]
        self.mean = self.mean + self.c_m * self.step_size * self.y_sum_w

    def step_size_control(self):
        """
        Update step size, computing evolution path sigma.
        :return:
        """
        self.evol_path_sigma = ((1.0 - self.c_sigma) * self.evol_path_sigma
                                + (np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * self.cov_mat_inv_sqrt @ (self.y_sum_w.reshape((n, 1)))).reshape(n))
        self.step_size = (self.step_size
                          * np.exp((self.c_sigma / self.d_sigma)
                                   * ((np.linalg.norm(self.evol_path_sigma) / approximated_E_normal) - 1.0)))

    def cov_matrix_adaption(self):
        """
        Update cov matrix, computing evolution path.
        :return:
        """
        h_sigma = 0
        if (np.linalg.norm(self.evol_path_sigma) / np.sqrt(1.0 - (1.0 - self.c_sigma)**(2.0*(self.generation+1)))) < (1.4 + 2.0 / (n + 1)) * approximated_E_normal:
            h_sigma = 1.0
        self.evol_path_c = (1.0 - self.c_c) * self.evol_path_c + h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * self.y_sum_w
        weight_converted = np.zeros(self.population_size)
        for i in range(self.population_size):
            if self.weight[i] >= 0:
                weight_converted[i] = self.weight[i] * 1.0
            else:
                weight_converted[i] = self.weight[i] * n / np.linalg.norm((self.cov_mat_inv_sqrt @ (self.y[i, :].reshape(n, 1))))**2
        delta_h_sigma = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        sum_of_products_of_y = np.zeros((n, n))
        for i in range(self.population_size):
            sum_of_products_of_y += weight_converted[i] * (self.y[i, :].reshape((n, 1))) @ (self.y[i, :].reshape((1, n)))
        self.cov_matrix = ((1.0 + self.c_1 * delta_h_sigma - self.c_1 - self.c_mu * np.sum(self.weight)) * self.cov_matrix
                           + self.c_1 * (self.evol_path_c.reshape((n, 1))) @ (self.evol_path_c.reshape((1, n)))
                           + self.c_mu * sum_of_products_of_y)

    def eigen_value_decomposition(self):
        """
        Perform eigen value decomposition of cov matrix.
        Also compute sqrt of inverse of cov matrix for following computation.
        :return:
        """
        self.eigen_values, self.eigen_vec_matrix = np.linalg.eig(self.cov_matrix)
        self.eigen_values = self.eigen_values.real  # 複素数が混じる場合があるため，実数のみ取り出す (おそらく問題は生じない)
        self.eigen_vec_matrix = self.eigen_vec_matrix.real  # 複素数が混じる場合があるため，実数のみ取り出す (おそらく問題は生じない)
        self.eigen_val_matrix = np.diag(np.sqrt(self.eigen_values))
        self.cov_mat_inv_sqrt = self.eigen_vec_matrix @ np.diag(1.0 / np.sqrt(self.eigen_values)) @ self.eigen_vec_matrix.T

    def plot_evolution(self):
        """
        Plot evolution of population on the contour color map of obj function.
        :return:
        """
        ax = plt.subplot()

        # obj_funcのコンター図
        cont = ax.contour(self.X_PLOT, self.Y_PLOT, self.obj_func_profit, levels=10)
        cont.clabel(fmt='%1.1f', fontsize=12)

        # Cの等確率線
        def gaussian_function(cx, cy):
            x = np.array([cx, cy])
            return np.exp(-0.5 * (x - self.mean).T @ np.linalg.inv(self.step_size * self.step_size * self.cov_matrix) @ (x - self.mean)) / np.sqrt(
                np.linalg.det(self.step_size * self.step_size * self.cov_matrix) * (2 * np.pi) ** n)

        C_profit = np.vectorize(gaussian_function)(self.X_PLOT, self.Y_PLOT)
        ax.contour(self.X_PLOT, self.Y_PLOT, C_profit, levels=[i for i in np.arange(0.01, 0.10, 0.01)],
                   colors=['yellow'], linewidths=[0.3], linestyles=['dashed'])

        # sample pointsの散布図，中心点，および最良解
        ax.scatter(self.x[:, 0], self.x[:, 1], color='black')
        ax.scatter(self.mean[0], self.mean[1], color='green')
        ax.scatter(self.best_solution[0], self.best_solution[1], color='red')

        # label
        plt.xlabel('x_1')
        plt.ylabel('x_2')

        ax.set_xlim(plot_range)
        ax.set_ylim(plot_range)
        ax.set_aspect('equal')

        plt.show()
