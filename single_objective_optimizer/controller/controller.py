import argparse
import json
import logging

import numpy as np
import tqdm

from single_objective_optimizer.model.settings import *
from single_objective_optimizer.model.cmaes_optimizer import CMAES
from single_objective_optimizer.model.dd_cmaes_optimizer import DDCMAES
from single_objective_optimizer.model.GA_optimizer import GA_JGG
from single_objective_optimizer.model import functions_to_be_optimized
from single_objective_optimizer.utils.utils import set_seed_num

logger = logging.getLogger('info_logger')


def get_argparser_options():
    parser = argparse.ArgumentParser(description='''
                                    This is a single objective optimizer 
                                    based on CMA-ES and dd-CMA-ES, its advanced version.
                                    ''')
    parser.add_argument('-g', '--num_of_generations', default=100, type=int,
                        help='number of generations (iterations)')
    parser.add_argument('-p', '--population_size', type=int,
                        help='population size or number of individuals.')
    parser.add_argument('-s', '--seed_num', type=int,
                        help='seed number for reproduction.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='if set, we disables progress bar.')
    args = parser.parse_args()
    return args


def create_output_file():
    # create output file
    with open('convergence_history.csv', 'w'):
        pass
    with open('best_solution_history.csv', 'w'):
        pass


def optimize():
    # get args
    args = get_argparser_options()
    assert args.num_of_generations >= 1, 'Option "num_of_generations" need to be positive. ' \
                                         'Got: {}'.format(args.num_of_generations)
    if not args.population_size is None:
        assert args.population_size >= 1, 'Option "population_size" need to be positive. ' \
                                          'Got: {}'.format(args.population_size)
    # save args
    logger.info('args options')
    logger.info(args.__dict__)
    with open('args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # fix seed number and save
    set_seed_num(args.seed_num)

    # create output file and folder
    create_output_file()

    # create optimizer
    optimizer = GA_JGG(functions_to_be_optimized.sphere_function,
                        population_size=args.population_size,
                       plot_evolution_or_not=True)

    # optimization iteration
    logger.info('Optimization start.')
    iterator = None
    if args.quiet:
        iterator = range(args.num_of_generations)
    else:
        iterator = tqdm.tqdm(range(args.num_of_generations))
    for g in iterator:
        with open('Generation.csv', 'w') as f:
            print(g + 1, file=f)
        optimizer.proceed_generation()
        with open('convergence_history.csv', 'a') as f:
            print(g + 1, optimizer.num_of_evaluation, optimizer.best_eval, sep=',', file=f)
        with open('best_solution_history.csv', 'a') as f:
            print(*optimizer.best_solution, sep=',', file=f)

    logger.info('Optimization end. Total generations: {0}, Best eval: {1}'.format(
        args.num_of_generations, optimizer.best_eval))
