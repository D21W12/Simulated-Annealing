import numpy as np

from abc import ABC, abstractmethod
from copy import deepcopy
from random import choice, random

from search_problem import Problem, Heuristic, Variable


class SimulatedAnnealing:
    problem: Problem
    temp: float
    minimizing: bool
    schedule: 'AnnealingSchedule'
    heuristic: Heuristic

    def __init__(self, problem: Problem,
                 schedule: 'AnnealingSchedule',
                 heuristic: Heuristic,
                 starting_temp: float = 10,
                 minimizing: bool = True):
        self.problem = problem
        self.temp = starting_temp
        self.minimizing = minimizing
        self.schedule = schedule
        self.heuristic = heuristic

    @staticmethod
    def p_accept(h_old: float, h_new: float, T: float, minimizing: bool = True):
        if minimizing:
            return np.e ** ((h_old - h_new) / T)
        return np.e ** ((h_new - h_old) / T)

    def iterate(self):
        heuristic: Heuristic = self.heuristic
        schedule: AnnealingSchedule = self.schedule
        temp: float = self.temp
        h_old: float = heuristic.evaluate(self.problem)

        neighbour: Problem = deepcopy(self.problem)

        variables: np.ndarray = neighbour.get_variables()
        variable: Variable = choice(variables)
        domain: np.ndarray = variable.get_domain(neighbour)

        value = choice(domain)
        variable.set_value(value, neighbour)

        h_new: float = heuristic.evaluate(neighbour)

        # print("Problem:")
        # print(self.problem)
        # print("Proposed neighbour:")
        # print(neighbour)
        # print(f"H_old: {h_old}, H_new: {h_new}")
        # print(f"Temperature: {self.temp}")
        p_accept = self.p_accept(h_old, h_new, temp, self.minimizing)
        if self.accept(h_old, h_new):
            # print("Accepted!")
            self.problem = neighbour
        self.temp = schedule.cool(temp)

    def accept(self, h_old: float, h_new: float) -> float:
        p: int
        T: float = self.temp
        if self.minimizing:
            if h_old > h_new:
                return True
            p = self.p_accept(h_old, h_new, T, True)
            return random() < p
        if h_old < h_new:
            return True
        p = self.p_accept(h_old, h_new, T, False)
        return random() < p

    def run(self, iterations: int, print_iterations: bool = False):
        for i in range(iterations):
            self.iterate()
            if print_iterations:
                print(self.problem)
        return self.problem


class AnnealingSchedule(ABC):

    @staticmethod
    @abstractmethod
    def cool(temp: float) -> float:
        pass


class GeometricCooling(AnnealingSchedule):

    @staticmethod
    def cool(temp: float) -> float:
        return 0.97 * temp


class LinearCooling(AnnealingSchedule):
    cooling_factor: float

    def __init__(self, cooling_factor: float):
        self.cooling_factor = cooling_factor

    def cool(self, temp: float) -> float:
        return self.cooling_factor * temp
