import numpy as np

from abc import ABC, abstractmethod


class Variable(ABC):

    @abstractmethod
    def get_domain(self, problem: 'Problem'):
        pass

    @abstractmethod
    def set_value(self, value, problem: 'Problem'):
        pass


class Problem(ABC):

    @abstractmethod
    def get_variables(self) -> np.ndarray:
        pass


class Heuristic(ABC):

    @staticmethod
    @abstractmethod
    def evaluate(problem: Problem) -> int | float:
        pass
