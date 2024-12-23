import numpy as np

from string import ascii_lowercase
from random import randint

from search_problem import Problem, Variable, Heuristic


class Queen(Variable):
    x: int
    y: int

    @staticmethod
    def attack(a: 'Queen', b: 'Queen') -> bool:
        if a.x == b.x:
            return True  # Attacking horizontally
        elif a.y == b.y:
            return True  # Attacking vertically
        elif a.x - a.y == b.x - b.y or a.x + a.y == b.x + b.y:
            return True  # Attacking diagonally
        return False

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_domain(self, problem: 'NQueens'):
        domain: list[tuple] = []
        n: int = problem.get_n()
        board: np.ndarray = problem.get_board()
        for i in range(n):
            for j in range(n):
                if board[i, j] == 0:
                    domain.append((i, j))
        return domain

    def set_value(self, value, problem: 'NQueens'):
        self.x, self.y = value
        problem.update_board()


class NQueens(Problem):
    n: int
    board: np.ndarray
    queens: np.ndarray

    def __init__(self, n: int):

        # Checking and setting n
        if n == 2 or n == 3:
            raise ValueError("The n queens problem is unsolvable for n = 2 and n = 3!")
        self.n = n

        # Initializing a random placement of n queens
        available = [(i, j) for i in range(n) for j in range(n)]
        queens: list = []
        for i in range(n):
            i_ = randint(0, len(available) - 1)
            x, y = available.pop(i_)
            queens.append(Queen(x, y))
        self.queens = np.array(queens, dtype=Queen)

        # Initializing board
        self.update_board()

    def get_n(self) -> int:
        return self.n

    def get_board(self):
        return self.board

    def update_board(self):
        n: int = self.get_n()
        self.board = np.zeros((n, n), dtype=int)
        for queen in self.queens:
            self.board[queen.x, queen.y] = 1

    def get_variables(self) -> np.ndarray:
        return self.queens

    def __str__(self) -> str:
        str_: str = " " * 5 + "   ".join(ascii_lowercase[:self.n])
        str_ += "\n   ┌" + "───┬" * (self.n - 1) + "───┐"
        for i in range(self.n):
            str_ += f"\n{i + 1:2d} │"
            for j in range(self.n):
                if self.board[i, j] == 1:
                    str_ += " ♕ │"
                else:
                    str_ += "   │"
            if i != self.n - 1:
                str_ += "\n   ├" + "───┼" * (self.n - 1) + "───┤"
        str_ += "\n   └" + "───┴" * (self.n - 1) + "───┘"
        return str_


class ThreatsHeuristic(Heuristic):

    @staticmethod
    def evaluate(problem: NQueens):
        evaluation: int = 0
        n: int = problem.get_n()
        queens: np.ndarray = problem.queens
        board: np.ndarray = problem.get_board()
        for i in range(n):
            queen: Queen = queens[i]
            other_queens = np.concat([queens[:i], queens[i + 1:]])
            for other_queen in other_queens:
                if Queen.attack(queen, other_queen):
                    evaluation += 1
        return evaluation
