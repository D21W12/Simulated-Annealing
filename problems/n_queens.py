import numpy as np

from string import ascii_lowercase, ascii_uppercase
from random import randint

from search_problem import Problem, Variable, Heuristic


ASCII = ascii_lowercase + ascii_uppercase


class Queen(Variable):
    """
    Class representing queen instances of the n queens problem. Stores the position of a singular queen.
    """
    x: int
    y: int

    @staticmethod
    def attacking(a: 'Queen', b: 'Queen') -> bool:
        """
        Static method for checking if two queens attack each other.

        Args:
            a (Queen): Queen number 1.
            b (Queen): Queen number 2.

        Returns:
            bool: True if the two queens are attacking each other, False otherwise.
        """
        if a.x == b.x:
            return True  # Attacking horizontally
        elif a.y == b.y:
            return True  # Attacking vertically
        elif a.x - a.y == b.x - b.y or a.x + a.y == b.x + b.y:
            return True  # Attacking diagonally
        return False  # No attack

    def __init__(self, x: int, y: int):
        """
        Constructor of the Queen class.

        Args:
            x (int): The x position of the queen (i.e. the row index).
            y (int): The y position of the queen (i.e. the column index).
        """
        self.x = x
        self.y = y

    def get_x(self):
        """
        Getter of the x position (row index) of the queen.
        """
        return self.x

    def get_y(self):
        """
        Getter of the y position (column index) of the queen.
        """
        return self.y

    def get_domain(self, problem: 'NQueens') -> list[tuple[int, int]]:
        """
        Getter of the domain of the queen instance. I.e. all possible positions the
        queen can take given a single problem instance.

        Args:
            problem (NQueens): The problem this queen is included in.

        Returns:
            list[tuple[int, int]]: List containing all available positions as a tuple of row and column index.
        """

        domain: list[tuple[int, int]] = []
        n: int = problem.get_n()
        board: np.ndarray = problem.get_board()

        # Looping through all positions on the board
        for i in range(n):
            for j in range(n):

                # If a position on the board is still empty, add it to the domain
                # of possible new positions of the queen.
                if board[i, j] == 0:
                    domain.append((i, j))

        return domain

    def set_value(self, value: tuple[int, int], problem: 'NQueens'):
        """
        Setter for the position of the queen. Updates the board of the problem instance
        accordingly.

        Args:
            value (tuple[int, int]): The position of the queen as a tuple of row and column index.
        """
        board = problem.get_board()
        board[self.x, self.y] = 0  # Erasing previous position of queen from board
        self.x, self.y = value
        board[self.x, self.y] = 1


class NQueens(Problem):
    """
    Class representing an instance of the n queens problem, to be solved by
    simulated annealing.
    """
    n: int
    board: np.ndarray
    queens: np.ndarray

    def __init__(self, n: int):
        """
        Constructor of the NQueens class.

        Args:
            n (int): The number of queens and board size of the problem instance.
        """

        # Checking and setting n
        if n == 2 or n == 3:
            raise ValueError("The n queens problem is unsolvable for n = 2 and n = 3!")
        self.n = n

        # Initialize empty board
        self.board = np.zeros((n, n), dtype=int)

        # Initializing a random placement of n queens
        available = [(i, j) for i in range(n) for j in range(n)]
        queens: list = []
        for i in range(n):
            i_ = randint(0, len(available) - 1)
            x, y = available.pop(i_)
            queens.append(Queen(x, y))
            self.board[x, y] = 1
        self.queens = np.array(queens, dtype=Queen)


    def get_n(self) -> int:
        """
        Getter for the n (i.e. number of queens and board size) of the problem instance.
        """
        return self.n

    def get_board(self) -> np.ndarray:
        """
        Getter for the numpy n-dimensional array representing the board of the problem instance.
        """
        return self.board

    def get_variables(self) -> np.ndarray:
        """
        Getter for all variables of the problem (i.e. all the queen instances).
        """
        return self.queens

    def __str__(self) -> str:
        """
        Method defining the string representation of the problem instance. Returning a
        representation of the board with all queen placements.
        """
        str_: str = " " * 5 + "   ".join(ASCII[:self.n])
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
    """
    Class representing the threats heuristic.
    """

    @staticmethod
    def evaluate(problem: NQueens) -> int:
        """
        Evaluation method returning evaluation score based on the number
        of threats per queen.

        Args:
            problem (Problem): The problem to evaluate.

        Returns:
            int: The evaluation score.
        """
        evaluation: int = 0
        n: int = problem.get_n()
        queens: np.ndarray = problem.queens

        # Looping through all queens
        for i in range(n):
            queen: Queen = queens[i]

            # Computing a list of all queens except the queen of the current loop.
            other_queens = np.concat([queens[:i], queens[i + 1:]])

            for other_queen in other_queens:
                if Queen.attacking(queen, other_queen):
                    evaluation += 1  # Queens are attacking each other

        return evaluation
