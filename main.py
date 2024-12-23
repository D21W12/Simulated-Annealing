from simulated_annealing import SimulatedAnnealing, GeometricCooling, LinearCooling

from problems.n_queens import NQueens, ThreatsHeuristic


if __name__ == '__main__':
    n_queens = NQueens(20)
    h = ThreatsHeuristic()

    sa = SimulatedAnnealing(
        problem=n_queens,
        schedule=LinearCooling(0.9999),
        heuristic=h,
        starting_temp=10,
        minimizing=True
    )

    found_solution = sa.run(iterations=100000, print_iterations=False)

    print("Final solution:")
    print(found_solution)

    evaluation = h.evaluate(found_solution)
    print(f"Final evaluation: {evaluation}")
