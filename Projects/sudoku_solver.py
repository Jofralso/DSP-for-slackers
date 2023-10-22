def print_sudoku(grid):
    for row in grid:
        print(' '.join(map(str, row)))

def is_valid_configuration(grid, row, col, num):
    # Check row and column
    for i in range(9):
        if grid[row][i] == num or grid[i][col] == num:
            return False

    # Check 3x3 box
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if grid[i][j] == num:
                return False

    return True

def identify_missing_locations(grid):
    missing_locations = []
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                missing_locations.append((i, j))
    return missing_locations

def solve_sudoku(grid, solutions):
    missing_locations = identify_missing_locations(grid)
    if not missing_locations:
        solutions.append([row[:] for row in grid])
        return

    row, col = missing_locations[0]

    for num in range(1, 10):
        if is_valid_configuration(grid, row, col, num):
            grid[row][col] = num
            solve_sudoku(grid, solutions)
            grid[row][col] = 0

def main():
    # Take user input for Sudoku matrix
    print("Enter the Sudoku puzzle (9x9 grid, use 0 for empty cells):")
    # Sudoku grid representation
    puzzle = [
    [0, 8, 1, 4, 0, 3, 0, 0, 7],
    [0, 0, 0, 5, 0, 2, 0, 6, 0],
    [0, 6, 4, 0, 8, 0, 0, 2, 0],
    [0, 1, 0, 0, 5, 8, 0, 0, 0],
    [6, 0, 3, 0, 2, 0, 0, 0, 0],
    [4, 0, 0, 1, 7, 0, 0, 0, 0],
    [9, 0, 0, 0, 4, 0, 2, 7, 0],
    [0, 0, 2, 0, 0, 7, 9, 8, 0],
    [0, 0, 0, 2, 0, 5, 4, 0, 0]
]

    #for i in range(9):
     #   row = list(map(int, input().split()))
       # puzzle.append(row)

    # Validate the initial configuration
    def is_valid_configuration(grid):
    # Check rows and columns
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    continue
                for k in range(9):
                    if k != j and grid[i][k] == grid[i][j]:
                        return False
                    if k != i and grid[k][j] == grid[i][j]:
                        return False
    
    # Check 3x3 subgrids
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                subgrid = []
                for x in range(i, i + 3):
                    for y in range(j, j + 3):
                        if grid[x][y] == 0:
                            continue
                        if grid[x][y] in subgrid:
                            return False
                        subgrid.append(grid[x][y])
        return True

    # Solve the Sudoku
    solutions = []
    solve_sudoku(puzzle, solutions)

    # Print solutions
    if len(solutions) == 0:
        print("No solution exists.")
    else:
        print("Sudoku Solution(s):")
        for i, solution in enumerate(solutions):
            print(f"Solution {i + 1}:")
            print_sudoku(solution)
            print()

if __name__ == "__main__":
    main()
