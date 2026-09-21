"""
Project #3: The Smart Navigator (A* vs. BFS)
Name: Mark Chmielecki
Date: September 21, 2026
"""

import heapq
from collections import deque
from tracemalloc import start

def get_neighbors(node, grid):
    """Returns valid North, South, East, West neighbors (0 = path, 1 = wall)."""
    neighbors = []
    rows, cols = len(grid), len(grid[0])
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = node[0] + dr, node[1] + dc
        if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0:
            neighbors.append((r, c))
    return neighbors

def manhattan_distance(a, b):
    """
    TASK: Implement Manhattan Distance h(n)
    Formula: |x1 - x2| + |y1 - y2|
    """
    x = abs(a[0] - b[0])    # Calculate absolute difference in x coordinates 
    y = abs(a[1] - b[1])    # Calculate absolute difference in y coordinates

    return x + y            # Return sum of absolute differences in x and y coordinates, which is the Manhattan distance between points a and b

def breadth_first_search(grid, start, goal):
    """
    TASK: Implement Breadth-First Search.
    - Use 'deque' for the frontier.
    - Return the total count of nodes visited.
    """
    nodes_visited = 0
    frontier = deque([start])                                   # Initialize the frontier with the start node
    visited_nodes = {start}                                     # Initialize the set of visited nodes

    while frontier:
        current = frontier.popleft()                            # Remove the current note from the frontier in FIFO order
        nodes_visited += 1                                      # Increment the count of nodes visited

        if current == goal:                                     # If the current node is the goal, break out of the loop
            break

        for neighbor in get_neighbors(current, grid):           # Get the neighbors of the current node, only zeroes (paths) are valid neighbors, 1's are walls
            if neighbor not in visited_nodes:                   # If the neighbor has been visited, skip it, otherwise add it to the visited nodes and add it to the frontier
                visited_nodes.add(neighbor)                     # Add the neighbor to the visited nodes set
                frontier.append(neighbor)                       # Add zero neighbors to the frontier for future exploration

    return nodes_visited                                        # Return the total count of nodes visited

def a_star_search(grid, start, goal):
    """
    TASK: Implement A* Search.
    - Use 'heapq' for the priority queue.
    - Use the tie-breaker: priority = (g + h) + (h * 0.001)
    - Return the total count of nodes visited.
    """
    nodes_visited = 0   
    frontier = []                                                               # Initialize frontier
    heapq.heappush(frontier, (0, 0, start))                                     # Pushes priority = 0, g/cost = 0, and start node onto the frontier heap
    #print (f"frontier {frontier}")                                           # Print the start and goal nodes for debugging
    visited_nodes = {start}                                                     # Initialize the set of visited nodes with start (0,0)

    while frontier:
        print (f"frontier {frontier}") 
        current_priority, current_g, current_node = heapq.heappop(frontier)     # Remove current node from frontier in priority order 
        nodes_visited += 1                                                              # Increment the count of nodes visited

        if current_node == goal:                                                # If the current node is the goal, break out of the loop
            break

        for neighbors in get_neighbors(current_node, grid):                      # Get the neighbors of the current node, only zeroes (paths) are valid neighbors, 1's are walls
            if neighbors not in visited_nodes:                                   # If the neighbor has been visited, skip it, otherwise add it to the visited nodes and add it to the frontier
                visited_nodes.add(neighbors)                                     # Add the neighbor to the visited nodes set
                g = current_g + 1                                               # 
                #print (f"the g is {g}")                                               # Increment the cost to reach the neighbor (g(n) = g(current) + 1)
                h = manhattan_distance(neighbors, goal)                          # Calculate the heuristic cost to reach the goal from the neighbor (h(n) = Manhattan distance)
                priority = (g + h) + (h * 0.001)                                # Calculate the priority of the neighbor using the tie-breaker formula
                heapq.heappush(frontier, (priority, g, neighbors))               # Add the neighbor to the frontier with its priority, g value, and coordinates

    return nodes_visited

def main():
    start = (0, 0)
    # ==========================================================
    # TEST MAZES
    # ==========================================================

    # 1. WINDING MAZE (10x10)
    m1 = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    # 2. CHECKERBOARD (10x10)
    m2 = [[(r+c)%2 if (0<r<9 and 0<c<9) else 0 for c in range(10)] for r in range(10)]

    # 3. THE SPIRAL (12x12)
    m3 = [[0]*12 for _ in range(12)]
    for i in range(2, 10):
        m3[2][i] = m3[i][10] = m3[10][11-i] = m3[11-i][2] = 1
    m3[3][2] = 0 

# ==========================================================
# TEST EXECUTION
# ==========================================================
    
    print(f"1. WINDING MAZE - BFS: {breadth_first_search(m1, start, (9,9))} | A*: {a_star_search(m1, start, (9,9))}")
   # print(f"2. CHECKERBOARD - BFS: {breadth_first_search(m2, start, (9,9))} | A*: {a_star_search(m2, start, (9,9))}")
   # print(f"3. THE SPIRAL   - BFS: {breadth_first_search(m3, start, (6,6))} | A*: {a_star_search(m3, start, (6,6))}")

if __name__ == "__main__":
    main()


