# Mark Chmielecki - Final Exam Problem 3
# This is similar to our project 3 A* implementation, so I used my code from that project as a starting point

import heapq
def manhattan_heuristic(current, goal):

    x = abs(current[0] - goal[0])    # Calculate absolute difference in x coordinates 
    y = abs(current[1] - goal[1])    # Calculate absolute difference in y coordinates
    
    return x + y                     # Return sum of absolute differences in x and y coordinates, which is the Manhattan distance between points a and b
    
def find_path(grid, start, goal):
    
    # Initialize the priority queue (frontier)
    # Format: (f_score, g_score, position)

    frontier = []
    # Push the starting node onto the frontier

    heapq.heappush(frontier, (0, 0, start))  
    visited = set()
    while frontier:
        f_score, g_score, position = heapq.heappop(frontier)    # Pop the node with the lowest f_score from the frontier
        
        if position in visited:                                 # Skip the node if it has already been visited
            continue
        
        if position == goal:                                    # Check if the current node is the goal (Return g_score)                                     
            return g_score

        visited.add(position)

        # Neighbors: Up, Down, Left, Right
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = position[0] + dx, position[1] + dy
            # Boundary and obstacle check
            if (0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and
                grid[nx][ny] == 0 and (nx, ny) not in visited):
                new_g = g_score + 1                                     # Calculate new_g and new_f
                h = manhattan_heuristic((nx, ny), goal)                 # Calculate cost to reach the goal from neighbor, using Manhattan distance as the heuristic
                f_score = (new_g + h) + (h * 0.001)     
                heapq.heappush(frontier, (f_score, new_g, (nx, ny)))    # Push the neighbor to the frontier                 
            
    return -1 # Return -1 if no path exists# No path found

# --- Example Usage ---
if __name__ == "__main__":
    # 0 = Path, 1 = Wall
    example_grid = [
        [0, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 1, 0]
    ]
    start_node = (0, 0)
    goal_node = (2, 0)
    result = find_path(example_grid, start_node, goal_node)
    print(f"Total path cost: {result}")