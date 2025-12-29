import numpy as np
import matplotlib.pyplot as plt
import random

# Create the Q-table and other parameters
Q_table = np.zeros([16,4])

discount = 0.9
learning = 0.8
reward = 0

### Define movement ###

# Starting position
n_init = 3
m_init = 0

n_new = n_init
m_new = m_init

n = n_init
m = m_init

# Start and goal positions
start = (n_init, m_init)
goal = (3, 3)


# Define action values
down = 'down' # 0
right = 'right' # 1
up = 'up' # 2
left = 'left' # 3

action_array = np.array([down, right, up, left])
action_id = 0

### START LEARNING ###
for _ in range(10000):
    while True:
        action = random.choice(action_array)

        if action == 'down' and n < 3:
            if not (n == 1 and m == 2):
                n_new = n + 1
                action_id = 0
                break

        if action == 'right' and m < 3:
            if not (n == 2 and m == 1):
                if not (n == 3 and m == 1):
                    m_new = m + 1
                    action_id = 1
                    break

        if action == 'up' and n > 0:
            n_new = n - 1
            action_id = 2
            break

        if action == 'left' and m > 0:
            if not (n == 2 and m == 3):
                if not (n == 3 and m == 3):
                    m_new = m - 1
                    action_id = 3
                    break


    if n_new == goal[0] and m_new == goal[1]:
        reward = 10
    else:
        reward = -1

    Q_table[4 * n + m][action_id] = Q_table[4 * n + m][action_id] + learning * (reward + discount * Q_table[4 * n_new + m_new][:].max() - Q_table[4 * n + m][action_id])

    n = n_new
    m = m_new

print("")
print("Q table:")
print(Q_table)


### GENERATE PATH ###
rows, cols = 4, 4

# Walls coordinates (row, col)
walls = [(2, 2)]

current_loc = start
coordinate_arrays = np.array([current_loc])

while current_loc != goal:

    current_state = Q_table[4 * current_loc[0] + current_loc[1]][:]
    best_action = current_state.argmax()

    if best_action == 0:
        tempV = current_loc[0]
        tempH = current_loc[1]
        tempV += 1
        current_loc = (tempV, tempH)
    elif best_action == 1:
        tempV = current_loc[0]
        tempH = current_loc[1]
        tempH += 1
        current_loc = (tempV, tempH)
    elif best_action == 2:
        tempV = current_loc[0]
        tempH = current_loc[1]
        tempV -= 1
        current_loc = (tempV, tempH)
    elif best_action == 3:
        tempV = current_loc[0]
        tempH = current_loc[1]
        tempH -= 1
        current_loc = (tempV, tempH)
    else:
        print("If you see this message, then something is wrong with the code!")

    coordinate_arrays = np.vstack((coordinate_arrays, [current_loc]))

print("")
print("Start -> Finish coordinates:")
print(coordinate_arrays)

### PLOTTING ###

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('#bbbbbb')  # medium gray

ax.set_xlim(0, cols)
ax.set_ylim(0, rows)
ax.set_xticks(range(cols+1))
ax.set_yticks(range(rows+1))
ax.set_xticklabels(range(cols+1), color='black', fontsize=20)
ax.set_yticklabels(range(rows+1), color='black', fontsize=20)
ax.set_facecolor('#dddddd')  # light gray
ax.grid(True)

# Invert y-axis so 0 is top and increases downwards
ax.invert_yaxis()

# Plot walls: adjust y coordinate since y=0 is top now
for (r, c) in walls:
    ax.add_patch(plt.Rectangle((c, r), 1, 2, color='black'))

# Plot start and goal: shift text by 0.5 to center within cells, adjust y for inverted axis
ax.text(start[1] + 0.25, start[0] + 0.25, 'Start', ha='center', va='center', color='green', fontsize=20, weight='bold')
ax.text(goal[1] + 0.25, goal[0] + 0.25, 'Goal', ha='center', va='center', color='blue', fontsize=20, weight='bold')

# Extract x (column) and y (row) coordinates from your array
x_coords = coordinate_arrays[:, 1] + 0.5  # column + 0.5 to center in cells
y_coords = coordinate_arrays[:, 0] + 0.5  # row + 0.5 to center in cells

# Plot red solid line path
ax.plot(x_coords, y_coords, 'r-', linewidth=3)

plt.title('Q-learning path in Grid World', fontsize=20, weight='bold')
plt.show()


































########
