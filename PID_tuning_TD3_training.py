'''
LICENSE AGREEMENT

In relation to this Python file:

1. Copyright of this Python file is owned by the author: Mark Misin
2. This Python code can be freely used and distributed
3. The copyright label in this Python file such as
copyright=ax_main.text(x,y,'© Mark Misin Engineering',size=z)
that indicate that the Copyright is owned by Mark Misin MUST NOT be removed.

WARRANTY DISCLAIMER!

This Python file comes with absolutely NO WARRANTY! In no event can the author
of this Python file be held responsible for whatever happens in relation to this Python file.
For example, if there is a bug in the code and because of that a project, invention,
or anything else it was used for fails - the author is NOT RESPONSIBLE!

'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from matplotlib import pyplot as plt

TRAINING = True
MAX_ACTION = 10000
STATE_DIM = 8
ACTION_DIM = 6
SAVE_MODE = True

# ================= Suspension System Environment ===================
class SuspensionSysEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # Reset the system states and other values (at the beginning of each episode)

        # Reset the time and relevant integrals
        self.time = 0.0
        self.dt = 0.02
        self.max_time = 10.0
        self.integral_error_x = 0.0
        self.integral_error_theta = 0.0

        # Load the constants every episode
        self.m_b = 1200 # [kg]
        self.J_b = 1700 # [kg*m^2]
        self.m_w_1 = 60 # [kg]
        self.m_w_2 = 50 # [kg]
        self.k_b_1 = 25800 # [N/m]
        self.k_b_2 = 22200 # [N/m]
        self.c_b_1 = 2150 # [N*s/m]
        self.c_b_2 = 1850 # [N*s/m]
        self.l_1 = 1.4 # [m]
        self.l_2 = 0.9 # [m]
        self.k_w_1 = 180000 # [N/m]
        self.k_w_2 = 180000 # [N/m]
        self.c_w_1 = 2250 # [N*s/m]
        self.c_w_2 = 2250 # [N*s/m]
        self.v = 6 # [m/s]
        self.A_y = 0.03 # [m]
        self.A_x = 6 # [m]
        self.g = 9.81 # [m/s^2] # Stays constant

        # Implement speed bumps
        self.sb_r_1 = 0.2 # [m] - speed bump radius
        self.sb_r_2 = 0.3 # [m] - speed bump radius

        # Determine when the front wheel reaches the center of the bump.
        self.sb_tp_1 = 1 # [s]
        self.sb_tp_2 = 4 # [s]

        # Calculate the locations of the bumps' center.
        self.xp_1 = self.v * self.sb_tp_1
        self.xp_2 = self.v * self.sb_tp_2

        # From dFu, dMu -> df1, df2 - transformation matrix and its inverse
        self.T_matrix = np.array([[1, 1],[-self.l_1, self.l_2]])
        self.T_matrix_inv = np.linalg.inv(self.T_matrix)

        # Calculate the relevant initial variable for the simulation
        self.target_pos = np.array([-0.25, 0]) # x_b_ref, theta_b_ref
        self.state, self.f_1_eq, self.f_2_eq, self.y_1_array, self.y_dot_1_array, self.y_2_array, self.y_dot_2_array = self.initial_state(self.target_pos[0], self.target_pos[1])
        self.state = np.round(self.state, 8)
        self.f_1_eq = np.round(self.f_1_eq, 8)
        self.f_2_eq = np.round(self.f_2_eq, 8)

        return self.state

    def compute_reward(self, df1, df2):

        # Initiate the reward with 0
        reward = 0

        # Target/reference states
        x_b_ref = self.target_pos[0]
        theta_b_ref = self.target_pos[1]

        # Current states
        x_b = self.state[0]          # vertical displacement of the body
        theta_b = self.state[2]      # pitch angle of the body

        # Penalize large absolute control input forces
        if abs(df1) > 1500:
            reward -= 10

        if abs(df2) > 1500:
            reward -= 10


        if abs(x_b - x_b_ref) < 0.03:
            # Reward small vertical vibration
            reward += 1
        else:
            # Penalize large vertical vibration
            reward -= 1

        if abs(theta_b - theta_b_ref) < 0.03:
            # Reward small angular vibration
            reward += 5
        else:
            # Penalize large angular vibration
            reward -= 5

        return reward


    def step(self, pid_gains):
        # Initiate a 1-time-step simulation

        # Extract PID gains (actions) from the argument
        Kp_f, Ki_f, Kd_f, Kp_m, Ki_m, Kd_m = pid_gains

        # Reference values for x_b and theta_b
        x1_ref = self.target_pos[0]
        x3_ref = self.target_pos[1]

        # First 4 states from state vector: x_b, x_dot_b, theta_b, theta_dot_b
        x1 = self.state[0]
        x2 = self.state[1]
        x3 = self.state[2]
        x4 = self.state[3]

        # Errors: e_x_b, e_theta_b
        error_x = x1_ref - x1
        error_theta = x3_ref - x3

        # Error integration
        self.integral_error_x += error_x * self.dt
        self.integral_error_theta += error_theta * self.dt

        # Error derivatives
        error_x_dot = -x2
        error_theta_dot = -x4

        # Control input changes using PID controllers: force and moment inputs
        dF_u = Kp_f * error_x + Ki_f * self.integral_error_x + Kd_f * error_x_dot
        dM_u = Kp_m * error_theta + Ki_m * self.integral_error_theta + Kd_m * error_theta_dot

        # Convert them to force input changes applied to each wheel
        df_vec = self.T_matrix_inv @ np.array([[dF_u],[dM_u]])

        # Extract the inputs from the vector
        df1 = df_vec[0][0]
        df2 = df_vec[1][0]

        # Add the input changes to the equilibrium values to get absolute inputs
        f_1 = self.f_1_eq + df1
        f_2 = self.f_2_eq + df2

        # Get rid of rounding errors.
        f_1 = np.round(f_1, 8)
        f_2 = np.round(f_2, 8)

        # Integrate over shorter time-steps (20x per episode)
        for k in range(0, 20):
            self.state = self.update_state(self.state, f_1, f_2)
            self.time += 0.001
            self.time = np.round(self.time, 8)

        # Get rid of numerical errors
        self.state = np.round(self.state, 8)

        # Compute immediate reward
        reward = self.compute_reward(df1, df2)

        # Check if the simulation is at its end
        done = self.time >= self.max_time

        return self.state, reward, done


    def initial_state(self, x_b_desired, theta_b_desired):
        # Initiate the car state and other relevant variables

        # Subscripts: 1 - rear, 2 - front
        m_b = self.m_b
        J_b = self.J_b
        m_w_1 = self.m_w_1
        m_w_2 = self.m_w_2
        k_b_1 = self.k_b_1
        k_b_2 = self.k_b_2
        c_b_1 = self.c_b_1
        c_b_2 = self.c_b_2
        l_1 = self.l_1
        l_2 = self.l_2
        k_w_1 = self.k_w_1
        k_w_2 = self.k_w_2
        c_w_1 = self.c_w_1
        c_w_2 = self.c_w_2
        g = self.g
        v = self.v
        dt = 0.001

        sb_r_1 = self.sb_r_1
        sb_r_2 = self.sb_r_2

        sb_tp_1 = self.sb_tp_1
        sb_tp_2 = self.sb_tp_2

        xp_1 = self.xp_1
        xp_2 = self.xp_2

        ### COMPUTE CONTROLLED EQUILIBRIUM POINT ###

        A_eq_nat=(k_b_1+k_b_2)/m_b
        B_eq_nat=(k_b_1*l_1-k_b_2*l_2)/m_b
        C_eq_nat=k_b_1/m_b
        D_eq_nat=k_b_2/m_b
        E_eq_nat=(k_b_1*l_1-k_b_2*l_2)/J_b
        F_eq_nat=(k_b_1*l_1**2+k_b_2*l_2**2)/J_b
        G_eq_nat=k_b_1*l_1/J_b
        H_eq_nat=k_b_2*l_2/J_b
        I_eq_nat=k_b_1/m_w_1
        J_eq_nat=k_b_1*l_1/m_w_1
        K_eq_nat=(k_b_1+k_w_1)/m_w_1
        L_eq_nat=k_b_2/m_w_2
        M_eq_nat=k_b_2*l_2/m_w_2
        N_eq_nat=(k_b_2+k_w_2)/m_w_2

        A_sys_eq_ct = np.array([[C_eq_nat, D_eq_nat, 1/m_b, 1/m_b],
                        [-G_eq_nat, H_eq_nat, -l_1/J_b, l_2/J_b],
                        [-K_eq_nat, 0, -1/m_w_1, 0],
                        [0, -N_eq_nat, 0, -1/m_w_2]])

        b_sys_eq_ct = np.array([[g + A_eq_nat*x_b_desired - B_eq_nat*np.sin(theta_b_desired)],
                        [-E_eq_nat*x_b_desired + F_eq_nat*np.sin(theta_b_desired)],
                        [g - I_eq_nat*x_b_desired + J_eq_nat*np.sin(theta_b_desired)],
                        [g - L_eq_nat*x_b_desired - M_eq_nat*np.sin(theta_b_desired)]])

        x_eq_ct = np.linalg.inv(A_sys_eq_ct) @ b_sys_eq_ct

        f_1_eq = x_eq_ct[2][0]
        f_2_eq = x_eq_ct[3][0]

        state = np.array([x_b_desired, 0, theta_b_desired, 0, x_eq_ct[0][0], 0, x_eq_ct[1][0], 0])

        ### ROAD PROFILE ###
        t = np.arange(0, self.max_time + dt, dt)
        x_r = np.arange(0, v * t[-1] + dt * v, dt * v)
        y_r = np.zeros(len(x_r))

        y_1 = np.zeros(len(t))
        y_dot_1 = np.zeros(len(t))
        y_2 = np.zeros(len(t))
        y_dot_2 = np.zeros(len(t))

        tau1_pos = int((l_1 / v) / dt + 1)
        tau2_pos = int((l_2 / v) / dt)

        ### CREATE THE ROAD PROFILE (DISTURBANCE) ###
        for i in range(len(x_r)):
            if (x_r[i] > xp_1 - sb_r_1) and (x_r[i] < xp_1 + sb_r_1):
                y_r[i] = sb_r_1 * np.sin(np.arccos((xp_1 - x_r[i]) / sb_r_1))
                y_1[i + tau1_pos] = y_r[i]
                y_dot_1[i + tau1_pos] = (y_r[i] - y_r[i-1]) / dt
                y_2[i - tau2_pos] = y_r[i]
                y_dot_2[i - tau2_pos] = (y_r[i] - y_r[i-1]) / dt
            elif (x_r[i] > xp_2 - sb_r_2) and (x_r[i] < xp_2 + sb_r_2):
                y_r[i] = sb_r_2 * np.sin(np.arccos((xp_2 - x_r[i]) / sb_r_2))
                y_1[i + tau1_pos] = y_r[i]
                y_dot_1[i + tau1_pos] = (y_r[i] - y_r[i-1]) / dt
                y_2[i - tau2_pos] = y_r[i]
                y_dot_2[i - tau2_pos] = (y_r[i] - y_r[i-1]) / dt
            else:
                y_r[i] = 0

        return state, f_1_eq, f_2_eq, y_1, y_dot_1, y_2, y_dot_2


    def update_state(self, state, f_1, f_2):

        X1 = state[0]
        X2 = state[1]
        X3 = state[2]
        X4 = state[3]
        X5 = state[4]
        X6 = state[5]
        X7 = state[6]
        X8 = state[7]

        m_b = self.m_b
        J_b = self.J_b
        m_w_1 = self.m_w_1
        m_w_2 = self.m_w_2
        k_b_1 = self.k_b_1
        k_b_2 = self.k_b_2
        c_b_1 = self.c_b_1
        c_b_2 = self.c_b_2
        l_1 = self.l_1
        l_2 = self.l_2
        k_w_1 = self.k_w_1
        k_w_2 = self.k_w_2
        c_w_1 = self.c_w_1
        c_w_2 = self.c_w_2
        g = self.g
        v = self.v
        dt = 0.001
        A_y = self.A_y
        A_x = self.A_x

        ### ROAD PROFILE (DISTURBANCE) ###
        y_1_array = self.y_1_array
        y_dot_1_array = self.y_dot_1_array
        y_2_array = self.y_2_array
        y_dot_2_array = self.y_dot_2_array

        y_1 = y_1_array[int(self.time / dt)]
        y_dot_1 = y_dot_1_array[int(self.time / dt)]
        y_2 = y_2_array[int(self.time / dt)]
        y_dot_2 = y_dot_2_array[int(self.time / dt)]


        ### SIMULATION WITH ACTUATORS ###
        F_k_b_1 = -k_b_1*(X1-l_1*np.sin(X3)-X5)
        F_c_b_1 = -c_b_1*(X2-l_1*np.cos(X3)*X4-X6)
        F_k_b_2 = -k_b_2*(X1+l_2*np.sin(X3)-X7)
        F_c_b_2 = -c_b_2*(X2+l_2*np.cos(X3)*X4-X8)
        F_k_w_1 = -k_w_1*(X5-y_1)
        F_c_w_1 = -c_w_1*(X6-y_dot_1)
        F_k_w_2 = -k_w_2*(X7-y_2)
        F_c_w_2 = -c_w_2*(X8-y_dot_2)


        # Compute the time derivatives (with control inputs)
        x1_dot = X2
        x2_dot = (F_k_b_1 + F_c_b_1 + F_k_b_2 + F_c_b_2 - m_b*g + f_1 + f_2)/m_b
        x3_dot = X4
        x4_dot = ((F_k_b_2 + F_c_b_2 + f_2)*l_2 - (F_k_b_1 + F_c_b_1 + f_1)*l_1)*np.cos(X3)/J_b
        x5_dot = X6
        x6_dot = (- F_k_b_1 - F_c_b_1 + F_k_w_1 + F_c_w_1 - m_w_1*g - f_1)/m_w_1
        x7_dot = X8
        x8_dot = (- F_k_b_2 - F_c_b_2 + F_k_w_2 + F_c_w_2 - m_w_2*g - f_2)/m_w_2


        # Update the states
        X1 = X1 + dt * x1_dot
        X2 = X2 + dt * x2_dot
        X3 = X3 + dt * x3_dot
        X4 = X4 + dt * x4_dot
        X5 = X5 + dt * x5_dot
        X6 = X6 + dt * x6_dot
        X7 = X7 + dt * x7_dot
        X8 = X8 + dt * x8_dot

        state = np.array([X1, X2, X3, X4, X5, X6, X7, X8])

        return state

# ================= Replay Buffer ===================
class ReplayBuffer:

    # Define deque array with a fixed size
    def __init__(self, max_size=1_000_000):
        self.buffer = deque(maxlen=max_size)

    # Build up the buffer
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # Sample a random batch and convert to tensors here
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        # Convert to single NumPy arrays FIRST (fast)
        states      = np.stack([item[0] for item in batch])
        actions     = np.stack([item[1] for item in batch])
        rewards     = np.stack([item[2] for item in batch])[:, np.newaxis]  # shape (batch, 1)
        next_states = np.stack([item[3] for item in batch])
        dones       = np.stack([item[4] for item in batch])[:, np.newaxis]  # shape (batch, 1)

        # Single fast tensor conversion
        return (torch.from_numpy(states).float(),
                torch.from_numpy(actions).float(),
                torch.from_numpy(rewards).float(),
                torch.from_numpy(next_states).float(),
                torch.from_numpy(dones).float())




    def __len__(self):
        return len(self.buffer)

# ================= Neural Networks ===================

"""
Weights Initialization Function:
Weights are initialized using Xavier technique - it helps solve the vanishing and
exploding gradient problem in the training process.
Biases are initialized with zero values.
"""
def weights_init_(m):
    if isinstance(m, nn.Linear): # Makes sure that weights only initiated in fully connected linear networks.
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    # Purpose of init - set the initial state of the object when it is created.
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__() # Initializes the init function of a parent class (constructor [init] of nn.Module).
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, action_dim)
        self.max_action = max_action
        self.apply(weights_init_) # Apply Xavier initialization to all linear layers

    # Forward pass
    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.relu(self.l3(a))
        return (self.max_action / 2) * (torch.tanh(self.l4(a)) + 1)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Structure of the critic

        # Critic 1
        self.l1 = nn.Linear(state_dim + action_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 1)

        # Critic 2
        self.l5 = nn.Linear(state_dim + action_dim, 128)
        self.l6 = nn.Linear(128, 128)
        self.l7 = nn.Linear(128, 64)
        self.l8 = nn.Linear(64, 1)

        self.apply(weights_init_)  # Apply Xavier initialization to all linear layers

    # Forward pass
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1) # Join together states and actions.

        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = torch.relu(self.l3(q1))
        q1 = self.l4(q1)

        q2 = torch.relu(self.l5(sa))
        q2 = torch.relu(self.l6(q2))
        q2 = torch.relu(self.l7(q2))
        q2 = self.l8(q2)

        return q1, q2

    def Q1(self, state, action): # This is used to update actor weights.
        sa = torch.cat([state, action], dim=1)
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = torch.relu(self.l3(q1))
        q1 = self.l4(q1)
        return q1

# ================= TD3 Agent ===================
class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise_rel=0.02,
        noise_clip_rel=0.05,
        policy_freq=2,
    ):

        # CPU vs GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, max_action) # Create actor instance
        self.actor_target = Actor(state_dim, action_dim, max_action) # Create target actor instance
        self.actor_target.load_state_dict(self.actor.state_dict()) # Give the same initial weights to target actor like the main actor.
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4) # Popular learning algorithm (Adam) to update weights and biases.
        # self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.99) # Decreases learning rate (lr) in time to improve convergence

        self.critic = Critic(state_dim, action_dim) # Create critic instance (includes two critics)
        self.critic_target = Critic(state_dim, action_dim) # Create target critic instance (includes two target critics)
        self.critic_target.load_state_dict(self.critic.state_dict()) # Give the same initial weights to target critic like the main critic.
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4) # Popular learning algorithm (Adam) to update weights and biases.
        # self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.99) # Decreases learning rate (lr) in time to improve convergence

        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise_rel * max_action
        self.noise_clip = noise_clip_rel * max_action
        self.policy_freq = policy_freq
        self.total_it = 0


    # Save the training state
    def save(self, filename):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_it': self.total_it,
        }
        torch.save(checkpoint, filename + ".pth")


    # Load the training state
    def load(self, filename):
        checkpoint = torch.load(filename + ".pth", map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_it = checkpoint.get('total_it', 0)

    def select_action(self, state):
        # Reshape the state vector to make it compatible with PyTorch.
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        # Returns action vector based on the input states.
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):

        # Early exit if not enough data in the buffer
        if len(replay_buffer) < batch_size:
            return

        # Increment total training iteration count
        self.total_it += 1

        # Sample a batch of transition from the replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Assign the networks to the right device:
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Create Gaussian noise for actions from target actor (policy noise: standard deviation)
        noise = (
            torch.randn_like(actions) * self.policy_noise
        ).clamp(-self.noise_clip, self.noise_clip)

        # Generate actions from target actor
        next_actions = (
            self.actor_target(next_states) + noise
        ).clamp(0.0, self.max_action)  # clamp to positive range!

        # Generate target Q-value from target critics: take minimum Q value, discount it, add immediate rewards to it
        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)

            target_Q = rewards + (1 - dones) * self.discount * target_Q


        # Generate Q values from current critics
        current_Q1, current_Q2 = self.critic(states, actions)

        # Compute the Mean Squared Errors (MSE) of both Q errors, and sum them up: 1/N * SUM((target Q - current Q_i)^2), where N is amount of batch transitions
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        self.critic_optimizer.zero_grad() # Resets gradients to zero
        critic_loss.backward() # Computes gradients of the loss w.r.t. all critic parameters via backpropagation.

        # Clip critic gradients by global norm - for more stable learning
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

        self.critic_optimizer.step() # The optimizer (typically Adam) reads the gradient values and updates weights (gradient descent)
        # self.critic_scheduler.step() # Updates the learning rate according to your scheduler (e.g., reduces LR every X steps)

        # Critics update every batch, but actor updates only every 2nd batch to let Q-values stabilize first.
        if self.total_it % self.policy_freq == 0:

            # Instead of taking gradient ascent of critic's Q-value, we take gradient descent of negative of that - mathematically the same thing
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

            # Same procedeure like with critics in updating weights
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            # Clip actor gradients by global norm - for more stable learning
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)

            self.actor_optimizer.step()
            # self.actor_scheduler.step()

            # Soft updates of target network parameters (zip is for correctly pairing the regular and target network layers) - PyTorch methods are used for this.
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )


# ================= Main Training Loop ===================
def main():

    # Create environment instance
    env = SuspensionSysEnv()

    # Create TD3 agent instance
    agent = TD3(STATE_DIM, ACTION_DIM, MAX_ACTION)

    print(f"Using device: {agent.device}")
    if agent.device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()


    # Load checkpoint with model, optimizer and metadata state
    try:
        agent.load("td3_model_checkpoint")
        print("Loaded saved model, optimizer states, and training metadata.")
    except FileNotFoundError:
        print("No saved model weights found, training from scratch.")

    # Create replay buffer instance
    replay_buffer = ReplayBuffer()

    episodes = 500
    max_steps = 501
    batch_size = 100

    # Choose exploration noise relative to max action
    EXPL_NOISE_REL = 0.05  # 5% of range

    # Run x amount of episodes in the training process
    for ep in range(episodes):
        # Initiate the an episode

        # Get the initial state
        state = env.reset()

        # Total accumulated reward per episode
        episodic_reward = 0

        # Number of samples before training starts
        learning_starts = 100

        for step in range(max_steps):
            # Initiate 1-time step simulation

            action = agent.select_action(state)

            exploration_noise = np.random.normal(
                0.0,
                EXPL_NOISE_REL * MAX_ACTION,
                size=ACTION_DIM
            )

            action = (action + exploration_noise).clip(0.0, MAX_ACTION)

            # Apply action to envioronment, get new state, reward, and if current state is final
            next_state, reward, done = env.step(action)

            # The replay buffer stores past experiences for sampling during training.
            replay_buffer.add(state, action, reward, next_state, float(done))

            # Assign next state variable to current state for the next episode
            state = next_state

            # Accumulate reward over episode
            episodic_reward += reward

            # Initiate training when enough experiences in the buffer
            if len(replay_buffer) > learning_starts:
                agent.train(replay_buffer, batch_size)

            # The end of simulation is reached - start a new episode
            if done:
                break

        print(f"Episode {ep+1} Reward: {episodic_reward:.2f}")
        learned_pid = agent.select_action(state)
        print(f"Learned PID gains (positive): Kp_f={learned_pid[0]:.3f}, Ki_f={learned_pid[1]:.3f}, Kd_f={learned_pid[2]:.3f}, Kp_m={learned_pid[3]:.3f}, Ki_m={learned_pid[4]:.3f}, Kd_m={learned_pid[5]:.3f}")

        # Save model weights after each episode
        if SAVE_MODE == True:
            agent.save("td3_model_checkpoint")


def evaluate_and_extract_pid_gains(env, agent, max_steps=501):

    # Obtain initial states
    state = env.reset()
    pid_gains_sequence = []
    total_reward = 0.0
    done = False

    for step in range(max_steps):

        # Select deterministic action (no noise)
        action = agent.select_action(state)

        # Create array for the PID constants throughout the simulation
        pid_gains_sequence.append(action)

        # Obtain next states, reward, done
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward

        # Exit the loop once the end simulation is reached
        if done:
            break

    # Compute average PID gains
    pid_gains_sequence = np.array(pid_gains_sequence)
    mean_pid_gains = np.mean(pid_gains_sequence, axis=0)
    median_pid_gains = np.median(pid_gains_sequence, axis=0)


    print(f"Evaluation total reward: {total_reward:.2f}")
    print(f"Learned mean PID gains: "
        f"Kp_f={mean_pid_gains[0]:.3f}, Ki_f={mean_pid_gains[1]:.3f}, Kd_f={mean_pid_gains[2]:.3f}, "
        f"Kp_m={mean_pid_gains[3]:.3f}, Ki_m={mean_pid_gains[4]:.3f}, Kd_m={mean_pid_gains[5]:.3f}")

    print(f"Learned Median PID gains: "
        f"Kp_f={median_pid_gains[0]:.3f}, Ki_f={median_pid_gains[1]:.3f}, Kd_f={median_pid_gains[2]:.3f}, "
        f"Kp_m={median_pid_gains[3]:.3f}, Ki_m={median_pid_gains[4]:.3f}, Kd_m={median_pid_gains[5]:.3f}")

    # Plot all 6 PID gains separately as a function of simulation time
    gain_names = ['Kp_f', 'Ki_f', 'Kd_f', 'Kp_m', 'Ki_m', 'Kd_m']
    plt.figure(figsize=(15, 12))
    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.plot(pid_gains_sequence[:, i], label=f'{gain_names[i]} over time')
        plt.xlabel('Step')
        plt.ylabel(f'{gain_names[i]} Value')
        plt.title(f'{gain_names[i]} over Episode')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if TRAINING == True:
        # Start the training process
        main()
    else:
        # Evaluate the trained PID constants

        # Create the environment instance
        env = SuspensionSysEnv()

        # Create the agent instance
        agent = TD3(STATE_DIM, ACTION_DIM, MAX_ACTION)

        # Load the trained weights
        agent.load("td3_model_checkpoint")  # your saved .pth filename without extension

        # Evaluate and extract PID gains
        evaluate_and_extract_pid_gains(env, agent)
