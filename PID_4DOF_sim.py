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
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import PID_tuning_TD3_training as PID_TD3_train


### USER INPUTS ###

ADAPTIVE_TUNING = True
CONTROLLER_APPLIED = True

# Subscripts: 1 - rear, 2 - front
m_b = 1200 # [kg]
J_b = 1850 # [kg*m^2]
m_w_1 = 60 # [kg]
m_w_2 = 50 # [kg]
k_b_1 = 25800 # [N/m]
k_b_2 = 22200 # [N/m]
c_b_1 = 2150 # [N*s/m]
c_b_2 = 1850 # [N*s/m]
l_1 = 1.4 # [m]
l_2 = 0.9 # [m]
k_w_1 = 180000 # [N/m]
k_w_2 = 180000 # [N/m]
c_w_1 = 2250 # [N*s/m]
c_w_2 = 2250 # [N*s/m]
g = 9.81 # [m/s^2]
v = 6 # [m/s]
A_y = 0.03 # [m]
A_x = 6 # [m]

if CONTROLLER_APPLIED == False:
    ADAPTIVE_TUNING = False

if ADAPTIVE_TUNING == False:

    Kp_f=8367.392
    Ki_f=9713.669
    Kd_f=6543.172
    Kp_m=407.543
    Ki_m=8468.434
    Kd_m=8319.826


### CREATE TIME AND DISTANCE ARRAYS ###
dt=0.001
t_end = 10 # [s]
t = np.arange(0,t_end+dt,dt)

### Implement speed bumps ###
sb_r_1 = 0.2 # [m] - speed bump radius
sb_r_2 = 0.3 # [m] - speed bump radius

# Determine when the front wheel reaches the center of the bump.
sb_tp_1 = 1 # [s]
sb_tp_2 = 5 # [s]

# Must be at least at 1 s. The second bump must be after it
if sb_tp_1 < 1:
    sb_tp_1 = 1

# Must be at least 1 s before the end
if sb_tp_2 > t_end - 1:
    sb_tp_2 = t_end - 1

# Calculate the locations of the bumps' center.
xp_1 = v * sb_tp_1
xp_2 = v * sb_tp_2

#####################

### LOAD THE AGENT ###

if ADAPTIVE_TUNING == True:
    state_dim = PID_TD3_train.STATE_DIM
    action_dim = PID_TD3_train.ACTION_DIM
    max_action = PID_TD3_train.MAX_ACTION

    # Get the agent
    agent_TD3 = PID_TD3_train.TD3(state_dim, action_dim, max_action)

    # Load the trained weights
    agent_TD3.load("result/td3_model_checkpoint")  # your saved .pth filename without extension



### COMPUTE NATURAL EQUILIBRIUM POINT ###
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


### CREATE THE ROAD PROFILE (DISTURBANCE) ###
x_r = np.arange(0, v*t[-1] + dt*v, dt*v)
y_r = np.zeros(len(x_r))

y_1 = np.zeros(len(t))
y_dot_1 = np.zeros(len(t))
y_2 = np.zeros(len(t))
y_dot_2 = np.zeros(len(t))

tau1_pos = int((l_1 / v) / dt + 1)
tau2_pos = int((l_2 / v) / dt)

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



### COMPUTE CONTROLLED EQUILIBRIUM POINT ###
x_b_desired = -0.25 # [m]
theta_b_desired = 0 # [rad]

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


### DEFINE INITIAL CONDITIONS AND ARRAYS ###

# Initial conditions

x10=x_b_desired # [m]
x20=0 # [m/s]
x30=theta_b_desired # [rad]
x40=0 # [rad/s]
x50=x_eq_ct[0][0] # [m]
x60=0 # [m/s]
x70=x_eq_ct[1][0] # [m]
x80=0 # [m/s]

# Simulate when no PID is applied

# Define 0 arrays for all the states (no controller)
X1_nc=np.zeros(len(t)) # x_b
X2_nc=np.zeros(len(t)) # x_b_dot
X3_nc=np.zeros(len(t)) # theta_b
X4_nc=np.zeros(len(t)) # theta_dot_b
X5_nc=np.zeros(len(t)) # x_w_1
X6_nc=np.zeros(len(t)) # x_w_1_dot
X7_nc=np.zeros(len(t)) # x_w_2
X8_nc=np.zeros(len(t)) # x_w_2_dot

# Fill in the first elements of state arrays with initial conditions (no controller)
X1_nc[0]=x10
X2_nc[0]=x20
X3_nc[0]=x30
X4_nc[0]=x40
X5_nc[0]=x50
X6_nc[0]=x60
X7_nc[0]=x70
X8_nc[0]=x80

# Define 0 arrays for all the states (controller)
X1=np.zeros(len(t)) # x_b
X2=np.zeros(len(t)) # x_b_dot
X3=np.zeros(len(t)) # theta_b
X4=np.zeros(len(t)) # theta_dot_b
X5=np.zeros(len(t)) # x_w_1 (back)
X6=np.zeros(len(t)) # x_w_1_dot (back)
X7=np.zeros(len(t)) # x_w_2 (front)
X8=np.zeros(len(t)) # x_w_2_dot (front)

# Fill in the first elements of state arrays with initial conditions (controller)
X1[0]=x10
X2[0]=x20
X3[0]=x30
X4[0]=x40
X5[0]=x50
X6[0]=x60
X7[0]=x70
X8[0]=x80

X2_dot_nc_array=np.zeros(len(t)) # x_b_dot_dot
X4_dot_nc_array=np.zeros(len(t)) # theta_b_dot_dot

X2_dot_array=np.zeros(len(t)) # x_b_dot_dot
X4_dot_array=np.zeros(len(t)) # theta_b_dot_dot

# Initialize the inputs
f_1 = f_1_eq
f_2 = f_2_eq

f_1_array=np.zeros(len(t))
f_2_array=np.zeros(len(t))

f_1_array[0] = f_1
f_2_array[0] = f_2

### START EULER SIMULATION ###
k_input=20
k_input_delta = k_input

error_x_prev = 0
error_theta_prev = 0
error_x_int = 0
error_theta_int = 0

T_matrix = np.array([[1, 1],[-l_1, l_2]])

if CONTROLLER_APPLIED == True:
    mode = 1
else:
    mode = 0

for i in range(len(t)):
    if i > 0:

        ### SIMULATION WITH NO ACTUATORS ###
        F_k_b_1 = -k_b_1*(X1_nc[i-1]-l_1*np.sin(X3_nc[i-1])-X5_nc[i-1])
        F_c_b_1 = -c_b_1*(X2_nc[i-1]-l_1*np.cos(X3_nc[i-1])*X4_nc[i-1]-X6_nc[i-1])
        F_k_b_2 = -k_b_2*(X1_nc[i-1]+l_2*np.sin(X3_nc[i-1])-X7_nc[i-1])
        F_c_b_2 = -c_b_2*(X2_nc[i-1]+l_2*np.cos(X3_nc[i-1])*X4_nc[i-1]-X8_nc[i-1])
        F_k_w_1 = -k_w_1*(X5_nc[i-1]-y_1[i-1])
        F_c_w_1 = -c_w_1*(X6_nc[i-1]-y_dot_1[i-1])
        F_k_w_2 = -k_w_2*(X7_nc[i-1]-y_2[i-1])
        F_c_w_2 = -c_w_2*(X8_nc[i-1]-y_dot_2[i-1])

        # Compute the time derivatives (without control inputs)
        x1_dot_nc = X2_nc[i-1]
        x2_dot_nc = (F_k_b_1 + F_c_b_1 + F_k_b_2 + F_c_b_2 - m_b*g + f_1_eq + f_2_eq)/m_b
        x3_dot_nc = X4_nc[i-1]
        x4_dot_nc = ((F_k_b_2 + F_c_b_2 + f_2_eq)*l_2 - (F_k_b_1 + F_c_b_1 + f_1_eq)*l_1)*np.cos(X3_nc[i-1])/J_b
        x5_dot_nc = X6_nc[i-1]
        x6_dot_nc = (- F_k_b_1 - F_c_b_1 + F_k_w_1 + F_c_w_1 - m_w_1*g - f_1_eq)/m_w_1
        x7_dot_nc = X8_nc[i-1]
        x8_dot_nc = (- F_k_b_2 - F_c_b_2 + F_k_w_2 + F_c_w_2 - m_w_2*g - f_2_eq)/m_w_2

        # Update the states - no controller
        X1_nc[i] = X1_nc[i-1]+dt*x1_dot_nc
        X2_nc[i] = X2_nc[i-1]+dt*x2_dot_nc
        X3_nc[i] = X3_nc[i-1]+dt*x3_dot_nc
        X4_nc[i] = X4_nc[i-1]+dt*x4_dot_nc
        X5_nc[i] = X5_nc[i-1]+dt*x5_dot_nc
        X6_nc[i] = X6_nc[i-1]+dt*x6_dot_nc
        X7_nc[i] = X7_nc[i-1]+dt*x7_dot_nc
        X8_nc[i] = X8_nc[i-1]+dt*x8_dot_nc

        # Update acceleration - no controller
        X2_dot_nc_array[i] = x2_dot_nc
        X4_dot_nc_array[i] = x4_dot_nc

        # Compute the PID error components for x and theta
        error_x = x_b_desired - X1[i-1]
        error_x_int = error_x_int + error_x * dt
        error_x_dot = (error_x - error_x_prev) / dt
        error_x_prev = error_x

        error_theta = theta_b_desired - X3[i-1]
        error_theta_int = error_theta_int + error_theta * dt
        error_theta_dot = (error_theta - error_theta_prev) / dt
        error_theta_prev = error_theta

        ### SIMULATION WITH ACTUATORS ###
        F_k_b_1 = -k_b_1*(X1[i-1]-l_1*np.sin(X3[i-1])-X5[i-1])
        F_c_b_1 = -c_b_1*(X2[i-1]-l_1*np.cos(X3[i-1])*X4[i-1]-X6[i-1])
        F_k_b_2 = -k_b_2*(X1[i-1]+l_2*np.sin(X3[i-1])-X7[i-1])
        F_c_b_2 = -c_b_2*(X2[i-1]+l_2*np.cos(X3[i-1])*X4[i-1]-X8[i-1])
        F_k_w_1 = -k_w_1*(X5[i-1]-y_1[i-1])
        F_c_w_1 = -c_w_1*(X6[i-1]-y_dot_1[i-1])
        F_k_w_2 = -k_w_2*(X7[i-1]-y_2[i-1])
        F_c_w_2 = -c_w_2*(X8[i-1]-y_dot_2[i-1])

        # Decrease the frequency at which force actutors change their values
        if i == k_input:

            if ADAPTIVE_TUNING == True:

                state = np.array([X1[i-1], X2[i-1], X3[i-1], X4[i-1], X5[i-1], X6[i-1], X7[i-1], X8[i-1]])
                action = agent_TD3.select_action(state)

                Kp_f = action[0]
                Ki_f = action[1]
                Kd_f = action[2]
                Kp_m = action[3]
                Ki_m = action[4]
                Kd_m = action[5]

            dF_u = Kp_f * error_x + Ki_f * error_x_int + Kd_f * error_x_dot
            dM_u = Kp_m * error_theta + Ki_m * error_theta_int + Kd_m * error_theta_dot

            df_vec = np.linalg.inv(T_matrix) @ np.array([[dF_u],[dM_u]])

            df1 = df_vec[0][0]
            df2 = df_vec[1][0]

            f_1 = f_1_eq + df1 * mode
            f_2 = f_2_eq + df2 * mode

            k_input=k_input+k_input_delta

        f_1_array[i] = f_1
        f_2_array[i] = f_2

        # Compute the time derivatives (with control inputs)
        x1_dot = X2[i-1]
        x2_dot = (F_k_b_1 + F_c_b_1 + F_k_b_2 + F_c_b_2 - m_b*g + f_1 + f_2)/m_b
        x3_dot = X4[i-1]
        x4_dot = ((F_k_b_2 + F_c_b_2 + f_2)*l_2 - (F_k_b_1 + F_c_b_1 + f_1)*l_1)*np.cos(X3[i-1])/J_b
        x5_dot = X6[i-1]
        x6_dot = (- F_k_b_1 - F_c_b_1 + F_k_w_1 + F_c_w_1 - m_w_1*g - f_1)/m_w_1
        x7_dot = X8[i-1]
        x8_dot = (- F_k_b_2 - F_c_b_2 + F_k_w_2 + F_c_w_2 - m_w_2*g - f_2)/m_w_2


        # Update the states - controller
        X1[i] = X1[i-1]+dt*x1_dot
        X2[i] = X2[i-1]+dt*x2_dot
        X3[i] = X3[i-1]+dt*x3_dot
        X4[i] = X4[i-1]+dt*x4_dot
        X5[i] = X5[i-1]+dt*x5_dot
        X6[i] = X6[i-1]+dt*x6_dot
        X7[i] = X7[i-1]+dt*x7_dot
        X8[i] = X8[i-1]+dt*x8_dot

        # Update acceleration - controller
        X2_dot_array[i] = x2_dot
        X4_dot_array[i] = x4_dot


# Round the states to avoid numerical errors
X1_nc = np.round(X1_nc, 8)
X2_nc = np.round(X2_nc, 8)
X3_nc = np.round(X3_nc, 8)
X4_nc = np.round(X4_nc, 8)
X5_nc = np.round(X5_nc, 8)
X6_nc = np.round(X6_nc, 8)
X7_nc = np.round(X7_nc, 8)
X8_nc = np.round(X8_nc, 8)

X1 = np.round(X1, 8)
X2 = np.round(X2, 8)
X3 = np.round(X3, 8)
X4 = np.round(X4, 8)
X5 = np.round(X5, 8)
X6 = np.round(X6, 8)
X7 = np.round(X7, 8)
X8 = np.round(X8, 8)

f_1_array = np.round(f_1_array, 8)
f_2_array = np.round(f_2_array, 8)

### ANIMATION ###
frame_amount=int(len(t)/30+1)
def update_plot(num):
    rod.set_data([-l_1*np.cos(X3[30*num]),l_2*np.cos(X3[30*num])],[X1[30*num]-l_1*np.sin(X3[30*num]),X1[30*num]+l_2*np.sin(X3[30*num])])
    rod_cg_h.set_data([-0.05,0.05],[X1[30*num], X1[30*num]])
    rod_cg_v.set_data([0,0],[X1[30*num]-0.05, X1[30*num]+0.05])
    back_wheel.set_data([-l_1-0.2, -l_1+0.2],[-1 + X5[30*num], -1 + X5[30*num]])
    front_wheel.set_data([l_2-0.2, l_2+0.2],[-1 + X7[30*num], -1 + X7[30*num]])
    y_dist.set_data(x_r - v*t[30*num],y_r-1.25)

    return rod, rod_cg_h, rod_cg_v, back_wheel, front_wheel, y_dist

# Define figure properties
fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(4,4)
plt.subplots_adjust(left=0.03,bottom=0.035,right=0.99,top=0.97,wspace=0.15,hspace=0.2)
ax=fig.add_subplot(gs[:,0:4],facecolor=(0.9,0.9,0.9))
ax.set_xlim(-2, 2)
ax.set_ylim(-1.5, 0.75)

rod,=ax.plot([],[],'k',linewidth=10,label='rod')
rod_cg_h,=ax.plot([],[],'r',linewidth=3,label='rod_cg_h')
rod_cg_v,=ax.plot([],[],'r',linewidth=3,label='rod_cg_v')
front_wheel,=ax.plot([],[],'k',linewidth=20,label='front_wheel')
back_wheel,=ax.plot([],[],'k',linewidth=20,label='back_wheel')
y_dist,=ax.plot(x_r,y_r,'-r',linewidth=3,label='Y-disturbance')

box_object1=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='b',lw=1)
moment=ax.text(1.5,0.5,'FRONT',size=20,color='b',bbox=box_object1)

box_object2=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='r',lw=1)
moment=ax.text(-1.75,0.5,'BACK',size=20,color='r',bbox=box_object2)

box_object3=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='g',lw=1)
ang_vel=ax.text(1.2,0,'v = '+str(np.round(v,2))+' m/s',size=20,color='g',bbox=box_object3)

if mode == 1:
    box_object2=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='g',lw=1)
    moment=ax.text(-0.25,0.5,'PID - APPLIED',size=20,color='g',bbox=box_object2)
else:
    box_object2=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='k',lw=1)
    moment=ax.text(-0.25,0.5,'PID - NOT APPLIED',size=20,color='k',bbox=box_object2)

dt_x=0.25
dt_y=0.25
plt.xticks(np.arange(-2,2+dt_x,dt_x))
plt.yticks(np.arange(-1.5,0.75+dt_y,dt_y))
copyright=ax.text(-2,0.75,'© Mark Misin Engineering',size=12)
plt.grid(True)

suspension_ani=animation.FuncAnimation(fig,update_plot,
    frames=frame_amount,interval=20,repeat=False,blit=True)
plt.show()

# # Matplotlib 3.3.3 needed or newer - comment out plt.show()
# Writer=animation.writers['ffmpeg']
# writer=Writer(fps=30,metadata={'artist': 'Me'},bitrate=1800)
# suspension_ani.save('PID_tuning_TD3.mp4',writer)
# exit()


### PLOTTING ###

### Plot x_b and theta_b ###
plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))

plt.subplot(2, 1, 1, facecolor=(0.9, 0.9, 0.9))
plt.plot(t,X1_nc)
plt.plot(t,X1,'r--')
plt.title(r"System's $x_b$ and $\theta_b$ = [state: $X_1$ and $X_3$] (absolute)",fontsize=15,loc='left')
plt.ylabel(r'$x_b$ [meters]',fontsize=15)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend([r"$x_b$ [no PID]", r"$x_b$ [controller] -> dt = "+str(dt)],loc='upper right',bbox_to_anchor=(1, 1.13),fontsize=15,facecolor=(0.9,0.9,0.9))

plt.subplot(2, 1, 2, facecolor=(0.9, 0.9, 0.9))
plt.plot(t,X3_nc)
plt.plot(t,X3,'r--')
plt.xlabel(r'time [seconds]',fontsize=15)
plt.ylabel(r'$\theta_b$ [rad]',fontsize=15)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend([r"$\theta_b$ [no PID]", r"$\theta_b$ [controller] -> dt = "+str(dt)],loc='upper right',bbox_to_anchor=(1, 1.13),fontsize=15,facecolor=(0.9,0.9,0.9))
plt.show()

# Plot x_b_dot_dot
plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))

plt.subplot(2, 1, 1, facecolor=(0.9, 0.9, 0.9))
plt.plot(t,X2_dot_nc_array)
plt.plot(t,X2_dot_array,'r--')
plt.title(r"System's $\ddot{x}_b$ and $\ddot{\theta}_b$ (absolute) = [state: $\dot{X}_2$ and $\dot{X}_4$]" ,fontsize=15,loc='left')
plt.ylabel(r'$\ddot{x}_b$ [m/$s^2$]',fontsize=15)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend([r"$\ddot{x}_b$ [no controller]", r"$\ddot{x}_b$ [controller] -> dt = "+str(dt)],loc='upper right',bbox_to_anchor=(1, 1.13),fontsize=15,facecolor=(0.9,0.9,0.9))

plt.subplot(2, 1, 2, facecolor=(0.9, 0.9, 0.9))
plt.plot(t,X4_dot_nc_array)
plt.plot(t,X4_dot_array,'r--')
plt.xlabel(r'time [seconds]',fontsize=15)
plt.ylabel(r'$\ddot{\theta}_b$ [rad/$s^2$]',fontsize=15)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend([r"$\ddot{\theta}_b$ [no controller]", r"$\ddot{\theta}_b$ [controller] -> dt = "+str(dt)],loc='upper right',bbox_to_anchor=(1, 1.13),fontsize=15,facecolor=(0.9,0.9,0.9))
plt.show()

### Plot x_b_dot and theta_dot ###
plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))

plt.subplot(2, 1, 1, facecolor=(0.9, 0.9, 0.9))
plt.plot(t,X2_nc)
plt.plot(t,X2,'r--')
plt.title(r"System's $\dot{x}_b$ and $\dot{\theta}_b$ = [state: $X_2$ and $X_4$] (absolute)",fontsize=15,loc='left')
plt.ylabel(r'$\dot{x}_b$ [m/s]',fontsize=15)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend([r"$\dot{x}_b$ [no PID]", r"$\dot{x}_b$ [controller] -> dt = "+str(dt)],loc='upper right',bbox_to_anchor=(1, 1.13),fontsize=15,facecolor=(0.9,0.9,0.9))

plt.subplot(2, 1, 2, facecolor=(0.9, 0.9, 0.9))
plt.plot(t,X4_nc)
plt.plot(t,X4,'r--')
plt.xlabel(r'time [seconds]',fontsize=15)
plt.ylabel(r'$\dot{\theta}_b$ [rad/s]',fontsize=15)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend([r"$\dot{\theta}_b$ [no PID]", r"$\dot{\theta}_b$ [controller] -> dt = "+str(dt)],loc='upper right',bbox_to_anchor=(1, 1.13),fontsize=15,facecolor=(0.9,0.9,0.9))
plt.show()


### Plot x_w_1 and x_w_2 ###
plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))

plt.subplot(2, 1, 1, facecolor=(0.9, 0.9, 0.9))
plt.plot(t,X5_nc)
plt.plot(t,X5,'r--')
plt.title(r"System's $x_{w1}$ and $x_{w2}$ = [state: $X_5$ and $X_7$] (absolute)",fontsize=15,loc='left')
plt.ylabel(r'$x_{w1}$ [meters]',fontsize=15)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend([r"$x_{w1}$ [no PID]", r"$x_{w1}$ [controller] -> dt = "+str(dt)],loc='upper right',bbox_to_anchor=(1, 1.13),fontsize=15,facecolor=(0.9,0.9,0.9))

plt.subplot(2, 1, 2, facecolor=(0.9, 0.9, 0.9))
plt.plot(t,X7_nc)
plt.plot(t,X7,'r--')
plt.xlabel(r'time [seconds]',fontsize=15)
plt.ylabel(r'$x_{w2}$ [meters]',fontsize=15)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend([r"$x_{w2}$ [no PID]", r"$x_{w2}$ [controller] -> dt = "+str(dt)],loc='upper right',bbox_to_anchor=(1, 1.13),fontsize=15,facecolor=(0.9,0.9,0.9))
plt.show()

### Plot x_w_1_dot and x_w_2_dot ###
plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))

plt.subplot(2, 1, 1, facecolor=(0.9, 0.9, 0.9))
plt.plot(t,X6_nc)
plt.plot(t,X6,'r--')
plt.title(r"System's $\dot{x}_{w1}$ and $\dot{x}_{w2}$ = [state: $X_6$ and $X_8$] (absolute)",fontsize=15,loc='left')
plt.ylabel(r'$\dot{x}_{w1}$ [m/s]',fontsize=15)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend([r"$\dot{x}_{w1}$ [no PID]", r"$\dot{x}_{w1}$ [controller] -> dt = "+str(dt)],loc='upper right',bbox_to_anchor=(1, 1.13),fontsize=15,facecolor=(0.9,0.9,0.9))

plt.subplot(2, 1, 2, facecolor=(0.9, 0.9, 0.9))
plt.plot(t,X8_nc)
plt.plot(t,X8,'r--')
plt.xlabel(r'time [seconds]',fontsize=15)
plt.ylabel(r'$\dot{x}_{w2}$ [m/s]',fontsize=15)
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend([r"$\dot{x}_{w2}$ [no PID]", r"$\dot{x}_{w2}$ [controller] -> dt = "+str(dt)],loc='upper right',bbox_to_anchor=(1, 1.13),fontsize=15,facecolor=(0.9,0.9,0.9))
plt.show()


# Plot the control inputs f_1 and f_2
plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))

plt.subplot(2, 1, 1, facecolor=(0.9, 0.9, 0.9))
plt.plot(t,f_1_array)
plt.title("Control inputs $f_1$ and $f_2$",fontsize=15,loc='left')
plt.ylabel('$f_1$ [N]',fontsize=15)
plt.grid()
plt.legend(["$f_1$ (control input - N)"],loc='upper right',bbox_to_anchor=(1, 1.13),fontsize=15,facecolor=(0.9,0.9,0.9))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.subplot(2, 1, 2, facecolor=(0.9, 0.9, 0.9))
plt.plot(t,f_2_array)
plt.xlabel('time [seconds]',fontsize=15)
plt.ylabel('$f_2$ [N]',fontsize=15)
plt.grid()
plt.legend(["$f_2$ (control input - N)"],loc='upper right',bbox_to_anchor=(1, 1.13),fontsize=15,facecolor=(0.9,0.9,0.9))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ticklabel_format(style='plain', axis='y', useOffset=False)
plt.show()
