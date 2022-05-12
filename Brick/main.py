from matplotlib.pyplot import *
from scipy.integrate import odeint
import numpy as np
from scipy.optimize import linprog
from qpsolvers import solve_qp
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matrices import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
from numpy import array, sin, cos, pi, zeros, cross

import time 
seconds = time.time()


def desired_trajectory(t, traj_params, period):
    # desired position, velocity, quaternion, angular velocity
    p_d = np.array(traj_params['state_d'][:3])
    v_d = np.array(traj_params['state_d'][3:6])
    q_d = np.array(traj_params['state_d'][6:10])
    angVel_d = np.array(traj_params['state_d'][10:13])
    return p_d, v_d, q_d, angVel_d


def system(state, t, Q):
    m, g = system_params['m'], system_params['g']

    rc, v, q, w = state[:3], state[3:6], state[6:10], state[10:13]    

    # equation 1
    dr = v

    # equation 2
    Q_v = Q[0]
    dv = m ** (-1) * Q_v - np.array([0, 0, -g])

    # equation 3
    dq = -1/2 * quaternion_product(vect2quat(w), q)

    # equation 4
    Im = np.array(system_params['I'])
    Im_inv = np.linalg.inv(Im)

    tau = Q[1]
    tau_prime = tau - cross_product(w, Im @ w)
    dw = Im_inv @ tau_prime

    dstate = np.concatenate([dr, dv, dq, dw])
    return dstate


def control(state, t, r_d, v_d, quat_d, w_d, params, uncertainty_mode):
    Kp_w = control_params['K1']
    Kd_w = control_params['K2']
    
    Kp_v = control_params['K3']
    Kd_v = control_params['K4']
    
    u_min = control_params['u_min']
    u_max = control_params['u_max']

    r_actual, v_actual, quat_actual, w_actual = state[:3], state[3:6], state[6:10], state[10:13]
    dr, dv = r_d - r_actual, v_d - v_actual
    dw = w_d - w_actual
    
    # q_star = np.hstack([dv + Kp_v @ dr, dw + Kp_w @ dtheta])
    
    q_err = quaternion_product(quat_d, conjugate(quat_actual))
    if q_err[0] < 0:
        q_err_axis = np.array(q_err[1:])
    else:
        q_err_axis = -np.array(q_err[1:])

    a_w = Kp_w @ q_err_axis + Kd_w @ dw
    a_v = Kp_v @ dr + Kd_v @ dv
    Y = regressor(w_actual, a_w, a_v)

    q_star = np.hstack([dv + Kp_v @ dr, dw + Kp_w @ q_err_axis])

    param_vector = p(params)
    
    if uncertainty_mode:
        Q, F = qp_solve_uncertainty(Y, state, q_star, param_vector, u_min, u_max)
    else:
        Q, F = qp_solve(Y, state, param_vector, u_min, u_max)

    return Q[:3], Q[3:], F

    # Q = Y @ params
    # return Q[:3], Q[3:]


def qp_solve(Y, state, param_vector, u_min, u_max):
    Bm = B(state)
    alpha = 0.0001
    P = Bm.T @ Bm + alpha * np.eye(8)
    q = (-1) * param_vector.T @ Y.T @ Bm

    I8 = np.eye(8)
    G = np.vstack([I8, -I8])
    h = np.hstack([np.tile([u_max], 8), np.tile([-u_min], 8)])

    F = solve_qp(P, q, G, h)
    Qm = Bm @ F
    return Qm, F


def qp_solve_uncertainty(Y, state, q_star, param_vector, u_min, u_max):
    # u_min = -100
    # u_max = 100
    z = Y.T @ q_star
    Kp_v = control_params['K3']
    Kp_w = control_params['K1']
    I3 = np.zeros((3,3))
    K = np.block([[Kp_v, I3], [I3, Kp_w]])

    Bm = B(state)
    Am = np.hstack([Bm, -Y])
    alpha = 0.0001
    P = Am.T @ Am + alpha * np.eye(15) # 15x15 matrix: 8 forces, 7 params
    q = (-2) * (param_vector.T @ Y.T + q_star @ K) @ Am

    # m, J
    p1_min = 0.1
    p1_max = 0.3

    p2_min = 0.1
    p2_max = 0.2

    p3_min = 0.001
    p3_max = 0.002

    p1_bounds = [p1_min, p1_max]
    p2_bounds = [p2_min, p2_max]
    p3_bounds = [p3_min, p3_max]

    zp = -linprog(z,
                 bounds=[p1_bounds, p2_bounds, p2_bounds, p2_bounds, p3_bounds, p3_bounds, p3_bounds]
                 ).fun  # minus solves maximize

    I8 = np.eye(8)
    G = np.zeros((17, 15))
    G[:16,:8] = np.vstack([I8, -I8])
    G[16, 8:] = -z
    h = np.hstack([np.tile([u_max], 8), np.tile([-u_min], 8), [zp]])

    X = solve_qp(P, q, G, h)
    F = X[:8]
    # print(F)
    # F = np.linalg.pinv(Bm) @ Y @ param_vector

    Qm = Bm @ F
    return Qm, F


def B(state):

    l = system_params['length']
    w = system_params['width']

    rc = state[:3]
    q  = state[6:10]

    R_trans = quaternion_rotation_matrix(q).T

    # p1, p2, p3, p4 = get_points(rc, q, l, w)

    # s1 = p1 - rc
    # s2 = p2 - rc
    # s3 = p3 - rc
    # s4 = p4 - rc
    
    # B1 = np.hstack([R_trans, R_trans, R_trans, R_trans]) #np.tile(R_trans, (1,4))
    # B2 = (-1) * np.hstack([skew(s1), skew(s2), skew(s3), skew(s4)])
    # B = np.vstack([B1, B2])
    
    positions_vectored = array([[0.156, 0.111, 0.085],
                                [0.156, -0.111, 0.085],
                                [-0.156, 0.111, 0.085],
                                [-0.156, -0.111, 0.085],
                                [0.0, 0.111, 0.],
                                [0.0, -0.111, 0.]]).T


    positions_heavy = array([[0.156, 0.111, 0.085],
                            [0.156, -0.111, 0.085],
                            [-0.156, 0.111, 0.085],
                            [-0.156, -0.111, 0.085],
                            [0.120, 0.218, 0.],
                            [0.120, -0.218, 0.],
                            [-0.120, 0.218, 0.],
                            [-0.120, -0.218, 0.]]).T

    # positions of thrusters
    alphas = array([pi/4, -pi/4,  -pi/4, pi/4])
    directions = array([1, 1, -1, -1, -1, 1, 1, -1])

    # matrix_vectored
    matrix_vectored = zeros((6, 6))
    matrix_heavy = zeros((6, 8))
    normal_vectors = zeros((3, 8))

    for thruster in range(8):
        if thruster < 4:
            normal_vectors[:, thruster] = directions[thruster] * array([cos(alphas[thruster]),
                                                                    -sin(alphas[thruster]),
                                                                    0])
        else:
            normal_vectors[:, thruster] = directions[thruster] * array([0, 0, 1])

        matrix_heavy[:3, thruster] = normal_vectors[:, thruster] # R_trans @ 
        matrix_heavy[3:, thruster] = cross(positions_heavy[:, thruster],
                                        normal_vectors[:, thruster])

        if thruster < 6:
            matrix_vectored[:3, thruster] = R_trans @ normal_vectors[:, thruster]
            matrix_vectored[3:, thruster] = cross(positions_vectored[:, thruster],
                                                normal_vectors[:, thruster])
    
    matrix_heavy[:3, :] = R_trans @ matrix_heavy[:3, :]

    return matrix_heavy


def V():
    V = np.zeros((12,4))
    V[2, 0] = 1
    V[5, 1] = 1
    V[8, 2] = 1
    V[11, 3] = 1
    return V


def L(w):
    L1 = np.diag(w)
    L2 = np.array([[w[1], w[2], 0],
                   [w[0], 0, w[2]],
                   [0, w[0], w[1]]
                  ])
    return np.hstack([L1, L2])


def regressor(w, dw, dv):
    g = np.array([0, 0, (-1) * system_params['g']])
    Y = np.zeros((6,7))
    Y[:3, 0] = dv + g
    Y[3:, 1:] = L(dw) + skew(w) @ L(w)
    return Y


def p(params):
    m = params['m']
    I = params['I']
    
    return np.array([m, I[0][0], I[1][1], I[2][2], I[0][1], I[0][2], I[1][2]])


def get_points(rc, q, length, width):
    dx = length / 2
    dy = width / 2
    dz = height / 2

    Hc = quaternion_affine_matrix(q, rc)

    H = Hc @ Tx(-dx) @ Ty(dy) @ Tz(-dz)
    p1 = H[:3, 3]

    H = Hc @ Tx(dx) @ Ty(dy) @ Tz(-dz)
    p2 = H[:3, 3]

    H = Hc @ Tx(dx) @ Ty(-dy) @ Tz(-dz)
    p3 = H[:3, 3]

    H = Hc @ Tx(-dx) @ Ty(-dy) @ Tz(-dz)
    p4 = H[:3, 3]

    H = Hc @ Tx(-dx) @ Ty(dy) @ Tz(dz)
    p5 = H[:3, 3]

    H = Hc @ Tx(dx) @ Ty(dy) @ Tz(dz)
    p6 = H[:3, 3]

    H = Hc @ Tx(dx) @ Ty(-dy) @ Tz(dz)
    p7 = H[:3, 3]

    H = Hc @ Tx(-dx) @ Ty(-dy) @ Tz(dz)
    p8 = H[:3, 3]


    
    # points = np.array([[-1, -1, -1],
    #                    [1, -1, -1],
    #                    [1, 1, -1],
    #                    [-1, 1, -1]
    #                    ,[-1, -1, 1],
    #                    [1, -1, 1],
    #                    [1, 1, 1],
    #                    [-1, 1, 1]
    #                    ]) @ np.diag([dx, dy, dz])
    points = np.array([
                    #    [-1, -1, -1],
                    #    [1, -1, -1],
                    #    [1, 1, -1],
                    #    [-1, 1, -1],
                       [-1, 1, 1],
                       [1, 1, 1],
                       [1, -1, 1],
                       [-1, -1, 1]
                       ]) @ np.diag([dx, dy, 0]) 
    # print(points)   

    # p1 = quaternion_multiply3(q, np.array([rc[0]-dx, rc[1]+dy, rc[2]-dz]), conjugate(q))
    # p2 = quaternion_multiply3(q, np.array([rc[0]+dx, rc[1]+dy, rc[2]-dz]), conjugate(q))
    # p3 = quaternion_multiply3(q, np.array([rc[0]+dx, rc[1]-dy, rc[2]-dz]), conjugate(q))
    # p4 = quaternion_multiply3(q, np.array([rc[0]-dx, rc[1]-dy, rc[2]-dz]), conjugate(q))
    # p5 = quaternion_multiply3(q, np.array([rc[0]-dx, rc[1]+dy, rc[2]+dz]), conjugate(q))
    # p6 = quaternion_multiply3(q, np.array([rc[0]+dx, rc[1]+dy, rc[2]+dz]), conjugate(q))
    # p7 = quaternion_multiply3(q, np.array([rc[0]+dx, rc[1]-dy, rc[2]+dz]), conjugate(q))
    # p8 = quaternion_multiply3(q, np.array([rc[0]-dx, rc[1]-dy, rc[2]+dz]), conjugate(q))
        
    # print('p1:', p1)
    return np.vstack([p1, p2, p3, p4, p5, p6, p7, p8])


def simulate_system(state_init, frequency, t_0, t_final):
    dT = 1/frequency
    t = np.arange(t_0, t_final, dT)
    t_star = np.linspace(0, dT, 5)
    per = 3
    state_prev = state_prev_modified = state_init
    states, states_mod, quats_d, angles = [], [], [], []
    poss_d = []
    U_pos, U_att, U_pos_mod = [], [], []
    Fs, Fs_mod = [], []
    for i in range(len(t)):
        t_curr = t[i]
        pos_d, vel_d, quat_d, angVel_d = desired_trajectory(t_curr, trajectory_params, per)

        u_real = control(state_prev, t_curr, pos_d, vel_d, quat_d, angVel_d, system_params, uncertainty_mode=False)
        u_modified = control(state_prev_modified, t_curr, pos_d, vel_d, quat_d, angVel_d, control_params, uncertainty_mode=True)

        state = odeint(system, state_prev, t_star, args=(u_real, ))
        state_modified = odeint(system, state_prev_modified, t_star, args=(u_modified, ))

        state_prev = state[-1]
        state_prev_modified = state_modified[-1]

        quats_d.append(quat_d)
        poss_d.append(pos_d)

        states.append(state_prev)
        states_mod.append(state_prev_modified)

        quat_cur = state_prev[6:10]
        quat_ = np.array([quat_cur[3], quat_cur[0], quat_cur[1], quat_cur[2]])
        angle = Rotation.from_quat(quat_)
        angles.append(angle.as_euler('xyz'))

        U_pos.append(u_real[0])
        U_pos_mod.append(u_modified[0])
        U_att.append(u_real[1])
        Fs.append(u_real[2])
        Fs_mod.append(u_modified[2])  

    quats_d = np.array(quats_d)
    poss_d = np.array(poss_d)
    states = np.array(states)
    states_mod = np.array(states_mod)
    angles = np.array(angles)
    U_pos = np.array(U_pos)
    U_pos_mod = np.array(U_pos_mod)
    U_att = np.array(U_att)
    Fs = np.array(Fs)
    Fs_mod = np.array(Fs_mod)    

    figure()
    text = ['$q_0$', '$q_1$', '$q_2$', '$q_3$']
    colors = ['r', 'g', 'b', 'k']
    styles = [':', '--', '-', '-.']
    for i in range(4):
        plot(t, states[:,i+6], color=str(colors[i]), linewidth=2.0, linestyle=str(styles[i]), label=str(text[i]))
        plot(t, states_mod[:,i+6], linewidth=2.0, linestyle=str(styles[i]), label=str(text[i]))
        plot(t, quats_d[:, i], color=str(colors[i]), linewidth=1.0, linestyle=':')
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'Quaternion ${q}$')
    xlabel(r'Time $t$ (s)')
    legend(loc='lower right')
    title('Quaternion')
    savefig('Brick/quaternion.png')


    figure()
    text = [r'$\phi$', r'$\theta$', r'$\psi$']
    for i in range(3):
        plot(t, angles[:,i], linewidth=2.0, label=str(text[i]))
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'Angle $rad$')
    xlabel(r'Time $t$ (s)')
    legend(loc='lower right')
    title('Orientation')
    savefig('Brick/orientation.png')


    figure()
    text = ['$x_{real}$', '$y_{real}$', '$z_{real}$']
    text1 = ['$x$', '$y$', '$z$']
    text2 = ['$x_{des}$', '$y_{des}$', '$z_{des}$']
    colors1 = ['orange', 'y', 'k']
    styles = ['-.', '--', '-']
    for i in range(1):
        i = 2
        plot(t, states[:, i], color=str(colors[i]), linewidth=2.0, linestyle=str(styles[i]), label=str(text[i]))
        plot(t, states_mod[:, i], color=str(colors1[i]), linewidth=2.0, linestyle=str(styles[i]), label=str(text1[i]))
        plot(t, poss_d[:, i], color=str(colors[i]), linewidth=1.5, linestyle=':', label=str(text2[i]))
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'Position ${p}$')
    xlabel(r'Time $t$ (s)')
    legend(loc='lower right')
    title('Position')
    savefig('Brick/position.png')


    figure()
    text = ['$v_x$', '$v_y$', '$v_z$']
    for i in range(3):
        plot(t, states[:, i+3], linewidth=2.0, label=str(text[i]))
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'Velocity ${p}$')
    xlabel(r'Time $t$ (s)')
    legend(loc='lower right')
    title('Velocity')
    savefig('Brick/velocity.png')


    figure()
    text = ['$u_x$', '$u_y$', '$u_z$']
    for i in range(3):
        plot(t, U_pos[:, i], linewidth=2.0, label=str(text[i]))
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'Position control ${U}$')
    xlabel(r'Time $t$ (s)')
    legend(loc='lower right')
    title('Position control')
    savefig('Brick/pos_control.png')

    figure()
    text = ['$u_{roll}}$', '$u_{pitch}}$', '$u_{yaw}}$']
    for i in range(3):
        plot(t, U_att[:, i], linewidth=2.0, label=str(text[i]))
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'Attitude control ${U}$')
    xlabel(r'Time $t$ (s)')
    legend(loc='lower right')
    title('Attitude control')
    savefig('Brick/attitude_control.png')


    figure()
    text = ['$F_{1}}$', '$F_{2}}$', '$F_{3}}$', '$F_{4}}$', '$F_{5}}$', '$F_{6}}$', '$F_{7}}$', '$F_{8}}$']
    for i in range(8):
        plot(t, Fs[:, i], linewidth=2.0, label=str(text[i]))
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'Force ${F}$')
    xlabel(r'Time $t$ (s)')
    legend(loc='lower right')
    title('Forces')
    savefig('Brick/force.png')


    figure()
    text = ['$F_{1}}$', '$F_{2}}$', '$F_{3}}$', '$F_{4}}$', '$F_{5}}$', '$F_{6}}$', '$F_{7}}$', '$F_{8}}$']
    for i in range(1):
        plot(t, Fs_mod[:, 3], linewidth=2.0, label=str(text[i]))
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])
    ylabel(r'Force ${F}$')
    xlabel(r'Time $t$ (s)')
    legend(loc='lower right')
    title('Forces')
    savefig('Brick/force_modified.png')


    close('all')
    # show()

    # ANIMATION
    def get_position(states):
        pos = np.vstack([states[:, 0], states[:, 1], states[:, 2]])
        return pos


    def get_current_orientation(i):
        return angles[i, 0], angles[i, 1], angles[i, 2]

    
    def get_current_quaternion(i):
        return states[i, 6:10] 


    def animate(num):
        ax.clear()
        rc = np.array([data[0, num], data[1, num], data[2, num]])
        quat = get_current_quaternion(num)

        points = get_points(rc, quat, length, width)

        # points = np.vstack([p1, p2, p3, p4])
        # points = np.array([[-1, -1, -1],
        #                    [1, -1, -1],
        #                    [1, 1, -1],
        #                    [-1, 1, -1]
        #                    ,[-1, -1, 1],
        #                    [1, -1, 1],
        #                    [1, 1, 1],
        #                    [-1, 1, 1]
        #                    ])
        Z = points
        verts = [[Z[0], Z[1], Z[2], Z[3]]
                ,[Z[4], Z[5], Z[6], Z[7]],
                [Z[0], Z[1], Z[5], Z[4]],
                [Z[2], Z[3], Z[7], Z[6]],
                [Z[1], Z[2], Z[6], Z[5]],
                [Z[4], Z[7], Z[3], Z[0]]
                ]

        ax.scatter3D(data[0, 0], data[1, 0], data[2, 0], color='b')

        state_d = trajectory_params['state_d']
        ax.scatter3D(state_d[0], state_d[1], state_d[2], color='r')

        ax.scatter3D(rc[0], rc[1], rc[2], color='b', alpha=.50)
        ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2], color='b', alpha=.50)
        collection = Poly3DCollection(verts, facecolors='b', linewidths=1, edgecolors='r', alpha=.10)
        ax.add_collection3d(collection)
        ax.set_xlim3d([-0.3, 0.5])
        ax.set_ylim3d([0.5, 1.5])
        ax.set_zlim3d([-0.1, 0.7])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    fig = figure()
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    data = get_position(states)

    ani = animation.FuncAnimation(fig, animate, frames=np.shape(data)[1],
                                    interval=dT) #repeat=False,
                                    
    # writer = animation.PillowWriter(fps=100)
    # ani.save('Brick/animation.gif', writer=writer)
    show()


t0 = 0
tf = 10.0
freq = 100 # 500
dT = 1/freq
g = 9.81

# Brick parameters
length = 0.5
width = 0.5
height = 0.2

m_real = 12

mat = np.array([[(width/2) ** 2 + (height/2) ** 2, 0.001, 0.001],
                    [0.001, (height/2) ** 2 + (length/2) ** 2, 0.001],
                    [0.001, 0.001, (length/2) ** 2 + (width/2) ** 2]]) / 3

I_real = m_real * mat

system_params = {'length': length,
                 'width': width,
                 'height': height,
                 'm': m_real,
                 'I': I_real,
                 'g': g,
                 }


m_modified = m_real * 0.8
I_modified = m_modified * mat

u_min = -500.0
u_max = 500.0

# gains for orientational components: Kp_w, Kd_w
K1 = np.diag([15, 15, 15])
K2 = np.diag([10, 10, 10])

# gains for positional components: Kp_v, Kd_v
K3 = np.diag([15, 15, 30])
K4 = np.diag([25, 25, 25])

K_pos = np.block([K1, K1])
K_att = np.block([K2, K2])

control_params = {'K1': K1, 'K2': K2,
                  'K3': K3, 'K4': K4,
                  'K_pos': K_pos, 'K_att': K_att,
                  'u_min': u_min, 'u_max': u_max,
                  'm': m_modified,
                  'I': I_modified,
                  }


# rot = Rotation.from_euler('xyz', [90, 45, 30], degrees=True)
# rot_quat = rot.as_quat()
# print(rot_quat)

# print(rot.as_euler('xyz'))

# rot = Rotation.from_quat(rot_quat)

# # Convert the rotation to Euler angles given the axes of rotation
# print(rot.as_euler('xyz'))

quat_0 = Rotation.from_euler('xyz', [10, 10, 5], degrees=True).as_quat()
quat_d = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()
print(quat_0, quat_d)


# initial position, velocity, quaternion, angular velocity
x0 = [0., 1.0, 0.0,
      0, 0, 0,
      quat_0[3], quat_0[0], quat_0[1], quat_0[2], 
      0., 0., 0.]

trajectory_params = {'state_d': [0.3, 0.7, 0.5,
                                 0., 0., 0.,
                                 quat_d[3], quat_d[0], quat_d[1], quat_d[2], 
                                 0., 0., 0.]}

simulate_system(state_init=x0, frequency=freq, t_0=t0, t_final=tf)


# seconds_last = time.time()

# print(seconds_last - seconds)