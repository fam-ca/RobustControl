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


def cross_product(p, q):
    return [p[1]*q[2] - p[2]*q[1],
            p[2]*q[0] - p[0]*q[2],
            p[0]*q[1] - p[1]*q[0]]


def quaternion_product(p,q):
    Q = np.array([[p[0], -p[1], -p[2], -p[3]],
                  [p[1],  p[0], -p[3],  p[2]],
                  [p[2],  p[3],  p[0], -p[1]],
                  [p[3], -p[2],  p[1],  p[0]]])
    return Q @ q


def conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def D():
    m, I = system_params['m'], system_params['I']
    ms = m * np.eye(3)
    Is = I * np.eye(3)
    zeros = np.zeros((3,3))

    D = np.block([[ms, zeros],
                  [zeros, Is]])
    return D


def h():
    m, g = system_params['m'], system_params['g']
    h = np.vstack([np.array([[0], [m*g]]), np.zeros((4,1))])
    return h


def B():
    state = [0.5,0.5,0.5,  # pos
             0.1,0.2,0.14,  # orient
             0,0.3,0.4,0.2]  # quaternion
    rc = state[:3]
    wc = state[3:6]
    l = system_params['length']
    w = system_params['width']

    mat_rc_wc = Tx(rc[0]) @ Ty(rc[1]) @ Tz(rc[2]) @ Rx(wc[0]) @ Ry(wc[1]) @ Rz(wc[2])

    c1 = (mat_rc_wc @ Tx(-l/2) @ Ty(w/2))[0:3, 3]
    c2 = (mat_rc_wc @ Tx( l/2) @ Ty(w/2))[0:3, 3]
    c3 = (mat_rc_wc @ Tx( l/2) @ Ty(-w/2))[0:3, 3]
    c4 = (mat_rc_wc @ Tx(-l/2) @ Ty(-w/2))[0:3, 3]

    s1 = c1 - rc
    s2 = c2 - rc
    s3 = c3 - rc
    s4 = c4 - rc

    eye3 = np.eye(3)
    B1 = np.tile(eye3, (1,4))
    B2 = np.hstack([skew(s1), skew(s2), skew(s3), skew(s4)])
    B = np.vstack([B1, B2])
    return B
# ----------------------------------------------------------------------------------------------------------------------------------

def desired_trajectory(t, traj_params, period):
    # desired position, velocity, quaternion, angular velocity
    p_d = np.array(traj_params['state_d'][:3])
    v_d = np.array(traj_params['state_d'][3:6])
    q_d = np.array(traj_params['state_d'][6:10])
    angVel_d = np.array(traj_params['state_d'][10:13])
    return p_d, v_d, q_d, angVel_d


def vect2quat(v):
    return np.concatenate([[0], v])


def quat2vect(q):
    return q[1:]


def ln_quat(q):
    mod_vectq = np.linalg.norm(q[1:])
    if mod_vectq < 1e-15:
        return np.zeros(3)
    else:
        return np.array(q[1:])*np.arccos(q[0])*mod_vectq**(-1)


def quat2euler_angles(q):
    phi = np.arctan2((2 * (q[0] * q[1] + q[2] * q[3])), (1 - 2 * (q[1] ** 2 + q[2] ** 2)))
    theta = np.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
    psi = np.arctan2((2 * (q[0] * q[3] + q[1] * q[2])), (1 - 2 * (q[2] ** 2 + q[3] ** 2)))
    return np.array([phi, theta, psi])


def euler2quat(roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qw, qx, qy, qz]


def system(state, t, u):
    m, g = system_params['m'], system_params['g']
    # v, dv - velocity and acceleration
    # q - quaternion
    # w - angular velocity

    r, v, q, w = state[:3], state[3:6], state[6:10], state[10:13]

    # equation 1 dp=dp
    dr = v

    # equation 2
    u_v = u[0]
    dv = u_v - np.array([0, 0, -g])

    # equation 3
    dq = -1/2 * quaternion_product(vect2quat(w), q)

    # equation 4
    Im = np.array(system_params['I'])
    Im_inv = np.linalg.inv(Im)
    tau = u[1]
    T = tau - cross_product(w, Im @ w)
    dw = np.dot(Im_inv, T)

    dstate = np.concatenate([dr, dv, dq, dw])
    return dstate


# def control(state, t, r_d, v_d, q_d, w_d):
#     m, g = system_params['m'], system_params['g']
#     r_actual, v_actual, q_actual, w_actual = state[:3], state[3:6], state[6:10], state[10:13]

#     K_pos = control_params['K_pos']
#     K_att = control_params['K_att']

#     theta_actual = 2 * ln_quat(q_actual)
#     theta_d = 2 * ln_quat(q_d)
#     x_att_err = np.concatenate([theta_actual - theta_d, w_actual - w_d])
#     u_att = -K_att @ x_att_err

#     x_pos_err = np.concatenate([r_actual-r_d, v_actual - v_d])
#     u_pos = -K_pos @ x_pos_err + np.array([0, 0, -g])

#     return u_pos, u_att


def control(state, t, r_d, v_d, q_d, w_d):
    m = system_params['m']

    K_p = control_params['K1']
    K_d = control_params['K2']

    r_actual, v_actual, q_actual, w_actual = state[:3], state[3:6], state[6:10], state[10:13]
    
    params = p(system_params)

    q_err = quaternion_product(q_d, conjugate(q_actual))
    if q_err[0]<0:
        q_err_axis = -np.array(q_err[1:])
    else:
        q_err_axis = np.array(q_err[1:])

    u_w = -K_p @ q_err_axis + K_d @ (w_d - w_actual)
    u_v = K_p @ (r_d - r_actual) + K_d @ (v_d - v_actual)
    Y = regressor(w_actual, u_w, u_v)
    u = Y @ params
    return m ** (-1) * u[:3], u[3:]


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


def p(sys_params):
    m = sys_params['m']
    Ixx = sys_params['Ixx']
    Iyy = sys_params['Iyy']
    Izz = sys_params['Izz']
    Ixy = sys_params['Ixy']
    Ixz = sys_params['Ixz']
    Iyz = sys_params['Iyz']
    return np.array([m, Ixx, Iyy, Izz, Ixy, Ixz, Iyz])


def simulate_system(state_init, frequency, t_0, t_final):
    dT = 1/frequency
    t = np.arange(t_0, t_final, dT)
    t_star = np.linspace(0, dT, 10)
    per = 3
    state_prev = state_init
    states, quats, angles = [], [], []
    U_pos, U_att = [], []
    for i in range(len(t)):
        t_curr = t[i]
        pos_d, vel_d, quat_d, angVel_d = desired_trajectory(t_curr, trajectory_params, per)
        u = control(state_prev, t_curr, pos_d, vel_d, quat_d, angVel_d)

        state = odeint(system, state_prev, t_star,
                           args=(u,))
        state_prev = state[-1]

        quats.append(state_prev[6:10])
        # d_w.append(state_prev[4:7])
        states.append(state_prev)
        angle = Rotation.from_quat(state_prev[6:10])
        angles.append(angle.as_euler('xyz'))
        U_pos.append(u[0])
        U_att.append(u[1])

    states = np.array(states)
    quats = np.array(quats)
    angles = np.array(angles)
    U_pos = np.array(U_pos)
    U_att = np.array(U_att)


    figure()
    text = ['$q_0$', '$q_1$', '$q_2$', '$q_3$']
    for i in range(4):
        plot(t, quats[:,i], linewidth=2.0, label=str(text[i]))
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
    text = ['$x$', '$y$', '$z$']
    for i in range(3):
        plot(t, states[:,i], linewidth=2.0, label=str(text[i]))
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
        xc, yc, zc = data[0, num], data[1, num], data[2, num]
        dx = length / 2
        dy = width / 2
        roll, pitch, yaw = get_current_orientation(num)
        quat = get_current_quaternion(num)
        Hc = Tx(xc) @ Ty(yc) @ Tz(zc) @ quaternion_affine_matrix(quat, [0, 0, 0]) #Rx(roll) @ Ry(pitch) @ Rz(yaw)
        H = Hc @ Tx(dx) @ Ty(-dy)
        p1 = H[:3, 3]

        H = Hc @ Tx(-dx) @ Ty(-dy)
        p2 = H[:3, 3]

        H = Hc @ Tx(-dx) @ Ty(dy)
        p3 = H[:3, 3]

        H = Hc @ Tx(dx) @ Ty(dy)
        p4 = H[:3, 3]

        points = np.vstack([p1, p2, p3, p4])
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
                # ,[Z[4], Z[5], Z[6], Z[7]],
                # [Z[0], Z[1], Z[5], Z[4]],
                # [Z[2], Z[3], Z[7], Z[6]],
                # [Z[1], Z[2], Z[6], Z[5]],
                # [Z[4], Z[7], Z[3], Z[0]]
                ]

        ax.scatter3D(data[0, 0], data[1, 0], data[2, 0], color='b')
        ax.scatter3D(data[0, -1], data[1, -1], data[2, -1], color='r')

        ax.scatter3D(xc, yc, zc, color='b', alpha=.50)
        ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2], color='b', alpha=.50)
        collection = Poly3DCollection(verts, facecolors='b', linewidths=1, edgecolors='r', alpha=.10)
        ax.add_collection3d(collection)
        ax.set_xlim3d([0.0, 2.])
        ax.set_ylim3d([0.0, 3.0])
        ax.set_zlim3d([-1.0, 1.0])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    fig = figure()
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    data = get_position(states)

    ani = animation.FuncAnimation(fig, animate, frames=np.shape(data)[1],
                                    interval=dT) #repeat=False,
                                    
    writer = animation.PillowWriter(fps=100)
    ani.save('Brick/animation.gif', writer=writer)
    show()


t0 = 0
tf = 10.0
freq = 100
dT = 1/freq

# Brick parameters
length = 0.8
width = 0.4
height = 0.1
m = 1.3
Ixx = Iyy = 0.5  # 6.5 * 10**(-4)
Izz = 0.4  # 1.2*10**(-3)
Ixy = Iyx = 0.001
Ixz = Izx = 0.001
Iyz = Izy = 0.001
g = 9.81

I = np.array([[Ixx, Ixy, Ixz],
              [Iyx, Iyy, Iyz],
              [Izx, Izy, Izz]])

K1 = np.diag([15, 15, 15])
K2 = np.diag([5, 5, 5])

K_pos = np.block([K1, K1])
K_att = np.block([K2, K2])

system_params = {'length': length,
                 'width': width,
                 'height': height,
                 'm': m,
                 'I': I,
                 'Ixx': Ixx,
                 'Iyy': Iyy,
                 'Izz': Izz,
                 'Ixy': Ixy,
                 'Ixz': Ixz,
                 'Iyz': Iyz,
                 'g': g,
                 }

control_params = {'K1': K1, 'K2': K2,
                  'K_pos': K_pos, 'K_att': K_att,
                  }


# rot = Rotation.from_euler('xyz', [90, 45, 30], degrees=True)
# rot_quat = rot.as_quat()
# print(rot_quat)

# print(rot.as_euler('xyz'))

# rot = Rotation.from_quat(rot_quat)

# # Convert the rotation to Euler angles given the axes of rotation
# print(rot.as_euler('xyz'))

quat_0 = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()
quat_d = Rotation.from_euler('xyz', [-20, 30, 30], degrees=True).as_quat()
print(quat_0, quat_d)


# initial position, velocity, quaternion, angular velocity
x0 = [1., 2.0, 0.0,
      0, 0, 0,
      0, 0, 0, 0,
      0., 0., 0.]
x0[6:10] = quat_0

trajectory_params = {'state_d': [0.5, 2.5, 1,
                                 0., 0., 0.,
                                 quat_d[0], quat_d[1], quat_d[2], quat_d[3],
                                 0., 0., 0.]}

simulate_system(state_init=x0, frequency=freq, t_0=t0, t_final=tf)