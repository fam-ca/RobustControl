from matplotlib.pyplot import *
from scipy.integrate import odeint
import numpy as np
from scipy.optimize import linprog
from qpsolvers import solve_qp
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matrices import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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

    mat_rc_wc = Tx(rc[0]) @ Ty(rc[1]) @ Tz(rc[2]) \
                @ Rx(wc[0]) @ Ry(wc[1]) @ Rz(wc[2])

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
# ---------------------------------------------------------------------

def desired_trajectory(t, traj_params, period):
    # desired position, velocity, quaternion, angular velocity
    p_d = np.array(traj_params['state_d'][:3])
    v_d = np.array(traj_params['state_d'][3:6])
    q_d = np.array(traj_params['state_d'][6:10])
    angVel_d = np.array(traj_params['state_d'][10:13])
    return p_d, v_d, q_d, angVel_d


def inertia_matrix():
    Ixx = system_params['Ixx']
    Iyy = system_params['Iyy']
    Izz = system_params['Izz']
    Is = np.diag([Ixx, Iyy, Izz])
    return Is


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


# def get_force(w):
#     # w - angular velocity
#     # k - constant coefficient
#     k = system_params['k']
#     F_th = np.array([0,0, 0])
#     # F_th = np.array([k * 10**2, k * 10**2, k * 10**2])
#     return F_th


def quat2euler_angles(q):
    phi = np.arctan2((2 * (q[0] * q[1] + q[2] * q[3])), (1 - 2 * (q[1] ** 2 + q[2] ** 2)))
    theta = np.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
    psi = np.arctan2((2 * (q[0] * q[3] + q[1] * q[2])), (1 - 2 * (q[2] ** 2 + q[3] ** 2)))
    return np.array([phi, theta, psi])


def system(state, t, u):
    m, g = system_params['m'], system_params['g']
    # dp, ddp - velocity and acceleration
    # q - quaternion
    # w - angular velocity

    p, dp, q, w = state[:3], state[3:6], state[6:10], state[10:13]

    # equation 1 dp=dp
    # equation 2
    # F = get_force(w)
    # qF = quaternion_product(q,vect2quat(F))
    # rotated_F = quaternion_product(qF, conjugate(q))
    # rotated_F_vect = quat2vect(rotated_F)
    u_pos = u[0]
    u_pd = u_pos - np.array([0, 0, -g])
    ddp = u_pd + np.array([0, 0, -g])
    # equation 3
    dq = -1/2 * quaternion_product(vect2quat(w), q)

    # equation 4
    Im = inertia_matrix()
    Im_inv = np.linalg.inv(Im)
    u_att = u[1]
    T = u_att - cross_product(w, Im @ w)
    dw = np.dot(Im_inv, T)

    dstate = np.concatenate([dp, ddp, dq, dw])
    return dstate


def control(state, t, p_d, dp_d, q_d, w_d):
    # actual/estimated quaternion and angular velocity
    g = system_params['g']
    p_actual, dp_actual, q_actual, w_actual = state[:3], state[3:6], state[6:10], state[10:13]
    q_actual_conjugate = conjugate(q_actual)
    # q_err = quaternion_product(q_d, q_actual_conjugate)

    K_pos = control_params['K_pos']
    K_att = control_params['K_att']

    # q_err3 = np.array(q_err[1:])
    theta_actual = 2 * ln_quat(q_actual)
    theta_d = 2 * ln_quat(q_d)
    x_att_err = np.concatenate([theta_actual - theta_d, w_actual - w_d])
    u_att = -K_att @ x_att_err

    # F_th = get_force(w_actual)
    # qF = quaternion_product(q_actual,vect2quat(F_th))
    # rotated_F = quaternion_product(qF, conjugate(q_actual))
    # u_p = quat2vect(rotated_F)

    x_pos_err = np.concatenate([p_actual-p_d, dp_actual - dp_d])
    u_pos = -K_pos @ x_pos_err
    # u = u_att + u_pos
    return u_pos, u_att


def simulate_system(state_init, frequency, t_0, t_final):
    dT = 1/frequency
    t = np.arange(t_0, t_final, dT)
    t_star = np.linspace(0, dT, 10)
    per = 3
    state_prev = state_init
    quats, d_w, states, angles = [], [], [], []
    U_pos, U_att = [], []
    for i in range(len(t)):
        t_curr = t[i]
        pos_d, vel_d, quat_d, angVel_d = desired_trajectory(t_curr, trajectory_params, per)
        u = control(state_prev, t_curr, pos_d, vel_d, quat_d, angVel_d)

        state = odeint(system, state_prev, t_star,
                           args=(u,))
        state_prev = state[-1]

        quats.append(state_prev[6:10])
        d_w.append(state_prev[4:7])
        states.append(state_prev)
        angle = quat2euler_angles(state_prev[6:10])
        angles.append(angle)
        U_pos.append(u[0])
        U_att.append(u[1])
    figure()
    text = ['$q_0$', '$q_1$', '$q_2$', '$q_3$']
    for i in range(4):
        plot(t, np.array(quats)[:,i], linewidth=2.0, label=str(text[i]))
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
        plot(t, np.array(angles)[:,i], linewidth=2.0, label=str(text[i]))
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
        plot(t, np.array(states)[:,i], linewidth=2.0, label=str(text[i]))
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
        plot(t, np.array(states)[:,i+3], linewidth=2.0, label=str(text[i]))
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
        plot(t, np.array(U_pos)[:, i], linewidth=2.0, label=str(text[i]))
    grid(color='black', linestyle='--', linewidth=0.7, alpha=0.7)
    xlim([t_0, t_final])

    ylabel(r'Position control ${U}$')
    xlabel(r'Time $t$ (s)')
    legend(loc='lower right')
    title('Position control')
    savefig('Brick/pos_control.png')

    close('all')
    # show()
    return np.array(states), np.array(angles)




t0 = 0
tf = 5.0
freq = 100
dT = 1/freq

# Brick parameters
length = 0.8
width = 0.4
height = 0.1
m = 1.3
I = 0.05
Ixx = Iyy = 0.5  # 6.5 * 10**(-4)
Izz = 0.4  # 1.2*10**(-3)
g = 9.81

K1 = np.diag([3, 3, 3])
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
                 'g': g,
                 }

control_params = {'K1': K1, 'K2': K2,
                  'K_pos': K_pos, 'K_att': K_att,
                  }

quat_0 = np.array([0.5, 0.5,0.1,0.5])
quat_0 = quat_0/np.linalg.norm(quat_0)

# initial position, velocity, quaternion, angular velocity
x0 = [1., 2.0, 0.0,
      0, 0, 0,
      0, 0, 0, 0,
      0., 0., 0.]
x0[6:10] = quat_0

trajectory_params = {'state_d': [0, 0., 1,
                                 0., 0., 0.,
                                 1., 0.0, 0.0, 0.0,
                                 0., 0., 0.]}

states, angles = simulate_system(state_init=x0, frequency=freq, t_0=t0, t_final=tf)


def get_position(states):
    pos = np.vstack([states[::10, 0], states[::10, 1], states[::10, 2]])
    return pos


def get_current_quat(i):
    return states[i, 6:10]


def get_current_orientation(i):
    return angles[i, 0], angles[i, 1], angles[i, 2]


def animate(num):
    ax.clear()
    xc, yc, zc = data[0][num], data[1][num], data[2][num]
    roll, pitch, yaw = get_current_orientation(num)
    Hc = Tx(xc) @ Ty(yc) @ Tz(zc)  @ Rx(roll) @ Ry(pitch) @ Rz(yaw)
    H = Hc @ Tx(length/2) @ Ty(-width/2)
    p1 = H[:3, 3]

    H = Hc @ Tx(-length / 2) @ Ty(-width / 2)
    p2 = H[:3, 3]

    H = Hc @ Tx(-length / 2) @ Ty(width / 2)
    p3 = H[:3, 3]

    H = Hc @ Tx(length / 2) @ Ty(width / 2)
    p4 = H[:3, 3]

    points = np.vstack([p1, p2, p3, p4])
    # points = np.array([[-p1[0], -p1[1], -p1[2]],
    #                    [p1[0], -p1[1], -p1[2]],
    #                    [p1[0], p1[1], -p1[2]],
    #                    [-p1[0], p1[1], -p1[2]]
    #                    ,[-1, -1, 1],
    #                    [1, -1, 1],
    #                    [1, 1, 1],
    #                    [-1, 1, 1]
    #                    ])
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

    # ax.scatter3D(data[0][:], data[1][:], data[2][:])
    ax.scatter3D(data[0][0], data[1][0], data[2][0], color='b')
    ax.scatter3D(data[0][-1], data[1][-1], data[2][-1], color='r')

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

# Attach 3D axis to the figure
fig = figure()
ax = p3.Axes3D(fig,  auto_add_to_figure=False)
fig.add_axes(ax)

data = get_position(states)

ani = animation.FuncAnimation(fig, animate, frames=np.shape(data)[1],
                                  interval=dT * 1e4) #repeat=False,
                                  
writer = animation.PillowWriter(fps=100)
ani.save('Brick/animation.gif', writer=writer)
show()



# """
# A simple example of an animated plot... In 3D!
# """
# import numpy as np
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as p3
# import matplotlib.animation as animation
#
# def Gen_RandLine(length, dims=2) :
#     """
#     Create a line using a random walk algorithm
#
#     length is the number of points for the line.
#     dims is the number of dimensions the line has.
#     """
#     lineData = np.zeros((dims, length))
#     print('before', lineData)
#     print()
#     lineData[:, 0] = np.random.rand(dims)
#     print("after", lineData)
#
#     for index in range(1, length) :
#         # scaling the random numbers by 0.1 so
#         # movement is small compared to position.
#         # subtraction by 0.5 is to change the range to [-0.5, 0.5]
#         # to allow a line to move backwards.
#         step = ((np.random.rand(dims) - 0.5) * 0.1)
#         lineData[:, index] = lineData[:, index-1] + step
#
#     return lineData
#
# def update_lines(num, dataLines, lines) :
#     for line, data in zip(lines, dataLines) :
#         # NOTE: there is no .set_data() for 3 dim data...
#         line.set_data(data[0:2, :num])
#         line.set_3d_properties(data[2,:num])
#     # print(lines)
#     return lines
#
# # Attaching 3D axis to the figure
# fig = plt.figure()
# ax = p3.Axes3D(fig)
#
# # Fifty lines of random 3-D lines
# n_lines = 10 # number of random lines
# n_points = 15 # number of random points
# data = [Gen_RandLine(15, 3) for index in range(n_lines)]
#
# # Creating fifty line objects.
# # NOTE: Can't pass empty arrays into 3d version of plot()
# lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
# # print(lines)
#
# # Setting the axes properties
# ax.set_xlim3d([0.0, 1.0])
# ax.set_xlabel('X')
#
# ax.set_ylim3d([0.0, 1.0])
# ax.set_ylabel('Y')
#
# ax.set_zlim3d([0.0, 1.0])
# ax.set_zlabel('Z')
#
# ax.set_title('3D Test')
#
# # Creating the Animation object
# line_ani = animation.FuncAnimation(fig, update_lines, 5, fargs=(data, lines),
#                               interval=50, blit=True)
#
# plt.show()





# !!! Vector animation
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation
#
# fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
#
# def get_arrow(theta):
#     x = np.cos(theta)
#     y = np.sin(theta)
#     z = 0
#     u = np.sin(2*theta)
#     v = np.sin(3*theta)
#     w = np.cos(3*theta)
#     return x,y,z,u,v,w
#
# quiver = ax.quiver(*get_arrow(0))
#
# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)
# ax.set_zlim(-2, 2)
#
# def update(theta):
#     global quiver
#     quiver.remove()
#     quiver = ax.quiver(*get_arrow(theta))
#
# ani = FuncAnimation(fig, update, frames=np.linspace(0,2*np.pi,200), interval=50)
# plt.show()

