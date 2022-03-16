# Planar drone
# robust classic control with torque optimization
# compared with robust qp control

from matplotlib.pyplot import *
from numpy.linalg import norm
from scipy.integrate import odeint
from scipy.optimize import linprog
from qpsolvers import solve_qp
from matrices import *
import matplotlib.animation as animation
import numpy as np
from collections import deque


def desired_trajectory(t, traj_params, period):
    a = traj_params['q_d']
    q_d = np.array([a[0], a[1], a[2]])
    dq_d = np.array([0, 0, 0])
    ddq_d = np.array([0, 0, 0])
    return q_d, dq_d, ddq_d


def D():
    m, I = system_params['m'], system_params['I']
    D = np.array([[m, 0, 0], [0, m, 0], [0, 0, I]])
    return D


def h():
    m, g = system_params['m'], system_params['g']
    h1 = 0
    h2 = m * g
    h3 = 0
    return np.array([h1, h2, h3])


def regressor(q, dq, ddq, g):
    ddr_x, ddr_y, ddalpha = ddq[:3]

    Y = np.zeros((3, 2))

    Y[0, 0] = ddr_x
    Y[1, 0] = ddr_y + g
    Y[2, 1] = ddalpha
    return Y


def system(x, t, u, sys_params):
    q, dq = x[:3], x[3:6]
    Dm = D()
    hm = h()
    ddq = np.dot(np.linalg.inv(Dm), u - 0.7 * hm)
    dx = np.concatenate([dq, ddq])
    return dx


def B(x):
    r = x[:3]
    d = system_params['d']
    c1 = (Rz(r[2]) @ Tx(r[0] - d / 2) @ Ty(r[1]))[0:3, 3]
    c2 = (Rz(r[2]) @ Tx(r[0] + d / 2) @ Ty(r[1]))[0:3, 3]
    s1 = c1 - r
    s2 = c2 - r
    # s1x = s1[0]
    # s1y = s1[1]
    # s2x = s2[0]
    # s2y = s2[1]
    B = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.], [-s1[1], s1[0], -s2[1], s2[0]]])
    return B


def qp_solve(Y, x, q_star, p, f_min, f_max, K, F_prev):
    z = np.dot(Y.T, q_star)  # 2x1
    gamma11 = 3.0
    gamma12 = 3.0
    gamma21 = 0.  # regulates smoothness of the force plot
    I = np.array([gamma11, gamma11, gamma11, gamma11, gamma11, gamma12])
    E = np.array([gamma21, gamma21, gamma21, gamma21, 0., 0.])
    M = np.append(F_prev, [0, 0])
    Bm = B(x)

    Am = np.hstack([Bm, -Y])

    P = .5 * np.dot(Am.T, Am) + np.diag(I) + np.diag(E)  # gamma*np.eye(6)  # this is a positive definite matrix
    Q = Y @ p + K @ q_star
    q = -np.dot(Q.T, Am) - M

    I = np.eye(4)

    mu = 0.7
    G = np.array([[-1, -mu, 0, 0, 0, 0],
                  [1, -mu, 0, 0, 0, 0],
                  [0, 0, -1, -mu, 0, 0],
                  [0, 0, 1, -mu, 0, 0],
                  [0, 0, 0, 0, -z[0], -z[1]]])

    # m, J
    p1_min = -0.1
    p1_max = 0.1

    p2_min = -0.005
    p2_max = 0.005

    p1_bounds = [p1_min, p1_max]
    p2_bounds = [p2_min, p2_max]

    zp = linprog(z,
                 bounds=[p1_bounds, p2_bounds]
                 ).fun  # minimizes

    s = np.array([Bm[2, 1], -Bm[2, 0]])  # s1x, s1y

    h = np.array([0, 0, 0, 0, -zp]).reshape((5,))
    A = None  # np.array([s[0], s[0], s[1], s[1], 0., 0.])
    b = None  # np.zeros(6)*0.

    X = solve_qp(P, q, G, h, A, b)  # X = [F, dp].T
    F = X[:4]
    u = Bm @ F

    return u, F


def classic_solve(z):
    eps = 0.9
    rho = 0.2
    if norm(z) > eps:
        delta_p = rho * z / norm(z)
    else:
        delta_p = rho * z / eps
    return delta_p


def compute_opt_control(u0, u_min, u_max):
    G = np.eye(3)
    q = np.array(-np.array(u0) @ G)
    C = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]])
    b = np.array([u_max[0], -u_min[0], u_max[1], -u_min[1], u_max[2], -u_min[2]])

    u_star = solve_qp(P=G, q=q, G=C, h=b)
    return u_star


def robust_control(x, t, q_d, dq_d, ddq_d, F_prev, controler_params, method):
    m, g, I = system_params['m'], system_params['g'], system_params['I']

    Lambda, K = control_params['K1'], control_params['K2']
    f_min, f_max = controler_params['f_min'], controler_params['f_max']

    q, dq = x[:3], x[3:6]

    e, de = q_d - q, dq_d - dq

    dq_s = dq_d + np.dot(Lambda, e)
    ddq_s = ddq_d + np.dot(Lambda, de)

    q_star = de + np.dot(Lambda, e)

    Y = regressor(q, dq_s, ddq_s, g)
    p = np.array([m, I])  # * 0.8
    # z = dot(Y.T, q_star)

    if method == 'robust_qp':
        u, F = qp_solve(Y, q, q_star, p, f_min, f_max, K, F_prev)
        return u, F
    elif method == 'robust_classic':
        dp = classic_solve(np.dot(Y.T, q_star))
        u = np.dot(Y, p + dp) + np.dot(K, q_star)
        return u, e[0], de[0]


def id_control(x, t, q_d, dq_d, ddq_d, control_params):
    q, dq = x[:3], x[3:6]

    K1, K2 = control_params['K1'], control_params['K2']

    e = q_d - q
    de = dq_d - dq

    v = ddq_d + np.dot(K1, e) + np.dot(K2, de)

    Dm = D()
    hm = h()
    u = Dm @ v + hm
    return u


def simulate_system(state_init, frequency, t_0, t_final, controller_params):
    dT = 1 / frequency
    t = np.arange(t_0, t_final, dT)
    t_star = np.linspace(0, dT, 10)
    per = 3
    theta_d_array = []
    u_min, u_max = controller_params['u_min'], controller_params['u_max']

    F_init = np.zeros(4)

    U, U_qp = [], []
    thetas, thetas_qp = [], []
    dthetas, dthetas_qp = [], []

    e_array, de_array = [], []

    # get desired
    # get actual
    # calc control
    # set control/simulate

    # classic robust control + torque optimization
    state_prev_opt = state_init
    F_prev = F_init
    for i in range(len(t)):
        t_curr = t[i]
        theta_d, dtheta_d, ddtheta_d = desired_trajectory(t_curr, trajectory_params, per)

        u, e, de = robust_control(state_prev_opt, t_curr, theta_d, dtheta_d, ddtheta_d, F_prev, control_params,
                                  'robust_classic')
        u_opt = compute_opt_control(u, u_min, u_max)
        state_opt = odeint(system, state_prev_opt, t_star,
                           args=(u_opt, system_params,))
        state_prev_opt = state_opt[-1]

        thetas.append(state_prev_opt[:3])
        # dthetas.append(state_prev_opt[3:6])
        U.append(u_opt)
        theta_d_array.append(theta_d)

        e_array.append(e)
        de_array.append(de)

    # QP robust control
    F_qp = []
    state_prev_qp = state_init
    for i in range(len(t)):
        t_curr = t[i]
        theta_d, dtheta_d, ddtheta_d = desired_trajectory(t_curr, trajectory_params, per)

        u_qp, F = robust_control(state_prev_qp, t_curr, theta_d, dtheta_d, ddtheta_d, F_prev, control_params,
                                 'robust_qp')
        state_qp = odeint(system, state_prev_qp, t_star,
                          args=(u_qp, system_params,))
        state_prev_qp = state_qp[-1]
        F_prev = F
        thetas_qp.append(state_prev_qp[:3])
        # dthetas_qp.append(state_prev_qp[3:6])
        U_qp.append(u_qp)
        F_qp.append(F)

    label1 = r'$x$ Robust + TO'
    label2 = r'$y$ Robust + TO'
    label3 = r'$x$ Robust_QP'
    label4 = r'$y$ Robust_QP'
    label5 = r'$\alpha$ Robust + TO'
    label6 = r'$\alpha$ Robust_QP'

    # POSITION PLOT
    figure()
    plot(t, np.array(thetas)[:, 0], color='r', linewidth=2., linestyle=':', label=label1)
    plot(t, np.array(thetas)[:, 1], color='b', linewidth=2., linestyle=':', label=label2)

    plot(t, np.array(thetas_qp)[:, 0], color='r', linewidth=2., label=label3)
    plot(t, np.array(thetas_qp)[:, 1], color='b', linewidth=2., label=label4)

    plot(t, np.array(theta_d_array)[:, 0], color='r', linewidth=2., linestyle='--', label='$x$ desired')
    plot(t, np.array(theta_d_array)[:, 1], color='b', linewidth=2., linestyle='--', label='$y$ desired')
    grid(color='k', linestyle='--', alpha=0.7)
    xlim([t_0, t_final])

    ylabel('Position $r$ (m)')
    xlabel('Time $t$ (s)')
    legend()
    savefig('2Ddrone/position.png')

    # ANGLE PLOT
    figure()
    plot(t, np.array(thetas)[:, 2], color='g', linewidth=2., linestyle=':', label=label5)
    plot(t, np.array(thetas_qp)[:, 2], color='g', linewidth=2., label=label6)
    plot(t, np.array(theta_d_array)[:, 2], color='g', linewidth=2., linestyle='--', label=r'$\alpha$ desired')
    grid(color='k', linestyle='--', alpha=0.7)
    xlim([t_0, t_final])

    ylabel(r'Angle ${\alpha}$ (rad)')
    xlabel(r'Time $t$ (s)')
    legend()
    savefig('2Ddrone/angle.png')

    # CONTROL PLOT
    figure()
    text = ['$U_x$ Robust + TO', '$U_y$ Robust + TO', r'$U_\alpha$ Robust + TO']
    color = ['r', 'b', 'g']
    for i in range(3):
        step(t, np.array(U)[:, i], color=str(color[i]), linewidth=2., linestyle=':', label=str(text[i]))

    text = ['$U_x$ Robust_QP', '$U_y$ Robust_QP', r'$U_\alpha$ Robust_QP']
    for i in range(3):
        step(t, np.array(U_qp)[:, i], color=str(color[i]), linewidth=2., label=str(text[i]))

    grid(color='black', linestyle='--', alpha=0.7)
    xlim([t_0, t_final])

    ylabel('Control $U$')
    xlabel('Time $t$ (s)')
    legend(loc='upper right')
    savefig('2Ddrone/control.png')

    # FORCE CONTROL PLOT
    F1x = np.array(F_qp)[:, 0]
    F1y = np.array(F_qp)[:, 1]
    F2x = np.array(F_qp)[:, 2]
    F2y = np.array(F_qp)[:, 3]

    figure()
    step(t, F1x, linewidth=2.0, color='r', label='$F_{1x}$')
    step(t, F1y, linewidth=2.0, color='r', linestyle='--', label='$F_{1y}$')

    grid(color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    xlim([t_0, t_final])

    ylabel('Force $F$')
    xlabel('Time $t$ (s)')
    legend(loc='upper right')
    savefig('2Ddrone/F1.png')

    figure()
    step(t, F2x, linewidth=2.0, color='b', label='$F_{2x}$')
    step(t, F2y, linewidth=2.0, color='b', linestyle='--', label='$F_{2y}$')

    grid(color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    xlim([t_0, t_final])

    ylabel('Force $F$')
    xlabel('Time $t$ (s)')
    legend()
    savefig('2Ddrone/F2.png')

    # show()
    close('all')

    # ANIMATION
    # get position x, position y, angle, forces
    thisx_array = np.array(thetas_qp)[::8, 0]
    thisy_array = np.array(thetas_qp)[::8, 1]
    thisalpha_array = np.array(thetas_qp)[::8, 2]

    F1x = np.array(F_qp)[::8, 0]
    F1y = np.array(F_qp)[::8, 1]
    F2x = np.array(F_qp)[::8, 2]
    F2y = np.array(F_qp)[::8, 3]

    history_x, history_y = deque(maxlen=len(thisx_array)), deque(maxlen=len(thisx_array))
    target_x, target_y = deque(maxlen=len(thisx_array)), deque(maxlen=len(thisx_array))

    t = np.arange(t_0, t_final, dT)

    def animate(i):
        global Q1, Q2
        xc, yc = thisx_array[i], thisy_array[i]
        alpha_c = thisalpha_array[i]
        L = system_params['d']
        H = Rz(alpha_c) @ Tx(xc - L/2) @ Ty(yc)
        x1, y1 = H[0][3], H[1][3]

        H = Rz(alpha_c) @ Tx(xc + L/2) @ Ty(yc)
        x2, y2 = H[0][3], H[1][3]

        thisx = [x1, x2]
        thisy = [y1, y2]

        f1x = [x1, x1 + 0.5 * F1x[i] / np.linalg.norm(np.array([F1x[i], F1y[i]]))]
        f1y = [y1, y1 + 0.5 * F1y[i] / np.linalg.norm(np.array([F1x[i], F1y[i]]))]

        f2x = [x2, x2 + 0.5 * F2x[i] / np.linalg.norm(np.array([F2x[i], F2y[i]]))]
        f2y = [y2, y2 + 0.5 * F2y[i] / np.linalg.norm(np.array([F2x[i], F2y[i]]))]

        if i == 0:
            history_x.clear()
            history_y.clear()

        history_x.appendleft(xc)
        history_y.appendleft(yc)

        target = trajectory_params['q_d'][:2]
        target_x.appendleft(target[0])
        target_y.appendleft(target[1])
        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        star.set_data(target_x, target_y)
        f1_vec.set_data(f1x, f1y)
        f2_vec.set_data(f2x, f2y)

        # global quiver
        # quiver = ax.quiver(*get_arrow(alpha_c))

        # beta1 = np.arctan2(F1x[i], F1y[i])
        # beta2 = np.arctan2(F2x[i], F2y[i])

        # Q1.remove()
        # Q2.remove()
        # Q1 = ax.quiver(x1, y1, np.cos(beta1), np.sin(beta1), color='b')
        # Q2 = ax.quiver(x2, y2, np.cos(beta2), np.sin(beta2), color='b')
        # print(beta1, beta2)
        # Q.set_UVC(u, v)

        ax.set_title('$t$ = ' + str("{:0.3f}".format(t[i])))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        return line, trace, star, f1_vec, f2_vec

    fig = figure()
    ax = subplot(autoscale_on=False
                 , xlim=(-1, 3), ylim=(-1, 2)
                 )
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    trace, = ax.plot([], [], '--', ms=1)
    star, = ax.plot([], [], 'r+', lw=4, ms=4)
    f1_vec, = ax.plot([], [], '^-', lw=2, color='b', alpha=0.4)
    f2_vec, = ax.plot([], [], '^-', lw=2, color='b', alpha=0.4)

    ani = animation.FuncAnimation(
        fig, animate, frames=len(thisy_array),
        # init_func=init_my_quiver,
        interval=dT * 1e4)

    # writer = animation.PillowWriter(fps=100)
    # ani.save('2Ddrone/animation.gif', writer=writer)

    show()


t0 = 0
tf = 5.0
freq = 100


# parameters
d = 1.
m = 1.
I = 0.05
g = 9.81

K1 = np.diag([3, 8, 3])
K2 = np.diag([3, 5, 3])

f_min = -20.
f_max = 20.

u_min = [-10.0, -10.0, -5.0]
u_max = [15.0, 15.0, 5.0]

system_params = {'d': d,  # size
                 'm': m,
                 'I': I,
                 'g': g}

control_params = {'K1': K1, 'K2': K2,
                  'u_min': u_min, 'u_max': u_max,
                  'f_min': f_min, 'f_max': f_max}

x0 = [0.1, 0.6, 0.27, 0, 0, 0]
trajectory_params = {'q_d': [1.2, 0.9, 0., 0., 0., 0.]}  # desired position

simulate_system(state_init=x0, frequency=freq, t_0=t0, t_final=tf,
                controller_params=control_params)