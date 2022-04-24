import numpy as np

def Rx(q):
    T = np.array([[1,         0,          0, 0],
                  [0, np.cos(q), -np.sin(q), 0],
                  [0, np.sin(q),  np.cos(q), 0],
                  [0,         0,          0, 1]], dtype=float)
    return T


def Ry(q):
    T = np.array([[ np.cos(q), 0, np.sin(q), 0],
                  [         0, 1,         0, 0],
                  [-np.sin(q), 0, np.cos(q), 0],
                  [         0, 0,         0, 1]], dtype=float)
    return T


def Rz(q):
    T = np.array([[np.cos(q), -np.sin(q), 0, 0],
                  [np.sin(q),  np.cos(q), 0, 0],
                  [        0,          0, 1, 0],
                  [        0,          0, 0, 1]], dtype=float)
    return T


def Tx(x):
    T = np.array([[1, 0, 0, x],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    return T


def Ty(y):
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, y],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    return T


def Tz(z):
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]], dtype=float)
    return T


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

        
def quaternion_affine_matrix(Q, pos):
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2 #2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2 #2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 3 #2 * (q0 * q0 + q3 * q3) - 1
     
    # 4x4 affine matrix
    aff_matrix = np.array([[r00, r01, r02, pos[0]],
                           [r10, r11, r12, pos[1]],
                           [r20, r21, r22, pos[2]],
                           [0, 0, 0, 1]])
                            
    return aff_matrix


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 3
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22],
                           ])
                            
    return rot_matrix