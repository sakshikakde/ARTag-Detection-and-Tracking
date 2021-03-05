import numpy as np

class KalmanFilter:
   
    def __init__(self, init_x, init_v, acc_variance):

        self.X_ = np.hstack([init_x, init_v])
        self.P_ = np.eye(16)
        self.acc_variance_ = acc_variance

    
    def predict(self, dt):
        # x = F x 
        # P = F P Ft + G Gt a
        F = np.eye(16)
        del_t = np.eye(8) * dt
        F[0:8, 8:16] = del_t

        G1 = np.ones((8,1)) * 0.5 * dt**2
        G2 = np.ones((8,1)) * dt
        G = np.vstack([G1, G2])

        X_pred = np.dot(F, self.X_)
        P_pred = np.dot(F, np.dot(self.P_, np.transpose(F))) + np.dot(G, np.transpose(G)) * self.acc_variance_

        self.X_ = X_pred
        self.P_ = P_pred
    
    def update(self, meas, meas_covar):
        # y = -H x + z

        z = np.array([meas]).reshape(8,1)
        R = np.array([meas_covar])

        H = np.zeros((8, 16)) 
        H[0:8, 0:8] = np.eye(8)

        y = - np.dot(H, self.X_) + z
        S = H.dot(self.P_).dot(H.T) + R

        K = self.P_.dot(H.T).dot(np.linalg.inv(S))
        new_x = self.X_ + K.dot(y)
        new_P = (np.eye(16) - K.dot(H)).dot(self.P_)

        self.P_ = new_P
        self.X_ = new_x


    def state(self):
        return self.X_

    def covar(self):
        return self.P_
    
    def position(self):
        pos = self.X_[0:8]
        return pos

    def velocity(self):
        vel = self.X_[8:16]
        return vel

kf = KalmanFilter([1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8], 0)
print(kf.position())
print(kf.velocity())
kf.predict(0.1)