import numpy as np

class KalmanFilter:
   
    def __init__(self, init_x, init_v, init_acc, acc_variance):

        self.X_ = np.vstack([init_x, init_v])
        self.P_ = np.eye(16) 
        self.Acc_ = init_acc

        self.acc_variance_ = acc_variance
        self.prev_vel_ = init_v
        self.new_vel_ = init_v

    
    def predict(self, dt):
        # x = F x 
        # P = F P Ft + G Gt a

        
        F = np.eye(16)
        del_t = np.eye(8) * dt
        F[0:8, 8:16] = del_t

        B = np.zeros((16, 8))
        B[0:8, 0:8] = np.ones((8,8)) * 0.5 * dt **2
        B[8:16, 0:8] = np.ones((8,8)) * dt

        G1 =  np.ones((8,1)) * 0.5 * dt**2
        G2 =  np.ones((8,1)) * dt
        G = np.vstack([G1, G2])

        X_pred = np.dot(F, self.X_) + np.dot(B, self.Acc_)
        P_pred = np.dot(F, np.dot(self.P_, np.transpose(F))) + np.dot(G, np.transpose(G)) * self.acc_variance_

        self.X_ = X_pred
        self.P_ = P_pred
    
    def update(self, meas, meas_covar, dt):
        # y = -H x + z

        z = np.array([meas]).reshape(8,1)
        R = np.array([meas_covar]).reshape(8,8)

        H = np.zeros((8, 16)) 
        H[0:8, 0:8] = np.eye(8,8)
        # print("H shape ", H.shape)
        # print("X shape ", self.X_.shape)
        y =  - np.dot(H, self.X_) + z
        # print("y shape ", y.shape)

        S = H.dot(self.P_).dot(H.T) + R
        
        K = self.P_.dot(H.T).dot(np.linalg.inv(S))
        X_update = self.X_ + K.dot(y)
        P_update = (np.eye(16) - K.dot(H)).dot(self.P_)

        self.new_vel_ = self.velocity()
        self.Acc_ = 0.07 * (self.new_vel_ - self.prev_vel_) / dt
        print("acc is ",  self.Acc_)
        self.prev_vel_ =  self.new_vel_

        self.X_ = X_update
        self.P_ = P_update
  


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

# kf = KalmanFilter([1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8], 0)
# print(kf.position())
# print(kf.velocity())
# kf.predict(0.1)