from utils.functions import *
import math

class Agent:

    def __init__(self, type=None, attribute=None, track_id=None, agent_id=None, obs_len=None, pred_len=None):

        '''
        type : category
        attribute : attribute
        track_id : annotation_token
        agent_id : agent index in a scene
        obs_len : num positions in a past trajectory
        pred_len : num positions in a future trajectory
        '''


        self.type = type
        self.attribute = attribute
        self.track_id = track_id
        self.agent_id = agent_id

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.trajectory = np.full(shape=(obs_len+pred_len, 4), fill_value=np.nan)

        self.wlh = np.full(shape=(3), fill_value=np.nan)
        self.heading_traj = 0.0
        self.speed = 0.0
        self.yaw = 0.0


    def bbox_3d(self):

        w, l, h = self.wlh

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])

        return np.vstack((x_corners, y_corners, z_corners))

    def bbox(self):

        w, l, h = self.wlh
        x, y = 0, 0

        x_front = x + (l / 2)
        x_back = x - (l / 2)
        y_left = y + (w / 2)
        y_right = y - (w / 2)

        bbox = [[x_front, y_left], [x_back, y_left], [x_back, y_right], [x_front, y_right], [x_front, y]]

        return np.array(bbox)

    def calc_speed(self, sample_period):

        '''
        sample period (Hz)
        '''

        obs_seq = self.trajectory[:self.obs_len, 1:3]
        obs_pos_last = obs_seq[-1]
        obs_pos_last_m1 = obs_seq[-2]

        vec1 = obs_pos_last - obs_pos_last_m1
        self.speed = 3.6 * sample_period * np.sqrt(np.sum(vec1 ** 2))

    def heading_from_traj(self):

        obs_seq = self.trajectory[:self.obs_len, 1:3]
        obs_pos_last = obs_seq[-1]
        obs_pos_last_m1 = obs_seq[-2]

        vec1 = (obs_pos_last - obs_pos_last_m1).reshape(1, 2)
        vec2 = np.concatenate([np.ones(shape=(1, 1)), np.zeros(shape=(1, 1))], axis=1)

        x1 = vec1[:, 0]
        y1 = vec1[:, 1]
        x2 = vec2[:, 0]
        y2 = vec2[:, 1]

        dot = y1 * y2 + x1 * x2  # dot product
        det = y1 * x2 - x1 * y2  # determinant

        heading = np.arctan2(det, dot)  # -1x because of left side is POSITIVE
        if (y1 < 0):
            heading = 2 * math.pi - 1 * np.abs(heading)  # from 0 to 360 degrees

        if (self.track_id == 'EGO'):
            self.heading_traj = 0.0
        else:
            self.heading_traj = heading




    def __repr__(self):
        return self.type + '/' + self.track_id
