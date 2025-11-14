import warnings
warnings.filterwarnings("ignore")
from utils.functions import *
from shapely import affinity
import colorsys
from shapely.geometry import MultiPolygon
from NuscenesDataset.nuscenes.map_expansion.arcline_path_utils import discretize_lane
from NuscenesDataset.nuscenes.utils.geometry_utils import transform_matrix
from NuscenesDataset.nuscenes.map_expansion.map_api import NuScenesMap

import NuscenesDataset.nuscenes.utils.data_classes as dc
import pyquaternion

class Map:

    def __init__(self, dataset_dir, nusc):

        self.nusc = nusc

        # Nuscenes Map loader
        self.nusc_maps = {}
        self.nusc_maps['singapore-onenorth'] = NuScenesMap(dataroot=dataset_dir, map_name='singapore-onenorth')
        self.nusc_maps['singapore-hollandvillage'] = NuScenesMap(dataroot=dataset_dir, map_name='singapore-hollandvillage')
        self.nusc_maps['singapore-queenstown'] = NuScenesMap(dataroot=dataset_dir, map_name='singapore-queenstown')
        self.nusc_maps['boston-seaport'] = NuScenesMap(dataroot=dataset_dir, map_name='boston-seaport')

        self.centerlines = {}
        self.centerlines['singapore-onenorth'] = self.nusc_maps['singapore-onenorth'].discretize_centerlines(resolution_meters=0.25)
        self.centerlines['singapore-hollandvillage'] = self.nusc_maps['singapore-hollandvillage'].discretize_centerlines(resolution_meters=0.25)
        self.centerlines['singapore-queenstown'] = self.nusc_maps['singapore-queenstown'].discretize_centerlines(resolution_meters=0.25)
        self.centerlines['boston-seaport'] = self.nusc_maps['boston-seaport'].discretize_centerlines(resolution_meters=0.25)

        # control params
        self.centerline_width = 1


    def transform_pc(self, R, pc):
        pc = np.matmul(R, np.concatenate([pc, np.ones(shape=(1, pc.shape[1]))], axis=0))[:3]
        return pc

    # def make_topview_map_loadertypeD(self, ego_pose, scene_location, x_range, y_range, map_size, obs_traj, category):
    def make_topview_map(self, scene, x_range, y_range, map_size, obs_traj, agent_id, category):

        scene_location = scene.city_name
        lidar_now_token = scene.lidar_token_seq[obs_traj.shape[0]-1]
        lidar_now_data = self.nusc.get('sample_data', lidar_now_token)
        ego_pose = self.nusc.get('ego_pose', lidar_now_data['ego_pose_token'])

        # ego-pose
        xyz = ego_pose['translation']
        R = transform_matrix(ego_pose['translation'], pyquaternion.Quaternion(ego_pose['rotation']), inverse=False)
        Rinv = transform_matrix(ego_pose['translation'], pyquaternion.Quaternion(ego_pose['rotation']), inverse=True)
        v = np.dot(R[:3, :3], np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])

        # map_size x map_size x 1, 0~255 (uint8)
        img_pedcross = (64.0 * self.draw_others(xyz, yaw, x_range, y_range, map_size, scene_location,
                                                'ped_crossing')).astype('uint8')

        # debug, 230713
        # road segment (map_size x map_size x 2)
        # img_roadseg = self.draw_road_segment(xyz, yaw, x_range, y_range, map_size, scene_location)


        # map_size x map_size x 3, 0~255 (float)
        img_centerlines = np.concatenate([img_pedcross, img_pedcross, img_pedcross], axis=2)
        img_centerlines = self.draw_centerlines(img_centerlines, Rinv, xyz, x_range, y_range, map_size,
                                                scene_location).astype('float')

        agent_dicts = []
        for _, id in enumerate(agent_id[0]):
            token = scene.id_2_token_lookup[id]
            agent_dicts.append(scene.agent_dict[token])

        # map_size x map_size x 1, 0~255 (float)
        img_traj = self.draw_agent_trajectories(obs_traj, x_range, y_range, map_size, agent_dicts, category)[:, :, 0].reshape(
            map_size, map_size, 1)

        # cv2.imshow("", img_centerlines.astype('uint8'))
        # cv2.waitKey(0)

        # map_size x map_size x 4, 0~255 (float)
        img = np.concatenate([img_centerlines, img_traj], axis=2)

        # img = np.sum(img, axis=2).reshape(map_size, map_size, 1)
        # cv2.imshow("", img.astype('uint8'))
        # cv2.waitKey(0)


        return img.astype('float')/255.0

    def draw_centerlines(self, img, Rinv, xyz, x_range, y_range, map_size, scene_location):

        # global coord.
        pose_lists = self.centerlines[scene_location]

        w_x_min = xyz[0] + x_range[0] - 10
        w_x_max = xyz[0] + x_range[1] + 10
        w_y_min = xyz[1] + y_range[0] - 10
        w_y_max = xyz[1] + y_range[1] + 10
        win_min_max = (w_x_min, w_y_min, w_x_max, w_y_max)

        for l in range(len(pose_lists)):

            cur_lane = pose_lists[l]
            l_x_max = np.max(cur_lane[:, 0])
            l_x_min = np.min(cur_lane[:, 0])
            l_y_max = np.max(cur_lane[:, 1])
            l_y_min = np.min(cur_lane[:, 1])

            lane_min_max = (l_x_min, l_y_min, l_x_max, l_y_max)

            if (correspondance_check(win_min_max, lane_min_max) == True):
                # global to agent-centric
                cur_lane = self.transform_pc(Rinv, cur_lane.T).T

                # draw
                img = self.draw_lines_on_topview_with_coloryaw(img, cur_lane[:, :2], x_range, y_range, map_size=map_size)

        # cv2.imshow("", img.astype('uint8'))
        # cv2.waitKey(0)

        return img.astype('float')

    def draw_agent_trajectories(self, obs_traj, x_range, y_range, map_size, agent_dicts, category):

        def to_global(pos, R, T):
            '''
            pos : N x 2
            R : 2 x 2
            T : 1 x 2
            '''
            return np.matmul(R, pos.T).T + T

        def to_agent(pos, R, T):
            '''
            pos : N x 2
            R : 2 x 2
            T : 1 x 2
            '''
            return np.matmul(R, (pos-T).T).T

        # # calc. agent bbox in SDV coordinate system at current time
        # yaw, trans_g = None, None
        # for _, agent in enumerate(agent_dicts):
        #     if (agent.track_id == 'EGO'):
        #         yaw = agent.yaw_global[obs_traj.shape[0]-1]
        #         trans_e = agent.trajectory_global_coord[obs_traj.shape[0] - 1, 1:3].reshape(1, 2)
        # R_e2g = rotation_matrix(yaw)
        # R_g2e = np.linalg.inv(R_e2g)

        seq_len, batch, dim = obs_traj.shape
        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        scale_y = float(map_size - 1) / axis_range_y
        scale_x = float(map_size - 1) / axis_range_x

        img = np.zeros(shape=(map_size, map_size, 3))
        for b in range(batch):

            if (category[b] == 0): circle_size = 5
            else: circle_size = 2

            col_pels = -(obs_traj[:, b, 1] * scale_y).astype(np.int32)
            row_pels = -(obs_traj[:, b, 0] * scale_x).astype(np.int32)

            col_pels += int(np.trunc(y_range[1] * scale_y))
            row_pels += int(np.trunc(x_range[1] * scale_x))

            for j in range(0, seq_len):

                if (np.isnan(col_pels[j])):
                    continue

                brightness = int(255.0 * float(j+1) / float(seq_len+1))
                color = (brightness, brightness, brightness)
                img = cv2.circle(img, (col_pels[j], row_pels[j]), circle_size, color, -1)

        # TODO: experiment with bbox image
        # # draw agent bbox
        # brightness = int(255.0 * float(seq_len) / float(seq_len + 1))
        # color = (brightness, brightness, brightness)
        # for b in range(batch):
        #     bbox = agent_dicts[b].bbox()    # bbox in agent-centric coord. system
        #
        #     if (agent_dicts[b].track_id == 'EGO'):
        #         continue
        #
        #     if (bbox is None):
        #         continue
        #
        #     R_a2g = rotation_matrix(agent_dicts[b].yaw_global[obs_traj.shape[0]-1])
        #     trans = agent_dicts[b].trajectory_global_coord[obs_traj.shape[0] - 1, 1:3].reshape(1, 2)
        #     bbox_g = to_global(bbox, R_a2g, trans)
        #     bbox_e = to_agent(bbox_g, R_g2e, trans_e)
        #
        #
        #     # to topview image domain
        #     col_pels = -(bbox_e[:, 1] * scale_y).astype(np.int32)
        #     row_pels = -(bbox_e[:, 0] * scale_x).astype(np.int32)
        #
        #     col_pels += int(np.trunc(y_range[1] * scale_y))
        #     row_pels += int(np.trunc(x_range[1] * scale_x))
        #
        #     cv2.line(img, (col_pels[0], row_pels[0]), (col_pels[1], row_pels[1]), color, 1)
        #     cv2.line(img, (col_pels[1], row_pels[1]), (col_pels[2], row_pels[2]), color, 1)
        #     cv2.line(img, (col_pels[2], row_pels[2]), (col_pels[3], row_pels[3]), color, 1)
        #     cv2.line(img, (col_pels[3], row_pels[3]), (col_pels[0], row_pels[0]), color, 1)

        return img

    def draw_lines_on_topview_with_coloryaw(self, img, line, x_range, y_range, map_size):

        diff = line[1:] - line[:-1]
        line_yaws = calc_yaw_from_points(diff) + np.pi # 0 to 2*pi
        line_angle_deg = line_yaws * 180 / np.pi

        # debug ---
        chk0 = line_angle_deg > 360
        chk1 = line_angle_deg < 0
        assert (np.count_nonzero(chk0) == 0)
        assert (np.count_nonzero(chk1) == 0)
        # debug ---

        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        scale_y = float(map_size - 1) / axis_range_y
        scale_x = float(map_size - 1) / axis_range_x

        col_pels = -(line[:, 1] * scale_y).astype(np.int32)
        row_pels = -(line[:, 0] * scale_x).astype(np.int32)

        col_pels += int(np.trunc(y_range[1] * scale_y))
        row_pels += int(np.trunc(x_range[1] * scale_x))

        for j in range(1, line.shape[0]):

            rgb_n = colorsys.hsv_to_rgb(line_angle_deg[j-1] / 360, 1., 1.)
            color = (int(255*rgb_n[2]), int(255*rgb_n[1]), int(255*rgb_n[0]))

            start = (col_pels[j], row_pels[j])
            end = (col_pels[j-1], row_pels[j-1])
            cv2.line(img, start, end, color, self.centerline_width)

        return img

    def draw_others(self, xyz, yaw, x_range, y_range, map_size, scene_location, layer_name):

        patch_box = (xyz[0], xyz[1], x_range[1]-x_range[0], y_range[1]-y_range[0])
        patch_angle = np.rad2deg(yaw) + 90
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        target_patch = self.nusc_maps[scene_location].explorer.get_patch_coord(patch_box, patch_angle)

        records = getattr(self.nusc_maps[scene_location], layer_name)

        polygon_list = []
        for record in records:
            polygon = self.nusc_maps[scene_location].extract_polygon(record['polygon_token'])

            if polygon.is_valid:
                new_polygon = polygon.intersection(target_patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                  origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])

                    # if (layer_name == 'stop_line'):
                    #     if (record['stop_line_type'] == 'TRAFFIC_LIGHT'):
                    #         polygon_list.append(new_polygon)
                    # else:
                    #     polygon_list.append(new_polygon)

                    if (layer_name == 'stop_line'):
                        if (record['stop_line_type'] == 'TRAFFIC_LIGHT' and len(record['traffic_light_tokens']) > 0):
                            polygon_list.append(new_polygon)
                    else:
                        polygon_list.append(new_polygon)

        local_box = (0.0, 0.0, patch_box[2], patch_box[3])
        canvas_size = (map_size, map_size)
        map_mask = self.nusc_maps[scene_location].explorer._polygon_geom_to_mask(polygon_list, local_box, layer_name, canvas_size)
        return np.fliplr(map_mask[:, :].reshape(map_size, map_size, 1))

    def draw_road_segment(self, xyz, yaw, x_range, y_range, map_size, scene_location):

        patch_box = (xyz[0], xyz[1], x_range[1]-x_range[0], y_range[1]-y_range[0])
        patch_angle = np.rad2deg(yaw) + 90
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        target_patch = self.nusc_maps[scene_location].explorer.get_patch_coord(patch_box, patch_angle)

        layer_name = 'road_segment'
        # layer_name = 'lane_divider'
        records = getattr(self.nusc_maps[scene_location], layer_name)

        polygon_list = []
        polygon_list_intersection = []
        for record in records:
            polygon = self.nusc_maps[scene_location].extract_polygon(record['polygon_token'])

            if polygon.is_valid:
                new_polygon = polygon.intersection(target_patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                  origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])

                    if (record['is_intersection']):
                        polygon_list_intersection.append(new_polygon)
                    else:
                        polygon_list.append(new_polygon)

        local_box = (0.0, 0.0, patch_box[2], patch_box[3])
        canvas_size = (map_size, map_size)
        map_mask = self.nusc_maps[scene_location].explorer._polygon_geom_to_mask(polygon_list, local_box, layer_name, canvas_size)
        map_mask_intersection = self.nusc_maps[scene_location].explorer._polygon_geom_to_mask(polygon_list_intersection, local_box, layer_name, canvas_size)

        map_mask = np.fliplr(map_mask[:, :].reshape(map_size, map_size, 1))
        map_mask_intersection = np.fliplr(map_mask_intersection[:, :].reshape(map_size, map_size, 1))

        return np.concatenate([map_mask, map_mask_intersection], axis=2)


    def __repr__(self):
        return f"Nuscenes Map Helper."


def in_range_points(points, x, y, z, x_range, y_range, z_range):

    points_select = points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], y < y_range[1], z > z_range[0], z < z_range[1]))]
    return np.around(points_select, decimals=2)


def correspondance_check(win_min_max, lane_min_max):

    # four points for window and lane box
    w_x_min, w_y_min, w_x_max, w_y_max = win_min_max
    l_x_min, l_y_min, l_x_max, l_y_max = lane_min_max

    w_TL = (w_x_min, w_y_max)  # l1
    w_BR = (w_x_max, w_y_min)  # r1

    l_TL = (l_x_min, l_y_max)  # l2
    l_BR = (l_x_max, l_y_min)  # r2

    # If one rectangle is on left side of other
    # if (l1.x > r2.x | | l2.x > r1.x)
    if (w_TL[0] > l_BR[0] or l_TL[0] > w_BR[0]):
        return False

    # If one rectangle is above other
    # if (l1.y < r2.y || l2.y < r1.y)
    if (w_TL[1] < l_BR[1] or l_TL[1] < w_BR[1]):
        return False

    return True

def calc_yaw_from_points(vec1):

    '''
    vec : seq_len x 2
    '''

    seq_len = vec1.shape[0]

    vec1 = vec1.reshape(seq_len, 2)
    vec2 = np.repeat(np.concatenate([np.ones(shape=(1, 1)), np.zeros(shape=(1, 1))], axis=1), seq_len, 0)

    x1 = vec1[:, 0]
    y1 = vec1[:, 1]
    x2 = vec2[:, 0]
    y2 = vec2[:, 1]

    dot = y1 * y2 + x1 * x2  # dot product
    det = y1 * x2 - x1 * y2  # determinant

    heading = np.arctan2(det, dot)  # -1x because of left side is POSITIVE

    return heading


def rotation_matrix(heading):

    m_cos = np.cos(heading)
    m_sin = np.sin(heading)
    m_R = np.array([m_cos, -1 * m_sin, m_sin, m_cos]).reshape(2, 2)
    return m_R