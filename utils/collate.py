from utils.functions import *

def seq_collate_VEHICLE(data):

    obs_traj, future_traj, obs_traj_a, future_traj_a, hdmap_img, R_map, R_traj, num_agents, bev, images, \
        intrinsics, extrinsics = zip(*data)


    _len = [objs for objs in num_agents]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # trajectory related tensors
    obs_traj_cat = torch.cat(obs_traj, dim=1)
    future_traj_cat = torch.cat(future_traj, dim=1)
    obs_traj_cat_a = torch.cat(obs_traj_a, dim=1)
    future_traj_cat_a = torch.cat(future_traj_a, dim=1)
    hdmap_img = torch.stack(hdmap_img, dim=0)
    R_map_cat = torch.cat(R_map, dim=0)
    R_traj_cat = torch.cat(R_traj, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    # topview related tensors
    bev_cat = torch.cat(bev, dim=0)
    if (images[0] is not None):
        intrinsics_cat = torch.cat(intrinsics, dim=0)
        extrinsics_cat = torch.cat(extrinsics, dim=0)
        images_cat = torch.cat(images, dim=0)
    else:
        images_cat, intrinsics_cat, extrinsics_cat = None, None, None

    data = {'obs': obs_traj_cat,
            'fut': future_traj_cat,
            'obs_a': obs_traj_cat_a,
            'fut_a': future_traj_cat_a,
            'hdmap_img' : hdmap_img,
            'Rm': R_map_cat,
            'Rt': R_traj_cat,
            'seq_start_end': seq_start_end,
            'num_agents': num_agents,
            'bev': bev_cat,
            'intrinsics': intrinsics_cat,
            'extrinsics': extrinsics_cat,
            'image': images_cat}

    return data
