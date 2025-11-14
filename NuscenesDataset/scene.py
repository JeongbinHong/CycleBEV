from utils.functions import *

class Scene:

    def __init__(self, sample_token=None, lidar_token_seq=None, agent_dict=None, city_name=None):

        self.sample_token = sample_token
        self.agent_dict = agent_dict
        self.lidar_token_seq = lidar_token_seq
        self.num_agents = len(agent_dict)
        self.city_name = city_name

    def make_id_2_token_lookup(self):
        self.id_2_token_lookup = {}
        for idx, key in enumerate(self.agent_dict):
            self.id_2_token_lookup[self.agent_dict[key].agent_id] = key

    def __repr__(self):
        return f"Sample ID: {self.sample_token}," \
               f" City: {self.city_name}," \
               f" Num agents: {self.num_agents}."
