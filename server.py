# server.py
import numpy as np
from utils import aggregate_feature_vectors

class Server:
    def __init__(self):
        self.global_features = None

    def receive(self, client_vectors):
        self.global_features = aggregate_feature_vectors(client_vectors)

    def send_global(self):
        return self.global_features
