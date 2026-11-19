"""Regression checks for frozen-feature actor adaptation."""

import unittest

import torch
from torch import nn

from restricted_actor import penultimate_features


class FeatureExtractionTest(unittest.TestCase):
    def test_shared_activations_reconstruct_full_actor(self):
        class Actor(nn.Module):
            def __init__(self):
                super().__init__()
                activation = nn.ELU()
                self.mlp = nn.Sequential(nn.Linear(3, 8), activation,
                                         nn.Linear(8, 8), activation, nn.Linear(8, 2))

            def get_latent(self, observations):
                return observations

        torch.manual_seed(0)
        actor = Actor()
        observations = torch.randn(32, 3)
        self.assertLess(len(list(actor.mlp.children())), len(list(actor.mlp)))
        reconstructed = actor.mlp[-1](penultimate_features(actor, observations))
        torch.testing.assert_close(reconstructed, actor.mlp(observations))


if __name__ == "__main__":
    unittest.main()
