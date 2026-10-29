import torch
from d9d.core.dist_context import DistributedContext
from d9d.metric import Metric


class MockMetric(Metric[torch.Tensor]):
    def __init__(self):
        self.val = torch.tensor(0.0)
        self.sync_called = False
        self.reset_called = False
        self.compute_called = False
        self.current_device = None

    def update(self, v):
        self.val = torch.tensor(v, device=self.current_device)

    def sync(self, dist_context: DistributedContext):
        self.sync_called = True
        self.val.add_(1.0)

    def compute(self) -> torch.Tensor:
        self.compute_called = True
        return self.val * 2

    def reset(self):
        self.reset_called = True
        self.val.zero_()
        self.sync_called = False
        self.compute_called = False

    def to(self, device):
        self.current_device = device
        self.val = self.val.to(device)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
