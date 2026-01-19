import torch
from torch.distributions.categorical import Categorical

from YRC.core.policy import Policy


class ProcgenPolicy(Policy):
    def __init__(self, model):
        self.model = model

    def forward(self, obs):
        return self.model.get_logit(obs)

    def predict(self, obs):
        dist, value = self.model(obs)
        return dist

    def act(self, obs, greedy=False):
        dist = self.predict(obs)
        if greedy:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action.cpu().numpy()

    def get_hidden(self, obs):
        return self.model.get_hidden(obs)

    @property
    def hidden_dim(self):
        return self.model.hidden_dim


class EnsemblePolicy(Policy):
    """Wrapper that combines multiple policies into an ensemble.

    - forward(): returns mean logits across members
    - act(): samples from mean softmax distribution
    - predict(): returns distribution from mean softmax
    - get_hidden(): returns mean hidden states
    - ensemble_members: property exposing member list (for variance computation)
    """

    def __init__(self, members: list):
        self.members = members
        self._device = members[0].model.fc_policy.weight.device

    def forward(self, obs) -> torch.Tensor:
        """Return mean logits across ensemble members."""
        logits = [m.forward(obs) for m in self.members]
        return torch.stack(logits).mean(dim=0)

    def predict(self, obs):
        """Return distribution from mean softmax probabilities."""
        logits = [m.forward(obs) for m in self.members]
        stacked = torch.stack(logits)  # [M, B, A]
        mean_probs = torch.softmax(stacked, dim=-1).mean(dim=0)  # [B, A]
        return Categorical(probs=mean_probs)

    def act(self, obs, greedy=False):
        dist = self.predict(obs)
        if greedy:
            return dist.probs.argmax(dim=-1).cpu().numpy()
        return dist.sample().cpu().numpy()

    def get_hidden(self, obs):
        """Return mean hidden states across members."""
        hiddens = [m.get_hidden(obs) for m in self.members]
        return torch.stack(hiddens).mean(dim=0)

    @property
    def ensemble_members(self):
        return self.members

    @property
    def hidden_dim(self):
        return self.members[0].hidden_dim

    def eval(self):
        for m in self.members:
            m.eval()
        return self
