import torch

import utils

class Agent:
    """An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, env_id, obs_space, model_dir, argmax=False, num_envs=1):
        _, self.preprocess_obss = utils.get_obss_preprocessor(env_id, obs_space, model_dir)
        self.acmodel = utils.load_model(model_dir)
        self.argmax = argmax
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size)

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)
        steer_dist, acc_dist = dist

        if self.argmax:
            steer_actions = steer_dist.probs.max(1, keepdim=True)[1]
            acc_actions = acc_dist.probs.max(1, keepdim=True)[1]
        else:
            steer_actions = steer_dist.sample()
            acc_actions = acc_dist.sample()

        if torch.cuda.is_available():
            steer_actions = steer_actions.cpu().numpy()
            acc_actions = acc_actions.cpu().numpy()

        actions = list(zip(steer_actions, acc_actions))
        return actions

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])