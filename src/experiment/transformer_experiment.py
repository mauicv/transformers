import torch


class TransformerExperiment():
    def __init__(
            self,
            loss_fn=None,
            opt=None,
            model=None,
        ):
        super().__init__()
        self.loss_fn = loss_fn
        self.opt = opt
        self.model = model

    def load_state(self, path):
        state = torch.load(path)
        self.loss_fn.load_state_dict(state['loss_fn'])
        self.opt.load_state_dict(state['opt'])

    def save_state(self, path):
        data = {
            'loss_fn': self.loss_fn.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, path)
