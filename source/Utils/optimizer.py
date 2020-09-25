import torch


# http://nlp.seas.harvard.edu/2018/04/03/attention.html#a-first--example
# Optimizer
class NoamOpt:
    """
    During warmup:
        learning rate = factor * sqrt(model_size) *  sqrt(step_num)

    After warmup:
        decay_factor = factor * sqrt(model_size) * warmup **(-1.5)
        learning rate = decay_factor * step_num
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


class InverseSqrt:
    """
    During warmup::
      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::
      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(step)
    """

    def __init__(self, warmup, optimizer, warmup_init_lr=1e-07, warmup_end_lr=0.0005):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.warmup_init_lr = warmup_init_lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup
        self.decay_factor = warmup_end_lr * warmup ** 0.5

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def rate(self):
        "Implement `lrate` above"
        step = self._step
        if step < self.warmup:
            return self.warmup_init_lr + step * self.lr_step
        else:
            return self.decay_factor * step ** (-0.5)