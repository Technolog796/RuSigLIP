# import math
# from torch.optim.lr_scheduler import LambdaLR
#
#
# class CosineDecayWithLinearWarmUp(LambdaLR):
#     def __init__(self, optimizer, warmup_steps, min_lr, max_lr, decay, last_epoch=-1):
#         super(optimizer, )
#         self.warmup_steps = warmup_steps
#         self.total_steps = total_steps
#         self.cycles = cycles
#         super(CosineWithLinearWarmupScheduler, self).__init__(optimizer, last_epoch)
#
#     def get_lr(self):
#         if self.last_epoch < self.warmup_steps:
#             return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
#         else:
#             progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
#             cosine_decay = 0.5 * (1 + math.cos(math.pi * self.cycles * 2 * progress))
#             return [base_lr * cosine_decay for base_lr in self.base_lrs]
#
#     def step(self, epoch=None):
#         if epoch is None:
#             epoch = self.last_epoch + 1
#         self.last_epoch = epoch
#         self.scheduler.last_epoch = epoch
#         self.scheduler.step(epoch)
#         self.warmup_scheduler.step(epoch)