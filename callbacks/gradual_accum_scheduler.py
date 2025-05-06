import pytorch_lightning as pl

# here we will define a custom callback that handles gradual gradient accumulation
# at the start of the training loop. This is not natively defined in PyTorch or Lightning
class GradualAccumulationScheduler(pl.Callback):
    def __init__(self, target_accumulation, warmup_epochs):
        super().__init__()
        self.target_accumulation = target_accumulation
        self.warmup_epochs = warmup_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch < self.warmup_epochs:
            ratio = (current_epoch + 1) / self.warmup_epochs
            accum = max(1, int(self.target_accumulation * ratio))
        else:
            accum = self.target_accumulation

        trainer.accumulate_grad_batches = accum
