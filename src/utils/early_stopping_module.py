import numpy as np
import torch
class EarlyStopping:
    
    def __init__(self, patience, delta, save_path):
        """
        patience: Number of epochs with not improved result which are tolerated before stop
        delta: Minimal improvement
        save_path: Path where improved model will be saved
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_avg_loss_min = np.inf

    def __call__(self, val_avg_loss, model):
        score = val_avg_loss

        if self.best_score is None:
            self.best_score = score   
            self.save_checkpoint(val_avg_loss, model)
        elif score > self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score   
            self.save_checkpoint(val_avg_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_avg_loss, model):
        '''Save model if there is decreasing loss in validation model.'''
        print(f'Validation loss decreasing {self.val_avg_loss_min:.5f} -> {val_avg_loss:.5f}.. Saving model...')
        torch.save(model.state_dict(), self.save_path)
        self.val_avg_loss_min = val_avg_loss