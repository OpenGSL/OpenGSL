class Recorder:
    def __init__(self, patience=100, criterion=None):
        self.patience = patience
        self.criterion = criterion
        self.best_loss = 100
        self.best_metric = 0
        self.wait = 0

    def add(self, loss_val, metric_val):
        flag = False
        if self.criterion is None:
            flag = True
        elif self.criterion == 'loss':
            flag = loss_val < self.best_loss
        elif self.criterion == 'metric':
            flag = metric_val > self.best_metric
        elif self.criterion == 'both':
            flag = loss_val < self.best_loss and metric_val > self.best_metric
        else:
            raise NotImplementedError

        if flag:
            self.best_metric = metric_val
            self.best_loss = loss_val
            self.wait = 0
        else:
            self.wait += 1

        flag_earlystop = self.patience and self.wait >= self.patience

        return flag, flag_earlystop