class Recorder:
    """
    Recorder Class.

    This records the performances of epochs in a single run. It determines whether the training has improved based
    on the provided `criterion` and determines whether the earlystop `patience` has been achieved.

    Parameters
    ----------
    patience : int
        The maximum epochs to keep training since last improvement.
    criterion : str
        The criterion to determine whether whether the training has improvement.
        - ``None``: Improvement will be considered achieved in any case.
        - ``loss``: Improvement will be considered achieved when loss decreases.
        - ``metric``: Improvement will be considered achieved when metric increases.
        - ``either``: Improvement will be considered achieved if either loss decreases or metric increases.
        - ``both``: Improvement will be considered achieved if both loss decreases and metric increases.
    """
    def __init__(self, patience=100, criterion=None):
        self.patience = patience
        self.criterion = criterion
        self.best_loss = 1e8
        self.best_metric = -1
        self.wait = 0

    def add(self, loss_val, metric_val):
        '''
        Function to add the loss and metric of a new epoch.

        Parameters
        ----------
        loss_val : float
        metric_val : float

        Returns
        -------
        flag : bool
            Whether improvement has been achieved in the epoch.
        flag_earlystop: bool
            Whether training needs earlystopping.
        '''
        flag = False
        if self.criterion is None:
            flag = True
        elif self.criterion == 'loss':
            flag = loss_val < self.best_loss
        elif self.criterion == 'metric':
            flag = metric_val > self.best_metric
        elif self.criterion == 'either':
            flag = loss_val < self.best_loss or metric_val > self.best_metric
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