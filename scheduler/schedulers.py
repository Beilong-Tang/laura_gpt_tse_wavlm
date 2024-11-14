class NewBobScheduler:
    """Scheduler with new-bob technique, used for LR annealing.

    The learning rate is annealed based on the validation performance.
    In particular: if (past_loss-current_loss)/past_loss< impr_threshold:
    lr=lr * annealing_factor.

    Arguments
    ---------
    initial_value : float
        The initial hyperparameter value.
    annealing_factor : float
        It is annealing factor used in new_bob strategy.
    improvement_threshold : float
        It is the improvement rate between losses used to perform learning
        annealing in new_bob strategy.
    patient : int
        When the annealing condition is violated patient times,
        the learning rate is finally reduced.

    Example
    -------
    >>> scheduler = NewBobScheduler(initial_value=1.0)
    >>> scheduler(metric_value=10.0)
    (1.0, 1.0)
    >>> scheduler(metric_value=2.0)
    (1.0, 1.0)
    >>> scheduler(metric_value=2.5)
    (1.0, 0.5)
    """

    def __init__(
        self,
        initial_value,
        annealing_factor=0.5,
        improvement_threshold=0.0025,
        patient=0,
    ):
        self.hyperparam_value = initial_value
        self.annealing_factor = annealing_factor
        self.improvement_threshold = improvement_threshold
        self.patient = patient
        self.metric_values = []
        self.current_patient = self.patient

    def __call__(self, metric_value):
        """Returns the current and new value for the hyperparameter.
        This one is efficient for error measure that we want to minimize the error.
        CAUTION: If we want to minimize the loss but the loss can go negative, this might not work.

        Arguments
        ---------
        metric_value : int
            A number for determining whether to change the hyperparameter value.
        Returns
        -------
        Current and new hyperparam value.
        """
        old_value = new_value = self.hyperparam_value
        if len(self.metric_values) > 0:
            prev_metric = self.metric_values[-1]
            # Update value if improvement too small and patience is 0
            if prev_metric == 0:  # Prevent division by zero
                improvement = 0
            else:
                improvement = (prev_metric - metric_value) / prev_metric
            if improvement < self.improvement_threshold:
                if self.current_patient == 0:
                    new_value *= self.annealing_factor
                    self.current_patient = self.patient
                else:
                    self.current_patient -= 1

        # Store relevant info
        self.metric_values.append(metric_value)
        self.hyperparam_value = new_value

        return old_value, new_value

class PatienceScheduler:
    def __init__(self, optimizer, patience=3, factor=0.5, min_lr=1e-6, mode='min'):
        """
        Args:
            optimizer: The optimizer whose learning rate will be adjusted.
            patience: Number of epochs with no improvement after which learning rate will be reduced.
            factor: Factor by which the learning rate will be reduced (new_lr = lr * factor).
            min_lr: A lower bound on the learning rate.
            mode: One of {'min', 'max'}. If 'min', the scheduler will reduce the LR when the monitored
                  value has stopped decreasing. If 'max', it will reduce the LR when the monitored
                  value has stopped increasing.
        """
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.mode = mode
        self.best_value = None
        self.num_bad_epochs = 0
        self.last_epoch = -1

    def step(self, current_value):
        """
        Called after each epoch to update the learning rate based on the current validation performance.

        Args:
            current_value: The current epoch's performance (e.g., validation loss or accuracy).
        """
        if self.best_value is None:
            # Initialize the best value with the first value.
            self.best_value = current_value

        if self._is_better(current_value, self.best_value):
            # If the current performance is better, update best_value and reset bad epochs.
            self.best_value = current_value
            self.num_bad_epochs = 0
        else:
            # Otherwise, increment the number of bad epochs.
            self.num_bad_epochs += 1

        # If bad epochs exceed patience, reduce learning rate.
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0

    def _is_better(self, current_value, best_value):
        """
        Determines if the current performance is better than the best recorded performance.

        Args:
            current_value: Current epoch's performance (e.g., loss or accuracy).
            best_value: Best performance recorded so far.
        """
        if self.mode == 'min':
            return current_value < best_value
        elif self.mode == 'max':
            return current_value > best_value
        else:
            raise ValueError("mode should be either 'min' or 'max'.")

    def _reduce_lr(self):
        """
        Reduce the learning rate of the optimizer by the factor provided, ensuring it doesn't go below min_lr.
        """
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
        print(f"Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")