from copy import deepcopy
from src.recommenders.sequential_model import SequentialModel

class CheckpointMonitor:
    '''
        Associate this a SequentialRecommender while training, save two sets of weights
        one for avgNDCG another for minNDCG for a groups
    '''
    def __init__(self, config):
        self.config = config
        self.metrics = config.val_logging_metrics # [nDCG@10, R@10, nDCG@20, R@20]
        self.patience = config.max_epochs

        self.best_minmetric_modelweight = {'bestmin_'+str(k): None for k in self.metrics}
        # self.best_minmetric_value = {'bestmin_'+str(k): float("-inf") for k in self.metrics}
        self.best_minmetric_epoch = {'bestmin_'+str(k)+'_epoch': 0 for k in self.metrics}

        self.best_avgmetric_modelweight = {'bestavg_'+str(k): None for k in self.metrics}
        # self.best_avgmetric_value = {'bestavg_'+str(k): float("-inf") for k in self.metrics}
        self.best_avgmetric_epoch = {'bestavg_'+str(k)+'_epoch': 0 for k in self.metrics}

    def update_save_average(self, val_metrics, epoch, model):
        for metric in self.metrics:
            key = 'bestavg_' + str(metric)
            issetkey = key + '_set'
            if val_metrics[issetkey]:
                # self.best_avgmetric_value['bestavg_' + key] = val_metrics[key]
                self.best_avgmetric_epoch[key + '_epoch'] = epoch
                self.best_avgmetric_modelweight[key] = deepcopy(model.state_dict())
    
    def update_save_bestmin(self, val_metrics, epoch, model):
        for metric in self.metrics:
            key = 'bestmin_' + str(metric)
            issetkey = key + '_set'
            if val_metrics[issetkey]:
                # self.best_minmetric_value[key]  = val_metrics[key]
                self.best_minmetric_epoch[key + '_epoch']  = epoch
                self.best_minmetric_modelweight[key] = deepcopy(model.state_dict())
        
    def update_patience(self, epoch):
        espatience = self.config.early_stop_patience_epochs
        if self.config.early_stop_on == "none":
            self.patience = espatience # no change to patience, train for fixed number of epochs no ES
        elif self.config.early_stop_on == 'average':
            key = 'bestavg_' + str(self.config.val_checkpointing_metric) + '_epoch'
            self.patience = max(espatience - (epoch - self.best_avgmetric_epoch[key]), 0)
        elif self.config.early_stop_on == 'min':
            key = 'bestmin_' + str(self.config.val_checkpointing_metric) + '_epoch'
            self.patience = max(espatience - (epoch - self.best_minmetric_epoch[key]), 0)
        else:
            raise AttributeError
        
    def update(self, val_metrics:dict, epoch:int, model:SequentialModel):
        self.update_save_average(val_metrics, epoch, model)
        self.update_save_bestmin(val_metrics, epoch, model)
        self.update_patience(epoch)
    
    def get_bestepochs_patience(self):
        res = self.best_avgmetric_epoch | self.best_minmetric_epoch
        res['patience'] = self.patience # max of the earlier two
        return res
        
    # def update(self, val_metrics:dict, epoch:int, model:SequentialModel):
    #     val_avgmetric = val_metrics[str(self.config.val_checkpointing_metric)]
    #     val_minmetric = val_metrics['min_' + str(self.config.val_checkpointing_metric)]

    #     if val_avgmetric > self.best_avgval_metric:
    #         print(f"New best average {self.config.val_checkpointing_metric}. Updating best weights for avgNDCG checkpointer")
    #         self.best_weights_avgmetric = deepcopy(model.state_dict())
    #         self.best_avgval_metric = val_avgmetric
    #         self.best_epoch_avgmetric = epoch
        
    #     if val_minmetric > self.best_minval_metric:
    #         print(f"New best min {self.config.val_checkpointing_metric}. Updating best weights for minNDCG checkpointer")
    #         self.best_weights_minmetric = deepcopy(model.state_dict())
    #         self.best_minval_metric = val_minmetric
    #         self.best_epoch_minmetric = epoch
        
    #     espatience = self.config.early_stop_patience_epochs
    #     self.loss_patience = max(espatience - (epoch - self.best_loss_epoch), 0) # epoch-best_loss_epoch is number of iters without improvement for loss
    #     # metric patience
    #     self.avgval_patience = max(espatience - (epoch - self.best_epoch_avgmetric), 0)
    #     self.minval_patience = max(espatience - (epoch - self.best_epoch_minmetric), 0)
        
    #     # all three below are metric based patience
    #     if self.config.early_stop_on == 'none':
    #         self.patience = self.config.max_epochs
    #     elif self.config.early_stop_on == 'overall':
    #         self.patience = self.avgval_patience
    #     elif self.config.early_stop_on == 'ming':
    #         self.patience = self.minval_patience
    #     elif self.config.early_stop_on == 'both':
    #         self.patience = max(self.avgval_patience, self.minval_patience)
    #     else:
    #         raise NotImplementedError
    
    # def get_best_metrics(self):
    #     res = {}
    #     res['best_avg_' + str(self.config.val_checkpointing_metric)] = self.best_avgval_metric
    #     res['best_min_' + str(self.config.val_checkpointing_metric)] = self.best_minval_metric
    #     res['best_avgval_epoch'] = self.best_epoch_avgmetric
    #     res['best_minval_epoch'] = self.best_epoch_minmetric
    #     res['avgval_patience'] = self.avgval_patience
    #     res['minval_patience'] = self.minval_patience
    #     res['patience'] = self.patience # max of the earlier two
    #     return res