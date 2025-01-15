from src.utils.ir_measures_converters import get_irmeasures_qrels, get_irmeasures_run
import pandas as pd
import ir_measures
from ir_measures import nDCG, R
import random
from collections import defaultdict

class MetricComputer(): #df_userid_groups can have intersecting groups here
    def __init__(self, data:pd.DataFrame, df_userid_groups:pd.DataFrame, metrics:list = [nDCG@20, nDCG@10, R@10, R@20]):
        '''
            data is train, val, test... with no group columns
            df_userid_groups is a pandas dataframe of shape (#users_ids, #groups), this mapping is constructed only from train data sequences
        '''
        self.df = data.merge(df_userid_groups, on='user_id') # self.df has the <user_id, item_id and groups indicator cols>
        self.metrics = metrics
        self.subgroups = df_userid_groups.attrs['sglist']
        self.group_cols = df_userid_groups.attrs['glist']
        self.evaluator = self.get_full_data_evaulator() # used for overall metric
        self.g_eval_dict = self.get_group_evaluators_dict() # used for group metrics
        self.results = {}

    def get_full_data_evaulator(self):
        self.fullqrels = get_irmeasures_qrels(self.df)
        return ir_measures.evaluator(self.metrics, self.fullqrels)

    def get_group_evaluators_dict(self) -> dict:
        def get_group_qrels() -> dict[str, pd.DataFrame]:
            qrel_dict = {}
            for gname in self.group_cols:
                qrel_dict[gname] = get_irmeasures_qrels(self.df[self.df[gname] == 1])
            return qrel_dict
        self.qrel_gdict = get_group_qrels()
        eval_dict = {}
        for k, v in self.qrel_gdict.items():
            eval_dict[k] = ir_measures.evaluator(self.metrics, v)
        return eval_dict

    def update_metrics(self, recs:list):
        # def get_results_gdict(run):
        #     results_gdict = {}
        #     for k, evaluator in self.g_eval_dict.items():
        #         results_gdict[k] = evaluator.calc_aggregate(run)
        #     return results_gdict
    
        # run = get_irmeasures_run(recs, self.df) # note run is a pd.Dataframe, shape topk * self.df users
        # results = self.evaluator.calc_aggregate(run) # returns a dictionary with metric as key and its value
        # results = {str(k): v for k, v in results.items()}
        run = get_irmeasures_run(recs, self.df)
        self.compute_average_metrics(run)
        self.compute_allgroup_metrics(run)
        self.update_maxmingap_bestsubgroups() # finds max min and gap amongst only subgroups, and best min metric for each subgroup
        self.update_maxmingap_bestgroups() # finds max min and gap amongst all groups
        # results_gdict = get_results_gdict(run) # gname key and corresponding metrics dictionary
        
        # gname, evalmetrics_g = random.choice(list(results_gdict.items())) # evalmetris_g is a dictionary with  
        # max_group_metric = defaultdict(lambda:float("-inf"))
        # min_group_metric = defaultdict(lambda:float("inf"))
        
        # for key in evalmetrics_g:
        #     key_across = str(key)
        #     for gname, res in results_gdict.items():
        #         results[str(key) + "_" + gname] = res[key]
        #         max_group_metric[key_across] = max(res[key], max_group_metric[key_across])
        #         min_group_metric[key_across] = min(res[key], min_group_metric[key_across])
        #     results['max_' + key_across] = max_group_metric[key_across]
        #     results['min_'+ key_across] = min_group_metric[key_across]
        #     results['gap_'+ key_across] = max_group_metric[key_across] - min_group_metric[key_across]

        # for gname, res in results_gdict.items():
        #     for key in res:
        #         results[str(key) + "_" + gname] = res[key]
        # self.results = results # TODO better name, clash with name in eval_checkpoint_groups_clean
    
    def compute_average_metrics(self, run):
        results = self.evaluator.calc_aggregate(run) # returns a dictionary with metric as key and its value
        results = {str(k): v for k, v in results.items()}
        self.results.update(results) # bestavg wiped if reinitialized, so update
        for metric in self.metrics:
            mt = str(metric)
            ba_key = 'bestavg_' + mt
            if (ba_key not in self.results.keys()) or (self.results[mt] > self.results[ba_key]):
                self.results[ba_key] = self.results[mt]
                self.results[ba_key + '_set'] = True
            else:
                self.results[ba_key + '_set'] = False

    def compute_allgroup_metrics(self, run):
        def get_results_gdict(run):
            results_gdict = {}
            for k, evaluator in self.g_eval_dict.items():
                results_gdict[k] = evaluator.calc_aggregate(run)
            return results_gdict
        results_gdict = get_results_gdict(run) # gname key and corresponding metrics dictionary
        for gname, res in results_gdict.items():
            for key in res:
                self.results[str(key) + "_" + gname] = res[key]

    def update_maxmingap_bestsubgroups(self): #min and max over the subgroups
        for sg in self.subgroups:
            for metric in self.metrics:
                max_sg_metric = float("-inf")
                min_sg_metric = float("inf")
                mt = str(metric)
                for gname in sg:
                    max_sg_metric = max(self.results[mt + '_' + gname], max_sg_metric)
                    min_sg_metric = min(self.results[mt + '_' + gname], min_sg_metric)
                sgname = sg[0].split("_g-")[0] # (subgroup_pop)_g-..
                self.results[f'max_{mt}_{sgname}'] = max_sg_metric
                self.results[f'min_{mt}_{sgname}'] = min_sg_metric
                self.results[f'gap_{mt}_{sgname}'] = max_sg_metric - min_sg_metric
                # if self.results[f'min_{metric}_{sgname}'] > self.results[f'maxmin_{metric}_{sgname}']: # FIXME fragile, assuming default dict of -infty
                #     self.results[f'maxmin_{metric}_{sgname}'] = self.results[f'min_{metric}_{sgname}']

    def update_maxmingap_bestgroups(self): #min and max over all groups
        for metric in self.metrics:
            max_metric = float("-inf")
            min_metric = float("inf")
            mt = str(metric)
            bm_key = 'bestmin_'+ mt
            for gname in self.group_cols:
                max_metric = max(self.results[mt + '_' + gname], max_metric)
                min_metric = min(self.results[mt + '_' + gname], min_metric)
            self.results[f'max_{mt}'] = max_metric
            self.results[f'min_{mt}'] = min_metric
            self.results[f'gap_{mt}'] = max_metric - min_metric
            if (bm_key not in self.results.keys()) or self.results[f'min_{mt}'] > self.results[bm_key]:  # FIXME fragile, assuming default dict of -infty
                self.results[bm_key] = self.results[f'min_{mt}']
                self.results[bm_key + '_set'] = True
            else:
                self.results[bm_key + '_set'] = False

    def get_metrics(self, split_name:str): # used for validation
        # return {f'{split_name}/{k}':v for k,v in self.results.items() if '_set' not in str(k)}
        return {f'{split_name}/{k}':v for k,v in self.results.items() 
                if (('bestmin' in k) or ('bestavg' in k)) and ('_set' not in str(k))}
    
    def get_onerun_testmetrics(self, split_name = 'test'): # used for test metrics
        return {f'{split_name}/{k}':v for k,v in self.results.items() 
                if ('bestmin' not in str(k)) and ('bestavg' not in str(k))}
    
    def save_results_to_json(self):
        pass
