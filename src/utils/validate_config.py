
import copy
import ir_measures
def get_config_dict(config):
    config_dict = copy.deepcopy(config.__dict__)
    for key, val in list(config_dict.items()):
        if issubclass(type(val), ir_measures.measures.Measure):
            val = str(val)
            config_dict[key] = val
        if key == "rerank_cutoffs" and type(val) == list:
            val = ",".join([str(cutoff) for cutoff in config_dict[key]])
            config_dict[key] = val
        if not(type(val) in (str, int, float, bool)):
            config_dict.pop(key)
    return config_dict