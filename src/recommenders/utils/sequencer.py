#converts train data to sequential format
#Also, handles the item id representations
#Uses item <0> as padding

from collections import defaultdict, Counter
import random
import torch
from src.recommenders.utils.loss import GroupInfo

class Sequencer(object):
    PAD_ITEM_ID = 0
    START_ITEM_ID = 1
    UNKNOWN_ITEM_ID= 2
    MASK_ITEM_ID = 3

    NUM_SPECIAL_ITEMS=4

    def __init__(self, training_data, group_info:GroupInfo, max_length=200, randomfrac_default=False):
        self.pad_item_id = self.PAD_ITEM_ID 
        self.start_item_id = self.START_ITEM_ID 
        self.unknown_item_id = self.UNKNOWN_ITEM_ID 
        self.mask_item_id = self.MASK_ITEM_ID
        
        self.df_userid_groups = group_info.df_ugmap
        self.group_cols = group_info.group_cols
        
        self.randomrac_default = randomfrac_default
        self.item_mapping_reverse = {self.pad_item_id: "<PAD>"}

        self.item_mapping = {"<PAD>": self.pad_item_id,
                             "<START>": self.start_item_id,
                             "<UNKNOWN>": self.unknown_item_id, 
                             "<MASK>": self.mask_item_id
                             }
        self.num_special_items = len(self.item_mapping)

        all_items = sorted(training_data.item_id.unique())
        self.trainitem_count = Counter(training_data.item_id) # number of times this item appeared in the training data
        self.N_train = len(training_data)
        # self.item_internalid_invfreq = defaultdict(lambda:1)  # count for internal item id, default to 1, want N_trainsize/freq item for each item
        for num, id in enumerate(all_items):
            external_id = id 
            internal_id = num +self.num_special_items
            self.item_mapping_reverse[internal_id] = external_id
            self.item_mapping[external_id] = internal_id # maps external item id to internal item id, user ids are unchanged
        user_sequences = defaultdict(list)
        user_sequences_invitem = defaultdict(list) # stores N_train/freq_item
        last_ts = -1000
        self.max_length=max_length
        for _, user_id, item_id, timestamp in training_data[["user_id", "item_id", "timestamp"]].itertuples():
            if timestamp < last_ts:
                raise ValueError("train data not sorted by timestamp")
            user_sequences[user_id].append(self.item_mapping[item_id]) # stores internal id
            user_sequences_invitem[user_id].append(self.N_train / self.trainitem_count[item_id]) # N_train/freq of item_id in trainingdatga
        self.user_sequences = dict(user_sequences)
        self.user_sequences_invitem = dict(user_sequences_invitem)
        
    #call this after model training finished, so that val actions are used for inference
    def add_val_actions(self, all_val_actions):
        for _, user_id, item_id, timestamp in all_val_actions[["user_id", "item_id", "timestamp"]].itertuples():
            if item_id in self.item_mapping: #only append to user seq if item in vocab o/w we dont have its learnt embedding etc
                self.user_sequences[user_id].append(self.item_mapping[item_id])
                self.user_sequences_invitem[user_id].append(self.N_train / self.trainitem_count[item_id])
        pass
        

    def get_sequences_batch(self, user_ids:list, randomfrac=None, shortening=0):
        sequences = [self.user_sequences[user_id] for user_id in user_ids]
        seqinv_weight = [self.user_sequences_invitem[user_id] for user_id in user_ids]
        longest = 0
        groups = None
        for i in range(len(user_ids)):
            if (randomfrac is None and self.randomrac_default) or randomfrac == True:
                try:
                    end = random.randint(1, len(sequences[i]))
                except:
                    pass
            else:
                end = len(sequences[i])
            start = max(0, end-self.max_length + 1 + shortening) # for most sequences in ML this is just negative so start normally, but for sequences that get truncated it matters, e.g. if train seq
            sequences[i] = [self.start_item_id] + sequences[i][start:end] # start item ID token is always added to the start of sequence
            longest = max(len(sequences[i]), longest) # finding longest user sequence within this batch to pad to that length
            seqinv_weight[i] = [1] + seqinv_weight[i][start:end]
        for i in range(len(sequences)):
            pad_length = (longest - len(sequences[i]))
            sequences[i] = [0] * pad_length + sequences[i] #left pad with padding token here zero 0. we want the newst items to be the latest in the sequence
            seqinv_weight[i] = [1] * pad_length + seqinv_weight[i] # all the 1s left of the original items will all be multiplied by in the loss calculation, change each of SASRec, LSTMRec forward!
        sequences = torch.tensor(sequences)
        seqinv_weight = torch.tensor(seqinv_weight) # should have same shape as sequences
        padding_mask = (sequences == self.pad_item_id).to(torch.float32)
        last_position_id = self.max_length  
        first_position_id = last_position_id - longest 
        position_ids = torch.arange(first_position_id, last_position_id)

        if self.df_userid_groups is not None:
            filtered = self.df_userid_groups[self.df_userid_groups['user_id'].isin(user_ids)]
            og_order = filtered.set_index('user_id').reindex(user_ids).reset_index() # original user ids order 
            groups = torch.tensor(og_order[self.group_cols].values) # assuming all are already fp32
    
        return {"sequences": sequences,
                "position_ids": position_ids, 
                "padding_mask": padding_mask,
                "groups": groups,
                "seqinv_weight":seqinv_weight}
    
    def repetitions_filter(self, user_ids, logits):
        for i in range(len(user_ids)):
            user_seq = self.user_sequences[user_ids[i]]
            logits[i][user_seq] = -float("inf")
        return logits

    def decode_topk(self, encoded_topk: torch.return_types.topk):
        internal_item_ids: torch.Tensor = encoded_topk.indices
        internal_item_ids = internal_item_ids.detach().cpu().numpy()
        res = []
        for row in range(len(internal_item_ids)):
            res_row = []
            for col in range(len(internal_item_ids[row])):
                internal_id = internal_item_ids[row][col]
                score = encoded_topk.values[row][col].item()
                external_id = self.item_mapping_reverse[internal_id]
                res_row.append((external_id, score))
            res.append(res_row)
        return res
