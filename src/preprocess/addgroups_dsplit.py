"""
addgroups.py: Adds groups that can be defined for any sequential dataset, for e.g
1) user groups based on sequence lengths: short, medium, long sequences based on quantiles
2) user groups based on popular items: Majority popular items, majority diverse, majority niche items in user sequence

Not the preprocessed and split mmaps must exist before this addgroup.py is run
"""
from argparse import ArgumentParser
from pathlib import Path
from src.utils.load_data import load_data
from src.utils.save_data import save_df
from src.utils.project_dirs import processed_data_root
import pandas as pd
from collections import Counter, defaultdict
import numpy as np
from src.utils.get_training_params import set_seed

DATA_DIR = Path("__file__").absolute().parent / "data"
RAND_SEED = 42

def group_seqlen_usersplit(df, small_size = 0.1, med_size=0.8, long_size=0.1) -> tuple[pd.DataFrame, list[str], list[int], list[int]]:
    '''
        splits based on users default 0.1 of all users are small sequence length
        0.8 of all users have medium sequence, rest 0.1 users have long sequence length
        In practice these thresholds correspond to new users, regular and power users
    '''
    ucount = Counter(df.user_id).most_common() # descending order
    df_useq = pd.DataFrame(ucount, columns=['user_id', 'seqlen'])
    gcols = ['g-small','g-medium' ,'g-long']
    n = len(df_useq)
    idx_1 = int(long_size * n)       # index to split 'small' and 'medium'
    idx_2 = int((long_size + med_size) * n)  # index to split 'medium' and 'large'
    for gcol in gcols: df_useq[gcol] = 0
    df_useq.loc[:idx_1, 'g-long'] = 1
    df_useq.loc[idx_1:idx_2, 'g-medium'] = 1
    df_useq.loc[idx_2:, 'g-small'] =  1

    usplits = [df_useq['g-small'].mean(), df_useq['g-medium'].mean(), df_useq['g-long'].mean()]
    print(f'User splits for small, medium and long seq users are {usplits}')
    # print(f'User splits for small, medium and long seq users are {df_useq['g-small'].mean(), df_useq['g-medium'].mean(), df_useq['g-long'].mean()}')
    df = df.merge(df_useq, on = 'user_id')
    dsplits = [df['g-small'].mean(), df['g-medium'].mean(), df['g-long'].mean()]
    # print(f'Training data splits for small, medium and long users are {df['g-small'].mean(), df['g-medium'].mean(), df['g-long'].mean()}')
    print(f'Training data splits for small, medium and long users are {dsplits}')

    df_useq[gcols] = df_useq[gcols].astype('float32')
    sgcols = ['subgroup_seq_' + gcol for gcol in gcols]
    remap_cols = {gname:sgcols[i] for i, gname in enumerate(gcols)}
    df_final = df_useq[['user_id'] + gcols].rename(columns = remap_cols)
    return df_final, sgcols, np.array(usplits), np.array(dsplits) #note the user id order may not be original sequence

def group_seqlen_datasplit(df, small_size = 1/3, med_size=1/3, long_size=1/3):
    # assert (small_size + med_size + long_size) == 1
    ucount = Counter(df.user_id).most_common() # descending order
    df_useq = pd.DataFrame(ucount, columns=['user_id', 'seqlen'])
    df = df.merge(df_useq, on = 'user_id')
    df.sort_values(by = ['seqlen', 'user_id'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    n = len(df)
    idx1 = int(small_size * n)
    idx2 = int((small_size + med_size) * n)

    #rough estimate of user splits required for balanced data split
    u_small = len(df.iloc[:idx1].user_id.unique())
    u_medium = len(df.iloc[idx1:idx2].user_id.unique())
    u_long  = len(df.iloc[idx2:].user_id.unique())
    tot = (u_small+u_medium+u_long) 
    return group_seqlen_usersplit(df, small_size= u_small/tot, med_size=u_medium/tot, long_size = u_long/tot) # user split that roughly get balanced groups


def add_item_popularity(df, top_item_frac=0.2):
    ''' 
        Used in groups_popitems_seq, which creates the user id to popular group mapping

        df is the training dataframe

        adds column 'item_is_pop' to the training dataframe which is 1 if that item is the top 20% of interacted items, 0 if not
        df has columns, user_id, item_id etc
    '''
    dd = df.drop_duplicates(subset=['user_id', 'item_id'])
    icount = Counter(dd.item_id) # counts how many unique users each item_id has been shown to
    ic_val = np.array([pair[1] for pair in icount.most_common()])  # number of unique users in descending sorted order
    pop_count = np.quantile(ic_val, 1 - top_item_frac)
    # print(f'The popular items are {(ic_val >= pop_count).mean()} of the itemset')
    item_to_pop = {} # 1 if item is popular else 0
    for item, count in icount.most_common():
        if count > pop_count:
            item_to_pop[item] = 1
        else:
            item_to_pop[item] = 0
    df['item_is_pop'] = df['item_id'].apply(lambda x : item_to_pop[x])
    return df

def groups_popitems_usersplit(df:pd.DataFrame, niche_size = 0.1, div_size = 0.8, pop_size = 0.1,
                              top_item_frac = 0.2) ->tuple[pd.DataFrame,list[str]]:
    '''
        A user is one of 3 types: popular, diverse, niche
        Popular if its in the top quantile for ratio of popular items in it's sequence 
        Like in Google SDRO paper https://dl.acm.org/doi/pdf/10.1145/3485447.3512255
        
        map each user_id to one of <popular viewer, diverse viewer or niche viewer>
        only using the training data sequence

        df: should be the training dataframe

        Returns 
        user_idgroupmap dataframe, has shape nusers x 4 cols (user_id, g-pop, g-diverse,g-nice)
    '''
    # assert (niche_size + div_size + pop_size) == 1
    gcols = ['g-niche', 'g-diverse', 'g-popular']
    df = add_item_popularity(df, top_item_frac=top_item_frac)
    df_popratio_user = df.groupby('user_id', as_index=False)['item_is_pop'].mean() # df_popratio_user has shape is number of users x 2 cols, item_is_pop column is now a ratio of pop items in that sequence
    
    df_popratio_user.sort_values(by = ['item_is_pop'], inplace=True)
    df_popratio_user.reset_index(drop = True, inplace=True)
    n = len(df_popratio_user)
    idx_1 = int(niche_size * n)       # index to split 'small' and 'medium'
    idx_2 = int((niche_size + div_size) * n)  # index to split 'medium' and 'large'
    for gcol in gcols: df_popratio_user[gcol] = 0
    df_popratio_user.loc[:idx_1, 'g-niche'] = 1
    df_popratio_user.loc[idx_1:idx_2, 'g-diverse'] = 1
    df_popratio_user.loc[idx_2:, 'g-popular'] =  1

    usplits = [df_popratio_user['g-niche'].mean(), df_popratio_user['g-diverse'].mean(), df_popratio_user['g-popular'].mean()]
    # print(f'User splits for niche, diverse and popular viewers are {df_popratio_user['g-niche'].mean(), \
    #     df_popratio_user['g-diverse'].mean(), df_popratio_user['g-popular'].mean()}')
    print(f'User splits for niche, diverse and popular viewers are {usplits}')

    df = df.merge(df_popratio_user, on = 'user_id') # basically now we have the columns which is a binary array for each user_id
    dsplits = [df['g-niche'].mean(), df['g-diverse'].mean(), df['g-popular'].mean()]
    # print(f'Training data splits for niche, diverse and popular viewers are {df['g-niche'].mean(), \
    #     df['g-diverse'].mean(), df['g-popular'].mean()}')
    print(f'Training data splits for niche, diverse and popular viewers are {dsplits}')
    
    sgcols = ['subgroup_pop_' + gcol for gcol in gcols]
    df_popratio_user[gcols] = df_popratio_user[gcols].astype('float32')
    remap_cols = {gname:sgcols[i] for i,gname in enumerate(gcols)}
    df_final = df_popratio_user[['user_id'] + gcols].rename(columns = remap_cols)
    return df_final, sgcols, np.array(usplits), np.array(dsplits)

def group_popitems_datasplit(df, niche_size = 1/3, div_size=1/3, pop_size=1/3, top_item_frac=0.2):
    gcols = ['g-niche', 'g-diverse', 'g-popular']
    df = add_item_popularity(df, top_item_frac=top_item_frac) #FIXME rename item_is_pop mean below
    df_popratio_user = df.groupby('user_id', as_index=False)['item_is_pop'].mean() # df_popratio_user has shape is number of users x 2 cols, item_is_pop column is now a ratio of pop items in that sequence
    
    df = df.drop(columns=['item_is_pop']).merge(df_popratio_user, on = 'user_id') #to avoid name clash itemp_is_pop #FIXME cleaner
    df.sort_values(by = ['item_is_pop', 'user_id'], inplace=True)
    df.reset_index(drop = True, inplace=True)
    n = len(df)
    idx1 = int(niche_size * n)
    idx2 = int((niche_size + div_size) * n)
    #rough estimate of user splits required for balanced data split
    u_niche = len(df.iloc[:idx1].user_id.unique())
    u_div = len(df.iloc[idx1:idx2].user_id.unique())
    u_pop  = len(df.iloc[idx2:].user_id.unique())
    tot = (u_niche+u_div+u_pop)
    return groups_popitems_usersplit(df, niche_size=u_niche/tot, div_size=u_div/tot, pop_size=u_pop/tot)


def main():
    set_seed(RAND_SEED)

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    # parser.add_argument("--popsplit", nargs=2, type=float, default = [0.1, 0.8, 0.1], required=False, help='user split size for niche, diverse item viewer')
    # parser.add_argument("--popname", type=str, required=False, default='popgroup')
    # parser.add_argument("--seqsplit", nargs=2, type=float, default = [0.1, 0.8, 0.1], required=False, help='user split size for small, medium user') # sequence legths
    # parser.add_argument("--seqname", type=str, required=False, default='seqgroup')
    # parser.add_argument("--popseqname", type=str, required=False, default='popseqgroup')

    args = parser.parse_args()
    data = load_data(args.dataset)

    gmap_dir = processed_data_root() / args.dataset / "uid_to_group" #·save user_id to group mapping dataframe, shape is number of users
    gmap_dir.mkdir(exist_ok=True, parents=True)

    cols = ["user_id", "item_id", "timestamp"]
    
    print("## NON INTERSECTING groups")
    print("--Popular items in sequence, datasplit 10-80-10")
    dfpop_1080, sgpop, usplitpop1080, dsplitpop1080 = group_popitems_datasplit(data['train'][cols], 
                                                     niche_size=0.1, div_size=0.8,
                                                     pop_size=0.1)
    dfpop_1080.attrs['sglist'] = [sgpop]
    dfpop_1080.attrs['glist'] = sgpop
    save_df(dfpop_1080, gmap_dir/"popdsplit_0.1_0.8_0.1.pkl")

    print("--Popular items in sequence, datasplit 20-60-20")
    dfpop_2060, sgpop, usplitpop2060, dsplitpop2060 = group_popitems_datasplit(data['train'][cols],
                                                       niche_size=0.2, div_size=0.6,
                                                       pop_size=0.2)
    dfpop_2060.attrs['sglist'] = [sgpop]
    dfpop_2060.attrs['glist'] = sgpop
    save_df(dfpop_2060, gmap_dir/"popdsplit_0.2_0.6_0.2.pkl")

    print("--Popular items in sequence, datasplit 33")
    dfpop_bal, sgpop, usplitpop33, dsplitpop33 = group_popitems_datasplit(data['train'][cols])
    dfpop_bal.attrs['sglist'] = [sgpop]
    dfpop_bal.attrs['glist'] = sgpop
    save_df(dfpop_bal, gmap_dir/f"popdsplit_balanced.pkl")        
#############
    print("--Seqlen, datasplit 10-80-10")
    dfseq_1080, sglen, usplitseq1080, dsplitseq1080 = group_seqlen_datasplit(data['train'][cols], 
                                                        small_size=0.1, 
                                                        med_size=0.8, 
                                                        long_size=0.1)
    dfseq_1080.attrs['sglist'] = [sglen]
    dfseq_1080.attrs['glist'] = sglen
    save_df(dfseq_1080, gmap_dir /"seqdsplit_0.1_0.8_0.1.pkl")
    
    print("--Seqlen, datasplit 20-60-20")
    dfseq_2060, sglen, usplitseq2060, dsplitseq2060  = group_seqlen_datasplit(data['train'][cols], 
                                                        small_size=0.2, 
                                                        med_size=0.6, 
                                                        long_size=0.2)
    dfseq_2060.attrs['sglist'] = [sglen]
    dfseq_2060.attrs['glist'] = sglen
    save_df(dfseq_2060, gmap_dir /"seqdsplit_0.2_0.6_0.2.pkl")

    print("--Seqlen, datasplit balanced")
    dfseq_bal, sglen, usplitseq33, dsplitseq33 = group_seqlen_datasplit(data['train'][cols])
    dfseq_bal.attrs['sglist'] = [sglen]
    dfseq_bal.attrs['glist'] = sglen
    save_df(dfseq_bal, gmap_dir /"seqdsplit_balanced.pkl")

    # two_rows = [usplitpop33 + usplitpop2060 + usplitpop1080 + usplitseq33 + usplitseq2060 + usplitseq1080,
    #             dsplitpop33 + dsplitpop2060 + dsplitpop1080 + dsplitseq33 + dsplitseq2060 + dsplitseq1080]
    two_rows = [[usplitpop33, usplitpop2060, usplitpop1080, usplitseq33, usplitseq2060,usplitseq1080],
                [dsplitpop33, dsplitpop2060, dsplitpop1080, dsplitseq33, dsplitseq2060, dsplitseq1080]]
    dsplit_df = pd.DataFrame(two_rows)
    save_df(dsplit_df, gmap_dir/"dsplit_info.pkl")
    # dsplit_df.to_csv(gmap_dir/"dsplit_info.csv", float_format='%.2f', index=False, header=False)

    ## INTERSECTING groups
    # print("pop and seqlen skewed intersection")
    # df_userid_popseq_skew = df_userid_pop_skew.merge(df_userid_seqlen_skew, on = 'user_id') #·has all 6 groups, 3 from pop viewer and 3 from cold start, these intersect!
    # df_userid_popseq_skew.attrs['sglist'] = [sgpop, sglen] #list of subgroups
    # df_userid_popseq_skew.attrs['glist'] = sgpop + sglen
    # save_df(df_userid_popseq_skew, gmap_dir /f"{args.popseqname}_skew.pkl")

    print("pop and seqlen balanced intersection")
    df = dfpop_bal.merge(dfseq_bal, on = 'user_id') #·has all 6 groups, 3 from pop viewer and 3 from cold start, these intersect!
    df.attrs['sglist'] = [sgpop, sglen] #list of subgroups
    df.attrs['glist'] = sgpop + sglen
    save_df(df, gmap_dir /"popseq_bal.pkl")

if __name__ == "__main__":
    main()