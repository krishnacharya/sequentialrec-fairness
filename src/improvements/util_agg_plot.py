import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils.project_dirs import get_dataset_group_metric_irmdir
from collections import defaultdict
from scipy.stats import ttest_rel
from src.utils.ir_measures_converters import get_irmeasures_qrels, get_irmeasures_run
from src.utils.load_data import *
import ir_measures
from ir_measures import nDCG, R

def get_merged_dsplit(df_bal, df_2060, df_1080, metric_col:str):
    '''
        all 3 dfs should have exactly 8 rows for the 8 methods: erm, cb, cb_log, ipw, ipw_log, gdro, sdro, joint_dro
        metric_col has to be a column in the dataframe
    '''
    assert df_bal.shape[0] == df_2060.shape[0] == df_1080.shape[0]

    cols = ['loss_type', metric_col]
    df_across = df_bal[cols].merge(df_2060[cols], on = 'loss_type', suffixes=('_bal','_2060'))
    df_across = df_across.merge(df_1080[cols], on = 'loss_type').rename(columns = {metric_col:f'{metric_col}_1080'})
    return df_across

def plot_df_dsplit(df_across, metric_name:str):
    '''
        df_across: is a merged dataframe across dsplit
        4 columns loss_type and ndcg for different splits

        metric_name is used for the title and ylabel

    '''
    assert df_across.shape[1] == 4
    splits = [c for c in df_across.columns if c != 'loss_type']
    x_names = ['split_33', 'split_206020', 'split_108010']
    x = range(3)  # x-axis for the three splits

    plt.figure(figsize=(10, 6))

    # Plot each loss_type curve
    for i, row in df_across.iterrows():
        # plt.plot(x, [row[split] for split in splits], marker='o')
        plt.plot(x, [100*row[split] for split in splits], label=row['loss_type'], marker='o') # plotting a percent

    # Setting labels and title
    plt.xticks(x, x_names, rotation=45)
    plt.xlabel('Data Splits')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} for Different Training Methods Across Splits')
    plt.legend(title='Method')

    # Show the plot
    plt.tight_layout()
    plt.show()

# NEW Sec 4.2

def get_pum(testdata, irmpath:str, metric=nDCG@20) -> pd.DataFrame:
    '''
        metric: irmeasures nDCG@20 or R@20
        computes per user ndcg or recall
    '''
    qrels = get_irmeasures_qrels(testdata)
    run = pd.read_csv(irmpath)
    run.query_id = run.query_id.astype("str")
    run.doc_id = run.doc_id.astype("str")
    results = defaultdict(lambda: defaultdict())

    for m in ir_measures.iter_calc([metric], qrels, run):
        results[m.query_id]["user_id"] = m.query_id
        results[m.query_id][str(m.measure)] = m.value

    return pd.DataFrame([dict(x) for x in results.values()])

def match_array_sizes(arr1, arr2):
    if arr1.size > arr2.size:
        arr1 = np.delete(arr1, np.random.choice(arr1.size, arr1.size - arr2.size, replace=False))
    elif arr2.size > arr1.size:
        arr2 = np.delete(arr2, np.random.choice(arr2.size, arr2.size - arr1.size, replace=False))
    return arr1, arr2

def get_statsig_df(df_methods, df_g, dataset:str, group:str, metric = nDCG@20):
    '''
        df_methods has the methods, 8 for non intersecting, 12 for intersecting (gdro, sdro, ipw have 2 of eachf or loss based on popgroups vs seqgroup)
        metric: nDCG@20 or R@20, avgval sorted, and using those checkpoints
        Computes statistical significance wrt ERM across all gcol

        group: popbal33, pop2060, pop1080, seqbal, seqbal2060, seqbal33, popseq these are the folders in which the irm csv are saved

        returns 
        statistical significance dataframe, mean and std of user ndcgs for each group 
    '''
    def get_meang(method_pumg, gcols, metric=nDCG@20) -> dict: # groupwise mean metric calculation
        meang = {}
        for g in gcols:
            meang[g] = method_pumg[method_pumg[f'{g}'] == 1.0][str(metric)].mean()
        return meang
    def get_meang_ttest(erm_pumg, method_pumg, metric = nDCG@20) -> dict:
        '''
           T test on user group NDCG arrays
        '''
        issig = {}
        for g in gcols:
            erm_g = erm_pumg[erm_pumg[f'{g}'] == 1.0][str(metric)].values
            method_g = method_pumg[method_pumg[f'{g}'] == 1.0][str(metric)].values
            erm_g, method_g = match_array_sizes(erm_g, method_g)
            _, pval = ttest_rel(erm_g, method_g)
            issig[g] = pval < 0.05
        return issig
    irm_paths = {}
    testdata = load_data(dataset)['test']
    for lt, jobname in df_methods[['loss_type','dir_name']].values:
        path = str(get_dataset_group_metric_irmdir(dataset, group, str(metric)) / jobname.split('/')[0]) + '_top_20_irm_run.csv'
        irm_paths[f'{lt}'] = path
    df_g['g-Overall'] = 1.0 # adding an column for overall NDCG ssig calculation etc
    gcols = [c for c in df_g.columns if c != 'user_id']
    methods_pumg = {k : get_pum(testdata, irmpath, metric=metric).merge(df_g, on = 'user_id') for k, irmpath in irm_paths.items()} # pum, for each user topK=20 recommendations, merged with group columns
    issglist = []
    meanglist = []
    for k, method_pumg in methods_pumg.items():
        if k == 'erm': # HACK y, also initializes dicitionary only if erm is first in the df_methods
            issig = {g: False for g in gcols}
            issig['Method'] = 'erm'
        else:
            issig = get_meang_ttest(methods_pumg['erm'], method_pumg, metric)
            issig['Method'] = k 
        issglist.append(issig)
        meang = get_meang(method_pumg, gcols, metric)
        meang['Method'] = k
        meanglist.append(meang)
    reoder = ['Method'] + gcols
    return pd.DataFrame(issglist)[reoder], pd.DataFrame(meanglist)[reoder]
    
def style_latex(value, is_max, is_second_max, add_star=False):
    formatted_value = f'{value:.3f}' # TODO add precision are param
    if is_max:
        formatted_value = f"\\textbf{{{formatted_value}}}"
    elif is_second_max:
        formatted_value = f"\\underline{{{formatted_value}}}"

    # Add superscript star if needed
    if add_star:
        formatted_value += "\\textsuperscript{*}"
    return formatted_value

def table_nonint_pop(df_popbal, df_pop2060, df_pop1080,
                     ss_popbal, ss_pop2060, ss_pop1080):
    '''
        These should have 5 columns
        1 for Method and 3 for Niche, Diverse,Popular, Avg/Overall nDCG

        ss is a boolean matrix, True if statistically significant
    '''
    assert df_popbal.shape == df_pop2060.shape == df_pop1080.shape == (8, 5)
    assert ss_popbal.shape == ss_pop2060.shape == ss_pop1080.shape == (8, 5)
    df_popbal.columns = ['Method', 'Niche_Gpopbal', 'Diverse_Gpopbal', 'Popular_Gpopbal', 'Overall_Gpopbal']
    ss_popbal.columns = ['Method', 'Niche_Gpopbal', 'Diverse_Gpopbal', 'Popular_Gpopbal', 'Overall_Gpopbal']

    df_pop2060.columns = ['Method', 'Niche_Gpop2060', 'Diverse_Gpop2060', 'Popular_Gpop2060', 'Overall_Gpop2060']
    ss_pop2060.columns = ['Method', 'Niche_Gpop2060', 'Diverse_Gpop2060', 'Popular_Gpop2060', 'Overall_Gpop2060']

    df_pop1080.columns = ['Method', 'Niche_Gpop1080', 'Diverse_Gpop1080', 'Popular_Gpop1080', 'Overall_Gpop1080']
    ss_pop1080.columns = ['Method', 'Niche_Gpop1080', 'Diverse_Gpop1080', 'Popular_Gpop1080', 'Overall_Gpop1080']

    df_merged = pd.merge(pd.merge(df_popbal, df_pop2060, on='Method'), df_pop1080, on='Method')
    ss_merged = pd.merge(pd.merge(ss_popbal, ss_pop2060, on='Method'), ss_pop1080, on='Method')
    df_merged['Method'] = ['ERM', 'CB', 'CB log', 'IPW', 'IPW log', 'GDRO', 'SDRO', 'CVaR']
    for group in ['Gpopbal', 'Gpop2060', 'Gpop1080']:
        for col in [f'Niche_{group}', f'Diverse_{group}', f'Popular_{group}', f'Overall_{group}']:
            max_value = df_merged[col].max()
            second_max_value = df_merged[col][df_merged[col] != max_value].max()
            df_merged[col] = df_merged.apply(
                lambda x: style_latex(x[col], 
                                    x[col] == max_value, 
                                    x[col] == second_max_value, 
                                    ss_merged[col][x.name]), axis=1)
    
    # Generate the LaTeX code for the table
    table_latex = """
    \\begin{tabular}{l|l l l l|l l l l|l l l l}
    \\toprule
    Method & \\multicolumn{4}{c|}{$G_{popbal}$} & \\multicolumn{4}{c|}{$G_{pop2060}$} & \\multicolumn{4}{c}{$G_{pop1080}$} \\\\
    & Niche & Diverse & Popular & Overall & Niche & Diverse & Popular & Overall & Niche & Diverse & Popular & Overall \\\\
    \\midrule
    """
    # Generate the rows of the table
    for _, row in df_merged.iterrows():
        table_latex += f"{row['Method']} & {row['Niche_Gpopbal']} & {row['Diverse_Gpopbal']} & {row['Popular_Gpopbal']} & {row['Overall_Gpopbal']} "
        table_latex += f"& {row['Niche_Gpop2060']} & {row['Diverse_Gpop2060']} & {row['Popular_Gpop2060']} & {row['Overall_Gpop2060']} "
        table_latex += f"& {row['Niche_Gpop1080']} & {row['Diverse_Gpop1080']} & {row['Popular_Gpop1080']} & {row['Overall_Gpop1080']} \\\\\n"

    # Close the table
    table_latex += """
    \\bottomrule
    \\end{tabular}
    """
    return table_latex



def table_nonint_seq(df_seqbal, df_seq2060, df_seq1080,
                     ss_seqbal, ss_seq2060, ss_seq1080):
    '''
        These should have 4 columns
        1 for Method and 3 for Short, Medium and Long

        ss is a boolean matrix, True if statistically significant
    '''
    assert df_seqbal.shape == df_seq2060.shape == df_seq1080.shape == (8, 5)
    assert ss_seqbal.shape == ss_seq2060.shape == ss_seq1080.shape == (8, 5)
    df_seqbal.columns = ['Method', 'Short_Gseqbal', 'Medium_Gseqbal', 'Long_Gseqbal', 'Overall_Gseqbal']
    ss_seqbal.columns = ['Method', 'Short_Gseqbal', 'Medium_Gseqbal', 'Long_Gseqbal', 'Overall_Gseqbal']

    df_seq2060.columns = ['Method', 'Short_Gseq2060', 'Medium_Gseq2060', 'Long_Gseq2060', 'Overall_Gseq2060']
    ss_seq2060.columns = ['Method', 'Short_Gseq2060', 'Medium_Gseq2060', 'Long_Gseq2060', 'Overall_Gseq2060']

    df_seq1080.columns = ['Method', 'Short_Gseq1080', 'Medium_Gseq1080', 'Long_Gseq1080', 'Overall_Gseq1080']
    ss_seq1080.columns = ['Method', 'Short_Gseq1080', 'Medium_Gseq1080', 'Long_Gseq1080', 'Overall_Gseq1080']

    df_merged = pd.merge(pd.merge(df_seqbal, df_seq2060, on='Method'), df_seq1080, on='Method')
    ss_merged = pd.merge(pd.merge(ss_seqbal, ss_seq2060, on='Method'), ss_seq1080, on='Method')
    df_merged['Method'] = ['ERM', 'CB', 'CBlog', 'IPW', 'IPWlog', 'GDRO', 'SDRO', 'CVaR']

    for group in ['Gseqbal', 'Gseq2060', 'Gseq1080', ]:
        for col in [f'Short_{group}', f'Medium_{group}', f'Long_{group}', f'Overall_{group}']:
            max_value = df_merged[col].max()
            second_max_value = df_merged[col][df_merged[col] != max_value].max()
            
            df_merged[col] = df_merged.apply(lambda x: style_latex(x[col], x[col] == max_value, \
                                    x[col] == second_max_value, ss_merged[col][x.name]), axis=1)

    # Generate the LaTeX code for the table
    table_latex = """
    \\begin{tabular}{l|l l l l|l l l l|l l l l}
    \\toprule
    Method & \\multicolumn{4}{c|}{$G_{seqbal}$} & \\multicolumn{4}{c|}{$G_{seq2060}$} & \\multicolumn{4}{c}{$G_{seq1080}$} \\\\
    & Short & Medium & Long & Overall & Short & Medium & Long & Overall & Short & Medium & Long & Overall \\\\
    \\midrule
    """

    # Generate the rows of the table
    for _, row in df_merged.iterrows():
        table_latex += f"{row['Method']} & {row['Short_Gseqbal']} & {row['Medium_Gseqbal']} & {row['Long_Gseqbal']} & {row['Overall_Gseqbal']}  "
        table_latex += f"& {row['Short_Gseq2060']} & {row['Medium_Gseq2060']} & {row['Long_Gseq2060']} & {row['Overall_Gseq2060']}"
        table_latex += f"& {row['Short_Gseq1080']} & {row['Medium_Gseq1080']} & {row['Long_Gseq1080']} & {row['Overall_Gseq1080']} \\\\\n"

    # Close the table
    table_latex += """
    \\bottomrule
    \\end{tabular}
    """
    return table_latex

def table_popseq(df_ps, ss_ps):
    assert df_ps.shape == ss_ps.shape == (12, 8)
    df_ps.columns = ['Method', 'Niche', 'Diverse', 'Popular', 'Short', 'Medium', 'Long', 'Overall']
    ss_ps.columns = ['Method', 'Niche', 'Diverse', 'Popular', 'Short', 'Medium', 'Long', 'Overall']

    methods = ['ERM', 'CB', 'CBlog', 'CVaR',
              'IPW\\textsubscript{pop}', 'IPWlog\\textsubscript{pop}','GDRO\\textsubscript{pop}', 'SDRO\\textsubscript{pop}', 
              'IPW\\textsubscript{seq}', 'IPWlog\\textsubscript{seq}', 'GDRO\\textsubscript{seq}', 'SDRO\\textsubscript{seq}']
    df_ps['Method'] = methods
    ss_ps['Method'] = methods
    
    groups = ['Niche', 'Diverse', 'Popular', 'Short', 'Medium', 'Long', 'Overall']
    for col in groups:
        max_value = df_ps[col].max()
        second_max_value = df_ps[col][df_ps[col] != max_value].max()
        df_ps[col] = df_ps.apply(lambda x: style_latex(x[col], 
                                x[col] == max_value, 
                                x[col] == second_max_value, 
                                ss_ps[col][x.name]), axis=1)
    # Generate the LaTeX code for the table
    table_latex = """
    \\begin{tabular}{l|l l l l l l |l}
    \\toprule
    Method & Niche & Diverse & Popular & Short &  Medium & Long & Overall
    \\midrule
    """
    # Generate the rows of the table
    for _, row in df_ps.iterrows():
        table_latex += f"{row['Method']} & {row['Niche']} & {row['Diverse']} & {row['Popular']} "
        table_latex += f"& {row['Short']} & {row['Medium']} & {row['Long']} & {row['Overall']} \\\\\n"
    # Close the table
    table_latex += """
    \\bottomrule
    \\end{tabular}
    """
    return table_latex
    

def method_boxplot(df_bal, df_2060, df_1080):
    assert df_bal.shape == df_2060.shape == df_1080.shape == (8, 4)

    df_bal.columns = ['Method', 'g1_bal', 'g2_bal', 'g3_bal']
    df_2060.columns = ['Method', 'g1_2060', 'g2_2060', 'g3_2060']
    df_1080.columns = ['Method', 'g1_1080', 'g2_1080', 'g3_1080']
    df_merged = pd.merge(pd.merge(df_bal, df_2060, on='Method'), df_1080, on='Method')
    df_merged['Method'] = ['ERM', 'CB', 'CBlog', 'IPW', 'IPWlog', 'GDRO', 'SDRO', 'CVaR']
    return df_merged

def save_boxplot_nonint(df_bal, df_2060, df_1080, dsplitnames:list[str], groupnames:list[str], metric:str, filepath):
    '''
        Plot for section 4.2, non intersecting
    '''
    df_merged = method_boxplot(df_bal, df_2060, df_1080)
    df_min_max = pd.DataFrame({
    'Method': df_merged['Method'],
    'bal_min': df_merged[['g1_bal', 'g2_bal', 'g3_bal']].min(axis=1),
    'bal_max': df_merged[['g1_bal', 'g2_bal', 'g3_bal']].max(axis=1),

    '2060_min': df_merged[['g1_2060', 'g2_2060', 'g3_2060']].min(axis=1),
    '2060_max': df_merged[['g1_2060', 'g2_2060', 'g3_2060']].max(axis=1),

    '1080_min': df_merged[['g1_1080', 'g2_1080', 'g3_1080']].min(axis=1),
    '1080_max': df_merged[['g1_1080', 'g2_1080', 'g3_1080']].max(axis=1),
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', color='gray', alpha=0.7)
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'orange', 'purple', 'yellowgreen', 'violet', 'cyan'] # 8 methods
    markers = ['o', 'P', 'D']
    box_width = 0.08
    positions = [1, 2, 3] # Positions for bal, 2060, 1080
    for i, method in enumerate(df_min_max['Method']):
        for j, pos in enumerate(positions):
            min_val = df_min_max.iloc[i, j*2+1]
            max_val = df_min_max.iloc[i, j*2+2]
            ax.bar(pos + i * 0.1, max_val - min_val, bottom=min_val, width=box_width, color=colors[i], edgecolor='black')
        # Add scatter markers for g1, g2, and g3 inside the box
        ax.scatter([1 + i * 0.1, 2 + i * 0.1, 3 + i * 0.1],
                [df_merged.loc[i, 'g1_bal'], df_merged.loc[i, 'g1_2060'], df_merged.loc[i, 'g1_1080']],
                marker=markers[0], color='black', label='g1' if i == 0 else "")

        ax.scatter([1 + i * 0.1, 2 + i * 0.1, 3 + i * 0.1],
                [df_merged.loc[i, 'g2_bal'], df_merged.loc[i, 'g2_2060'], df_merged.loc[i, 'g2_1080']],
                marker=markers[1], color='black', label='g2' if i == 0 else "")

        ax.scatter([1 + i * 0.1, 2 + i * 0.1, 3 + i * 0.1],
                [df_merged.loc[i, 'g3_bal'], df_merged.loc[i, 'g3_2060'], df_merged.loc[i, 'g3_1080']],
                marker=markers[2], color='black', label='g3' if i == 0 else "")

    # Set x-ticks and labels
    ax.set_xticks([1.4, 2.4, 3.4])
    ax.set_xticklabels(dsplitnames)
    # ax.set_xlabel('Group')
    ax.set_ylabel(f'{metric}')
    # ax.set_title('Min-Max "Boxes" with Markers for g1, g2, g3')

    # Adjust the y-axis limits to leave some space between the lowest value and the x-axis
    y_min = df_min_max[['bal_min', '2060_min', '1080_min']].min().min() * 0.95  # 5% lower for gap
    y_max = df_min_max[['bal_max', '2060_max', '1080_max']].max().max() * 1.05 # 5% higher for gap
    ax.set_ylim(y_min, y_max)

    # Add a horizontal line at y=0 (or other reference line if needed)
    ax.axhline(0, color='black', linewidth=1.0, linestyle='--')

    handles_method = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(df_min_max['Method']))]
    legend1 = ax.legend(handles_method, df_min_max['Method'], bbox_to_anchor=(0, 1), loc='upper left', title='Methods')
    handles_marker = [plt.Line2D([0], [0], marker=markers[0], color='w', markerfacecolor='black', markersize=10, label='g1'),
                    plt.Line2D([0], [0], marker=markers[1], color='w', markerfacecolor='black', markersize=10, label='g2'),
                    plt.Line2D([0], [0], marker=markers[2], color='w', markerfacecolor='black', markersize=10, label='g3')]

    legend2 = ax.legend(handles_marker, groupnames, bbox_to_anchor=(0.11, 1), loc='upper left', title='Groups')
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.show()


def plot_nonint_3groups_droimprovement(df33, df_2060, df_1080, groupnames:list[str],
                                        filepath, xticklist = ['$G_{pop33}$', '$G_{pop2060}$', '$G_{pop1080}$'],
                                        same_ylim=False, metric='NDCG@20'):
    '''
        Plot percentage improvement over ERM for the DRO methods
        return dataframe of improvement
    '''
    dataframes = [df33, df_2060, df_1080]
    methods_to_plot = ['group_dro', 's_dro', 'joint_dro']
    colors = {'group_dro': 'blue', 's_dro': 'green', 'joint_dro': 'red'}
    method_rename = {'group_dro': 'GDRO', 's_dro': 'SDRO', 'joint_dro': 'CVaR'}
    improvement_data = {group: {method: [] for method in methods_to_plot} for group in groupnames}
    # First pass: calculate improvements over ERM and store in a dictionary
    for group in groupnames:
        for method in methods_to_plot:
            for df in dataframes:
                erm_value = df.loc[df['Method'] == 'erm', group].values[0]
                method_value = df.loc[df['Method'] == method, group].values[0]
                improvement = ((method_value - erm_value) / abs(erm_value)) * 100
                improvement_data[group][method].append(improvement)

    # To align y-axis across all plots, determine the global min and max y values
    y_min = min(min(values) for group_data in improvement_data.values() for values in group_data.values())
    y_max = max(max(values) for group_data in improvement_data.values() for values in group_data.values())
    y_min *= 1.1 if y_min < 0 else 0.9
    y_max *= 1.1 if y_max > 0 else 1.1

    # Create a figure and axis for each plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, group in enumerate(groupnames):
        ax = axes[i]
        for method in methods_to_plot:
            ax.plot([1, 2, 3], improvement_data[group][method], label=method_rename[method], color=colors[method], marker='o')
        
        ax.set_title(f'Improvement {groupnames[i]} group')
        # ax.set_xlabel('Group split')
        if i == 0:
            ax.set_ylabel(f'% Improvement {metric}') # only the first y axis labelled 
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(xticklist)
        if same_ylim:
            ax.set_ylim(y_min, y_max)  # Set consistent y-axis limits
        ax.legend()
        ax.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(filepath)
    plt.show()

# def get_2sg_intersecting(df_ps_n, filepath):
#     # Extract ERM values directly
#     erm_values = df_ps_n[df_ps_n['Method'] == 'ERM'][['Niche', 'Diverse', 'Popular', 'Short', 'Medium', 'Long']].values[0]

#     dro_methods = ['CVaR', 'GDRO_pop', 'SDRO_pop', 'GDRO_seq', 'SDRO_seq']
#     method_rename = {'CVaR': '$CVaR$', 'GDRO_pop': '$GDRO_{pop}$', 'SDRO_pop': '$SDRO_{pop}$',
#                      'GDRO_seq': '$GDRO_{seq}$', 'SDRO_seq': '$SDRO_{seq}$'}
#     df_selected = df_ps_n[df_ps_n['Method'].isin(dro_methods)]

#     # Calculate percentage improvements over ERM
#     for i, col in enumerate(['Niche', 'Diverse', 'Popular', 'Short', 'Medium', 'Long']):
#         df_selected[f'{col}_improvement'] = ((df_selected[col] - erm_values[i]) / erm_values[i]) * 100

#     # Plot 1: Popular Groups (Niche, Diverse, Popular)
#     fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

#     pop_cols = ['Niche', 'Diverse', 'Popular']
#     seq_cols = ['Short', 'Medium', 'Long']

#     bar_width = 0.15  # Adjusted width to fit all bars without overlap
#     x = np.arange(len(pop_cols))

#     # Adjust x positions for multiple bars
#     for i, method in enumerate(dro_methods):
#         ax[0].bar(x + i * bar_width, df_selected.loc[df_selected['Method'] == method, [f'{col}_improvement' for col in pop_cols]].values[0], 
#                   bar_width, label=method_rename[method])

#     ax[0].set_xticks(x + bar_width * 2)  # Adjusted for 5 groups
#     ax[0].set_xticklabels(pop_cols)
#     ax[0].set_ylabel('% Improvement in NDCG@20')
#     ax[0].legend(loc='upper right')
#     ax[0].grid(True)
#     ax[0].margins(0)  # Remove top and bottom whitespace

#     # Plot 2: Sequence Length Groups (Short, Medium, Long)
#     x_seq = np.arange(len(seq_cols))

#     for i, method in enumerate(dro_methods):
#         ax[1].bar(x_seq + i * bar_width, df_selected.loc[df_selected['Method'] == method, [f'{col}_improvement' for col in seq_cols]].values[0], 
#                   bar_width, label=method_rename[method])

#     ax[1].set_xticks(x_seq + bar_width * 2)  # Adjusted for 5 groups
#     ax[1].set_xticklabels(seq_cols)
#     # ax[1].set_xlabel('Sequence Length Groups')
#     ax[1].grid(True)
#     ax[1].margins(0)  # Remove top and bottom whitespace

#     # Add y-ticks to the second plot and ensure they are visible
#     ax[1].set_yticks(ax[0].get_yticks())
#     ax[1].yaxis.set_tick_params(labelleft=True)

#     # Tight layout
#     plt.tight_layout()
#     plt.savefig(filepath)
#     plt.show()


def get_2sg_intersecting(df_ps_n, filepath):
    # Extract ERM values directly
    erm_values = df_ps_n[df_ps_n['Method'] == 'ERM'][['Niche', 'Diverse', 'Popular', 'Short', 'Medium', 'Long']].values[0]

    dro_methods = ['CVaR', 'GDRO_pop', 'SDRO_pop', 'GDRO_seq', 'SDRO_seq']
    method_rename = {'CVaR': '$CVaR$', 'GDRO_pop': '$GDRO_{pop}$', 'SDRO_pop': '$SDRO_{pop}$',
                     'GDRO_seq': '$GDRO_{seq}$', 'SDRO_seq': '$SDRO_{seq}$'}
    hatch_patterns = {'CVaR': '///', 'GDRO_pop': '\\\\', 'SDRO_pop': 'xx', 'GDRO_seq': '--', 'SDRO_seq': '++'}

    df_selected = df_ps_n[df_ps_n['Method'].isin(dro_methods)]

    # Calculate percentage improvements over ERM
    for i, col in enumerate(['Niche', 'Diverse', 'Popular', 'Short', 'Medium', 'Long']):
        df_selected[f'{col}_improvement'] = ((df_selected[col] - erm_values[i]) / erm_values[i]) * 100

    # Plot 1: Popular Groups (Niche, Diverse, Popular)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    pop_cols = ['Niche', 'Diverse', 'Popular']
    seq_cols = ['Short', 'Medium', 'Long']

    bar_width = 0.15  # Adjusted width to fit all bars without overlap
    x = np.arange(len(pop_cols))

    # Adjust x positions for multiple bars
    for i, method in enumerate(dro_methods):
        ax[0].bar(x + i * bar_width, 
                  df_selected.loc[df_selected['Method'] == method, [f'{col}_improvement' for col in pop_cols]].values[0], 
                  bar_width, label=method_rename[method], hatch=hatch_patterns[method])

    ax[0].set_xticks(x + bar_width * 2)  # Adjusted for 5 groups
    ax[0].set_xticklabels(pop_cols)
    ax[0].set_ylabel('% Improvement in NDCG@20')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].margins(0)  # Remove top and bottom whitespace

    # Plot 2: Sequence Length Groups (Short, Medium, Long)
    x_seq = np.arange(len(seq_cols))

    for i, method in enumerate(dro_methods):
        ax[1].bar(x_seq + i * bar_width, 
                  df_selected.loc[df_selected['Method'] == method, [f'{col}_improvement' for col in seq_cols]].values[0], 
                  bar_width, label=method_rename[method], hatch=hatch_patterns[method])

    ax[1].set_xticks(x_seq + bar_width * 2)  # Adjusted for 5 groups
    ax[1].set_xticklabels(seq_cols)
    ax[1].grid(True)
    ax[1].margins(0)  # Remove top and bottom whitespace

    # Add y-ticks to the second plot and ensure they are visible
    ax[1].set_yticks(ax[0].get_yticks())
    ax[1].yaxis.set_tick_params(labelleft=True)

    # Tight layout
    plt.tight_layout()
    plt.savefig(filepath)
    plt.show()
