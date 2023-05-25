#%% Import packages and modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

dir_cur = os.path.dirname(__file__)
dir_base = dir_cur[:dir_cur.find('Code')]
sys.path.append(dir_base + 'Code/Repos')
import mumeeg as my

#%% Define study, models, and other variables

study = 'MEG' # 'MEG' or 'EEG'
dir_in = dir_base + 'Results/ResultsMEEG/Results%s/' % study

baseline_regs = {'MEG': ['FirstNote', 'Oboe', 'Flute', 'HighNote', 'LowNote', 'RMS', 'VarEnvGTF', 'Flatness', 'Rep1'],
               'EEG': ['FirstNote', 'HighNote', 'LowNote', 'RMS', 'VarEnvGTF', 'Flatness', 'Rep1']}
baseline_regs = baseline_regs[study]

model_classes = ['MTFine']
if study == 'MEG':
    models_cl = {'Temperley': [10], 'IDyOM_stm': [1], 'IDyOM_ltm': [2], 'IDyOM_both': [75], 'MTFine': [8]}
elif study == 'EEG':
    models_cl = {'Temperley': [10], 'IDyOM_stm': [1], 'IDyOM_ltm': [2], 'IDyOM_both': [75], 'MTFine': [7]}
    
estimates_d = {'estimates1': ['s'],
                'estimates2': ['s', 'u'],
                'estimates3': ['s', 'u', 'sxu']}

permute     = 'none'
models = []
labels = []
models.append('Onset'), labels.append('Onset')
models.append('models-'+''.join(baseline_regs) + '_estimates-_permute-none'), labels.append('Baseline')
for model_class in model_classes:
    for cl in models_cl[model_class]:
        for e in estimates_d:
            model = '%s_%03d' % (model_class, cl) if isinstance(cl, (int, np.integer)) else'%s_%s' % (model_class, cl)
            model = 'models-'+''.join(baseline_regs + [model]) + '_estimates-'+''.join(estimates_d[e]) + ('_permute-%s' % permute) 
            models.append(model)
            label = '%s %d %s ' % (model_class, cl, ' '.join(estimates_d[e]))
            labels.append(label)

#models      = my.select_models(dir_in, incl_onsetonly=True)

colors      = my.get_colors(models)
sub_ids     = {'EEG': np.arange(1,21), 
               'MEG': np.arange(1,36)}[study]
pp_meas     = 'r' # 'r' or 'r2'

#%% Load data

bad_chs = ['MLT41-4304','MRO52-4304'] if study == 'MEG' else []
infos, evokeds, crossval_results = my.load_results(dir_in, models, sub_ids, bad_chs=bad_chs, pp_meas=pp_meas)
info = infos['Onset'][sub_ids[-1]]

# Compute grand average
ga = my.get_ga_evoked(evokeds)
         
# Select channels
select_ch = True
if select_ch:
    ch_selec, ch_names= my.get_ch_roi(crossval_results, infos, model='Onset')
else:
    ch_selec, ch_names = 'all', infos['Onset'][sub_ids[-1]]['ch_names']

model_perf, model_perf_sub= {}, {}
for model in models:
    model_perf[model], model_perf_sub[model] = my.get_overall_model_perf(crossval_results[model], ch_selec=ch_selec)

#%% Compare predictive performance (Fig. 7B)

data = {model: model_perf_sub[model] for model in models}
data = pd.DataFrame.from_dict(data)
i_baseline = 1
baseline = models[i_baseline]
my.plot_model_comparison_meeg(models, data, 
                              colors['discrete'], pp_meas, i_baseline=i_baseline,
                              figsize=(my.SINGLECOLUMN,my.DOUBLECOLUMN/1.7),
                              labels=labels)

#%% Run stats

table = data.subtract(data[baseline], axis = 0).describe().applymap(lambda x: f"{x:0.3f}")

data_interim = {label: data[label] for label in models}
ttest_res = my.paired_ttest(data_interim, alpha=0.05, method='bonferroni')
[print('%s - %s | t=%.02f  p_corr=%.2e, d=%.2f' 
       % (*(ttest_res.iloc[i,[0,1,2,4,6]].values),)) 
      for i in range(0, len(ttest_res))] # if ttest_res.iloc[i,5]]

#%% Plot correlation between surprise and uncertainty (Fig. 7A)

io_music        = my.get_music_io(study)
meta_music      = pd.read_csv(io_music['dir_music'] + io_music['fn_meta_music'])
df_music        = my.get_df_music(io_music, meta_music, load_from_disc=True)
df_MCCC         = my.get_df_MCCC(io_music, load_from_disc=True)
select_note     = {'yes': list(df_music['note_number'] != 0), 
                   'no': [True]*len(df_music)}['yes']
first_notes     = df_music['note_number'].values == 0
m = []
for model_class in model_classes:
    for cl in models_cl[model_class]:
        model = '%s_%03d' % (model_class, cl) if isinstance(cl, (int, np.integer)) else'%s_%s' % (model_class, cl)
        m.append(model)
        
results_note = my.get_results_note(m, io_music['dir_probs'], df_music)
results_comp = my.get_results_comp(m, results_note, df_music)
colors_music = my.get_colors(m)
df_note      = my.results2dfnote(results_note, m, df_music, models_cl)
df_comp      = my.results2dfcomp(results_comp, df_note, m, models_cl)

model = m[0]
data={}
data['x'] = results_note['uncertainty'][model]
data['y'] = results_note['surprise'][model]
    
g = sns.JointGrid(data=data, x='x', y='y', marginal_ticks=True, 
                  ratio=4, height=my.SINGLECOLUMN*(2/3))
g.plot_joint(
    sns.histplot,
    cmap=("light:%s" % colors_music['discrete'][model]), pmax=.8, cbar=False,
    stat = 'density')
g.plot_marginals(sns.histplot, element="step",
                  color=colors_music['discrete'][model], edgecolor=None,
                  stat = 'density')

# Add labels
g.set_axis_labels('Uncertainty', 'Surprise')

# Adjust axes layout
g.ax_joint.set_yticks([0,5,10])
g.ax_joint.set_ylim([0,12])
g.ax_joint.set_xticks([0,3])
g.ax_joint.set_xlim(left=0)

my.adjust_spines_sharedxy(g.ax_marg_x, ['left'])
my.adjust_spines_sharedxy(g.ax_marg_y, ['bottom'])
my.adjust_spines_sharedxy(g.ax_joint, ['left', 'bottom'])
plt.tight_layout()
plt.subplots_adjust(hspace=1*my.cm, wspace=1*my.cm)

# Add colorbar
cax = g.fig.add_axes([.6, .65, .2, .05])
cbar = plt.colorbar(g.ax_joint.collections[0], cax=cax, drawedges=False, orientation='horizontal')
sns.despine(ax=cax, left=True, bottom=True)
cax.set_frame_on(False)
cax.set_title('Density')
