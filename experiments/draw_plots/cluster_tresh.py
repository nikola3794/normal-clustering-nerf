################################################################################
#                           FORMATTING                    
################################################################################
import os

import pandas as pd 
import numpy as np

from matplotlib import pyplot as plt 

pgf_with_latex = {                      # setup matplotlib to use latex for output
    #"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[detect-all,locale=DE]{siunitx}",
        r"\usepackage[gensymb]"
        ])
    }
plt.rcParams.update(pgf_with_latex)

fig_fontsize = 22
axis_fontsize = fig_fontsize*0.8
legend_fontsize = fig_fontsize*0.6
marker_size=7

dpi = 600

plt.rcParams["figure.figsize"] = (9,3)

################################################################################
#                           LOADING DATA                    
################################################################################

# Mean results
results_dir = '/home/nipopovic/MountedDirs/euler/work_specta/experiment_logs/ngp_mt/cvpr23_results/supp_tresh_clust'
df_mean = pd.read_csv(os.path.join(results_dir, 'multi_batch_mean.csv'))
cluster_thresh_mean = df_mean['param/loss_norm_can_tres'].values
psnr_mean = df_mean['metric/rgb/psnr'].values
yaw_abs_mean = df_mean['metric/ang/clust/yaw_abs'].values
pitch_abs_mean = df_mean['metric/ang/clust/pitch_abs'].values
roll_abs_mean = df_mean['metric/ang/clust/roll_abs'].values
# Median results
df_median = pd.read_csv(os.path.join(results_dir, 'multi_batch_median.csv'))
cluster_thresh_median = df_median['param/loss_norm_can_tres'].values
psnr_median = df_median['metric/rgb/psnr'].values
yaw_abs_median = df_median['metric/ang/clust/yaw_abs'].values
pitch_abs_median = df_median['metric/ang/clust/pitch_abs'].values
roll_abs_median = df_median['metric/ang/clust/roll_abs'].values

psnr_baseline_mean = [25.86]

x_min = 0.002
x_max = 0.55
################################################################################
#                           DRAWING PSNR                     
################################################################################
y_min = 24.5
y_max = 28.5
# variance plot 
# https://stackoverflow.com/questions/43064524/plotting-shaded-uncertainty-region-in-line-plot-in-matplotlib-when-data-has-nans

plt.plot(
    cluster_thresh_mean,
    psnr_mean,
    color='b',
    marker='^',
    label='Ours',
    markersize=marker_size
)

plt.plot(
    [x_min, x_max],
    psnr_baseline_mean*2,
    color='red',
    linestyle='dashed',
    #marker='o',
    label='Baseline',
    markersize=marker_size
)


plt.xlabel('Cluster threshold $t$ (log scale)', fontsize=fig_fontsize)
plt.ylabel('PSNR [dB] $\\uparrow$', fontsize=fig_fontsize)
plt.title(f'Cluster threshold effects on novel-view rendering', fontsize=fig_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid()
plt.xticks(fontsize=axis_fontsize)
plt.yticks(fontsize=axis_fontsize)
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.xscale('log')
#plt.show()

plt.savefig(os.path.join(results_dir, 'cluster_thresh_psnr_mean.pdf'), bbox_inches='tight', dpi=dpi)
plt.close()

################################################################################
#                           DRAWING ANGLE ERRORS                  
################################################################################
y_min = 0.0
y_max = 10.0
plt.plot(
    cluster_thresh_median,
    yaw_abs_median,
    color='y',
    marker='o',
    label="Yaw",
    markersize=marker_size
)
plt.plot(
    cluster_thresh_median,
    pitch_abs_median,
    color='c',
    marker='x',
    label="Pitch",
    markersize=marker_size
)
plt.plot(
    cluster_thresh_median,
    roll_abs_median,
    color='g',
    marker='s',
    label="Roll",
    markersize=marker_size
)

plt.xlabel('Cluster threshold $t$ (log scale)', fontsize=fig_fontsize)
plt.ylabel('$L_1$ error $[{}^{\circ}]$ $\downarrow$', fontsize=fig_fontsize)
plt.title(f'Cluster threshold effects on Manhattan frame estimation', fontsize=fig_fontsize)
plt.legend()
plt.grid()
plt.xticks(fontsize=axis_fontsize)
plt.yticks(fontsize=axis_fontsize)
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.xscale('log')
#plt.show()

plt.savefig(os.path.join(results_dir, 'cluster_thresh_angles_median.pdf'), bbox_inches='tight', dpi=dpi)
plt.close()