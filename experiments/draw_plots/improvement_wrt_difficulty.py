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
marker_size=12

dpi = 600

plt.rcParams["figure.figsize"] = (9,3)

################################################################################
#                           LOADING DATA                    
################################################################################

# Mean results
results_dir = '/home/nipopovic/MountedDirs/euler/work_specta/experiment_logs/ngp_mt/cvpr23_results'

name = [
    'ScanNet', #~400
    ' Hypersim-A', #~80
    ' Hypersim-B', #~80
    ' Hypersim-C', #~80
    'Replica', #75

    'Hypersim-A-12',
    'Hypersim-A-9',
    'Hypersim-A-6',
]
psnr_baseline = [
    17.78,
    25.86,
    20.75,
    17.79,
    34.30,

    18.02,
    16.79,
    15.75
]
psnr_ours = [
    20.79,
    27.20,
    22.45,
    19.43,
    35.13,

    20.50,
    19.14,
    16.67
]

sorted_idx = np.argsort(psnr_baseline)
name = [name[i] for i in sorted_idx]
psnr_baseline = [psnr_baseline[i] for i in sorted_idx]
psnr_ours = [psnr_ours[i] for i in sorted_idx]
# (x/y-1)*100.0
improvement_ration = [(x/y-1)*100.0 for x, y in zip(psnr_ours, psnr_baseline)]



################################################################################
#                           DRAWING PSNR                     
################################################################################
x_min = 16.0
x_max = 37.0
y_min = 0.0
y_max = 21.0
# y_min = 0.8*100.0
# y_max = 1.3*100.0
# y_min = -0.01*100.0
# y_max = 0.3*100.0
# plt.plot(
#     [x_min, x_max],
#     [1.0]*2,
#     color='gray',
#     linestyle='dashed',
#     #marker='o',
#     #label='Baseline',
#     markersize=marker_size
# )
# variance plot 
# https://stackoverflow.com/questions/43064524/plotting-shaded-uncertainty-region-in-line-plot-in-matplotlib-when-data-has-nans

plt.plot(
    [x for i, x in enumerate(psnr_baseline) if ' Hypersim' in name[i]],
    [x for i, x in enumerate(improvement_ration) if ' Hypersim' in name[i]],
    color='b',
    marker='o',
    label='Hypersim',
    linestyle='dotted',
    markersize=marker_size
)

# plt.plot(
#     [x for i, x in enumerate(psnr_baseline) if 'Hypersim-A-' in name[i]],
#     [x for i, x in enumerate(improvement_ration) if 'Hypersim-A-' in name[i]],
#     color='c',
#     marker='o',
#     label='Hypersim sparse',
#     linestyle='dotted',
#     markersize=marker_size
# )


plt.scatter(
    [x for i, x in enumerate(psnr_baseline) if 'ScanNet' in name[i]],
    [x for i, x in enumerate(improvement_ration) if 'ScanNet' in name[i]],
    color='r',
    marker='*',
    label='ScanNet',
    s=marker_size**2
)

plt.scatter(
    [x for i, x in enumerate(psnr_baseline) if 'Replica' in name[i]],
    [x for i, x in enumerate(improvement_ration) if 'Replica' in name[i]],
    color='g',
    marker='^',
    label='Replica',
    s=marker_size**2
)


plt.xlabel('Baseline PSNR [dB]', fontsize=fig_fontsize)
plt.ylabel('Improvement [\\%] $\\uparrow$', fontsize=fig_fontsize)
plt.title(f'Improvements of our method w.r.t. scene difficulty', fontsize=fig_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid()
plt.xticks(fontsize=axis_fontsize)
plt.yticks(fontsize=axis_fontsize)
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
#plt.show()
#plt.xscale("log")

plt.savefig(os.path.join(results_dir, 'improvement_wrt_difficulty.pdf'), bbox_inches='tight', dpi=dpi)
plt.close()
