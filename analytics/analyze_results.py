import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['figure.figsize'] = (18.55, 9.86)
matplotlib.rcParams['axes.titlesize'] = 'medium'

import matplotlib.pyplot as plt
import os
from analytics.utils import get_experiment_files


# -- Load experiment Data
def export_results(experiment_path):
    data, cfgs, df = get_experiment_files(experiment_path, files={"log.csv": "read_csv"})

    experiments_group = df.groupby("experiment_id", sort=False)

    save_path = os.path.join(experiment_path, "analysis")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    not_plot = ['frames', 'run_id', 'run_index', 'update', 'cfg_id', 'comment', 'commit', 'experiment',
                'extra_logs', 'out_dir', 'resume', 'title', '_experiment_parameters.environment',  'run_name', 'cfg_dir']

    plot_data = set(df.columns) - set(not_plot)
    plot_data = [x for x in plot_data if "." not in x]
    x_axis = "frames"

    for plot_p in plot_data:
        size = int(pow(experiments_group.ngroups, 1/2))+1
        plt.figure()
        # Iterate through continents
        share_ax = None
        for i, (experiment_name, exp_gdf) in enumerate(experiments_group):
            # create subplot axes in a 3x3 grid
            ax = plt.subplot(size, size, i + 1)  # n rows, n cols, axes position
            if share_ax is None:
                share_ax = ax
            # plot the continent on these axes
            exp_gdf.groupby("run_id").plot(x_axis, plot_p, ax=ax, legend=False)
            # set the title
            ax.set_title(exp_gdf.iloc[0].title[13:-3])
            # set the aspect
            # adjustable datalim ensure that the plots have the same axes size
            # ax.set_aspect('equal', adjustable='data lim')
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 3))

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0.25)
        plt.savefig(f"{save_path}/{plot_p}.png")
        plt.close()


if __name__ == "__main__":
    experiment_path = "results/2019Mar26-170306_multiple_envs_icm/"
    export_results(experiment_path)
