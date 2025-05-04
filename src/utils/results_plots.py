import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df_dir = "qtt_history_logs"
for filename in os.listdir(df_dir):

    df = pd.read_csv(os.path.join(df_dir, filename))

    df['cumulative_cost'] = df['cost'].cumsum()

    plt.figure(figsize=(10, 6))
    plt.plot(df['cumulative_cost'], df['score'], marker='o', linestyle='-', color='b')

    # Adding labels and title
    plt.xlabel('Cumulative Cost (Time)')
    plt.ylabel('Score')
    plt.title('Score vs. Cumulative Cost (Time)')
    f_name = filename.split(".")[0]
    plt.savefig(f'result_plots/traj_{f_name}.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    df['config_id_str'] = df['config_id'].astype(str)
    config_counts = df['config_id_str'].value_counts()
    plt.bar(config_counts.index, config_counts.values, color='g')
    plt.xlabel('Config ID')
    plt.ylabel('Frequency')
    plt.title('Frequency of Each Config ID')

    plt.tight_layout()
    plt.savefig(f'result_plots/freq_{f_name}.png', dpi=300, bbox_inches='tight')

    # # Plotting
    # plt.figure(figsize=(10, 6))
    # sns.lineplot(x=range(len(df)), y='score', data=df, marker='o', hue='config_id', palette='tab10')

    # plt.title('HPO Tuning: Score Progression over Iterations')
    # plt.xlabel('Tuning Step')
    # plt.ylabel('Score')
    # plt.legend(title='Config ID')
    # plt.grid(True)
    # plt.tight_layout()

    # # Save instead of show
    # f_name = filename.split(".")[0]
    # plt.savefig(f'result_plots/{f_name}.png', dpi=300, bbox_inches='tight')
