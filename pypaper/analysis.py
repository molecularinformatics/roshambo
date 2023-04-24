import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score


def calc_roc_auc(
    actives_file,
    decoys_file,
    score="ComboTanimoto",
    n_bootstraps=1000,
    eevs=None,
    plot=True,
    random_state=1,
):
    actives_df = pd.read_csv(actives_file, sep="\t")
    decoys_df = pd.read_csv(decoys_file, sep="\t")

    # Create a combined dataframe with the true labels and the scores
    combined_df = pd.concat([actives_df, decoys_df], ignore_index=True)
    combined_df["True Label"] = [1] * len(actives_df) + [0] * len(decoys_df)
    combined_df.sort_values(score, ascending=False, inplace=True)

    # Define the EEVs of interest
    if not eevs:
        eevs = [0.005, 0.01, 0.02, 0.05]

    # Initialize an array to store the bootstrap AUC values
    auc_values = np.zeros(n_bootstraps)

    # Initialize an array to store the bootstrap ROCE values
    roce_values = np.zeros((len(eevs), n_bootstraps))

    # Loop over the bootstrap samples
    for i in range(n_bootstraps):
        # Sample the data with replacement
        bootstrap_sample = combined_df.sample(
            frac=1, replace=True, random_state=random_state
        )

        # Calculate the AUC for the bootstrap sample
        auc_values[i] = roc_auc_score(
            bootstrap_sample["True Label"], bootstrap_sample[score]
        )

        # Calculate the ROC curve and AUC for the bootstrap sample
        fpr, tpr, thresholds = roc_curve(
            bootstrap_sample["True Label"], bootstrap_sample[score]
        )

        # Loop over the EEVs and compute the ROCE for the bootstrap sample
        for j, eev in enumerate(eevs):
            # Calculate the index corresponding to the EEV
            index = np.searchsorted(fpr, eev, side="right")

            # Compute the TPR and FPR at the selected index
            tpr_eev = tpr[index]
            fpr_eev = fpr[index]

            # Compute the ROCE at the selected EEV
            roce = tpr_eev / fpr_eev
            roce_values[j, i] = roce

    # Compute the 95% confidence interval and mean for the AUC
    ci_lower = np.percentile(auc_values, 2.5)
    ci_upper = np.percentile(auc_values, 97.5)
    mean_auc = np.mean(auc_values)

    # Compute the 95% confidence interval for the ROCE and mean at each EEV
    ci_roce_lower = np.percentile(roce_values, 2.5, axis=1)
    ci_roce_upper = np.percentile(roce_values, 97.5, axis=1)
    roce_mean = np.mean(roce_values, axis=1)

    # Plot the ROC curve
    if plot:
        fig, ax = plt.subplots()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {mean_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random guess")
        ax.tick_params(direction="in", labelsize=16, length=6)
        for spine in ["top", "bottom", "left", "right"]:
            ax.spines[spine].set_linewidth(2)

        plt.xlabel("False Positive Rate", fontsize=18, fontweight="bold")
        plt.ylabel("True Positive Rate", fontsize=18, fontweight="bold")
        plt.title("ROC Curve", fontsize=18, fontweight="bold")
        plt.legend(fontsize=16, frameon=False)
        plt.savefig("auc_roc.jpg", dpi=500)
        plt.close()

    # Save results to a file
    df = pd.DataFrame(
        {
            "Run Name": ["AUC"] + [f"{str(i * 100)}% Enrichment" for i in eevs],
            "Mean": np.insert(roce_mean, 0, mean_auc),
            "CI_Lower": np.insert(ci_roce_lower, 0, ci_lower),
            "CI_Upper": np.insert(ci_roce_upper, 0, ci_upper),
        }
    )
    df = df.round(2)
    df.to_csv("analysis.csv", sep="\t", index=False)

    df_rates = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    df_rates.to_csv("roc.csv", sep="\t", index=False)
    return df, df_rates


# # set the positions of the bars and error bars
# pos = np.arange(len(rocs))
# width = 0.35
#
# # create the figure and axis objects
# fig, ax = plt.subplots()
#
# # plot the bars and error bars
# fig, ax = plt.subplots()
# ax.bar(pos, rocs, width, yerr=[np.array(rocs)-np.array(rocs_lower), np.array(rocs_upper)-np.array(rocs)], alpha=0.5, color='b', label='rocs')
# ax.bar(pos+width, pypaper, width, yerr=[np.array(pypaper)-np.array(pypaper_lower), np.array(pypaper_upper)-np.array(pypaper)], alpha=0.5, color='g', label='pypaper')
# ax.tick_params(direction='in', labelsize=16, length=6)
# for spine in ["top", "bottom", "left", "right"]:
#     ax.spines[spine].set_linewidth(2)
# ax.set_xlabel('Dataset', fontsize=18, fontweight="bold")
# ax.set_ylabel('Mean AUC', fontsize=18, fontweight="bold")
# ax.set_xticks(pos+width/2)
# ax.set_xticklabels(['aces', 'adrb1', 'egfr', 'jak2'], fontsize=18,)
# ax.set_title('Mean AUC with 95% confidence interval', fontsize=18, fontweight="bold")
# ax.legend(fontsize=16, frameon=False, bbox_to_anchor=(1.02, 0.5), loc='center left')
# plt.savefig("auc_rocs.jpg", dpi=500, bbox_inches="tight")
#
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# pos = np.arange(len(rocs))
# width = 0.35
#
# # create the figure and axis objects
# fig, ax = plt.subplots()
#
# # plot the bars for rocs and pypaper
# for i in range(len(rocs[0])):
#     print(i, [sum(row[:i+1]) for row in rocs])
#     ax.bar(pos, [row[i] for row in rocs], width, alpha=0.5, color='b', label='rocs' if i == 0 else None, bottom=[sum(row[:i+1]) for row in rocs])
#
# for i in range(len(pypaper[0])):
#     ax.bar(pos+width, [row[i] for row in pypaper], width, alpha=0.5, color='g', label='pypaper' if i == 0 else None, bottom=[sum(row[:i+1]) for row in pypaper])
#
# # set the axis labels and title
# ax.set_ylabel('Values')
# ax.set_xticks(pos+width/2)
# ax.set_xticklabels(['1', '2', '3', '4'])
# ax.set_title('Stacked Bar Plot')
#
# # add a legend
# ax.legend()
#

# fig, ax = plt.subplots()
# width = 0.35
# x = np.arange(4)
# pos1 = x - width / 2
# pos2 = x + width / 2
# bars1 = ax.bar(pos1, rocs[0], width, color="#807FFF", label='1% Enrichment')
# bars2 = ax.bar(pos1, rocs[1], width, color="#7FC080", bottom=rocs[0], label='2% Enrichment')
# bars3 = ax.bar(pos1, rocs[2], width, color="gray", bottom=rocs[0]+rocs[1], label='5% Enrichment')
# bars4 = ax.bar(pos2, pypaper[0], width, color="#807FFF",)
# bars5 = ax.bar(pos2, pypaper[1], width, color="#7FC080", bottom=pypaper[0])
# bars6 = ax.bar(pos2, pypaper[2], width, color="gray", bottom=pypaper[0]+pypaper[1])
# ax.tick_params(direction='in', labelsize=16, length=6)
# for spine in ["top", "bottom", "left", "right"]:
#     ax.spines[spine].set_linewidth(2)
# ax.set_xlabel('Dataset', fontsize=18, fontweight="bold")
# ax.set_ylabel('Enrichment Factor', fontsize=18, fontweight="bold")
# ax.set_xticks(pos1+width/2)
# ax.set_xticklabels(['aces', 'adrb1', 'egfr', 'jak2'], fontsize=18,)
# ax.set_title('ROCS vs. PYPAPER', fontsize=18, fontweight="bold")
# ax.legend(fontsize=16, frameon=False, bbox_to_anchor=(1.02, 0.5), loc='center left')
# plt.savefig("auc_rocs2.jpg", dpi=500, bbox_inches="tight")
