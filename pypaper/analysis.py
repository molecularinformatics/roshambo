import random

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
    random_state=None,
    interpolation=False,
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
        # Check if there are at least two unique labels in the sample
        if len(bootstrap_sample["True Label"].unique()) < 2:
            auc_values[i] = np.nan
            roce_values[:, i] = np.nan
            continue
        else:
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
                if interpolation:
                    roce_values[j, i] = np.true_divide(np.interp(eev, fpr, tpr), eev)
                else:
                    # Calculate the index corresponding to the EEV
                    index = np.searchsorted(fpr, eev, side="right")

                    # Compute the TPR and FPR at the selected index
                    tpr_eev = tpr[index]
                    fpr_eev = fpr[index]

                    # Compute the ROCE at the selected EEV
                    roce = tpr_eev / fpr_eev
                    roce_values[j, i] = roce

    # Compute the 95% confidence interval and mean for the AUC
    ci_lower = np.nanpercentile(auc_values, 2.5)
    ci_upper = np.nanpercentile(auc_values, 97.5)
    mean_auc = np.nanmean(auc_values)

    # Compute the 95% confidence interval for the ROCE and mean at each EEV
    ci_roce_lower = np.nanpercentile(roce_values, 2.5, axis=1)
    ci_roce_upper = np.nanpercentile(roce_values, 97.5, axis=1)
    roce_mean = np.nanmean(roce_values, axis=1)

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

    # Compute FPR and TPR from full data
    fpr, tpr, thresholds = roc_curve(combined_df["True Label"], combined_df[score])
    df_rates = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    df_rates.to_csv("roc.csv", sep="\t", index=False)
    return df, df_rates


def plot_mult_roc(
    rates_dict,
    analysis_dict,
    colors_dict=None,
    title="ROC",
    figsize=(6, 5),
    filename="roc_comparison.jpg",
):
    # Checkpoint 1: Check if the lengths of rates_dict and analysis_dict are equal
    # and greater than or equal to 2
    if (
        len(rates_dict) != len(analysis_dict)
        or len(rates_dict) < 2
        or len(analysis_dict) < 2
    ):
        raise ValueError(
            "Lengths of rates_dict and analysis_dict must be equal and "
            "greater than or equal to 2"
        )

    # Checkpoint 2: Generate random colors if colors_dict is not provided or
    # is not of the same length as rates_dict and analysis_dict
    if not colors_dict or len(colors_dict) != len(rates_dict):
        colors_dict = {
            key: f"#{random.randint(0x000000, 0xFFFFFF):06x}"
            for key in rates_dict.keys()
        }

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random guess")

    for key in rates_dict.keys():
        df = pd.read_csv(rates_dict[key], sep="\t")
        auc_df = pd.read_csv(analysis_dict[key], sep="\t")
        auc_mean = auc_df.loc[auc_df["Run Name"] == "AUC", "Mean"].values[0]
        color = colors_dict[key]
        label = f"{key.capitalize()} (AUC={auc_mean:.2f})"
        ax.plot(df["FPR"], df["TPR"], label=label, color=color, linewidth=2)

    # Set axis labels and legend
    ax.set_xlabel("False Positive Rate", fontsize=18, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=18, fontweight="bold")
    ax.legend(frameon=False, fontsize=18)

    # Set title and border thickness
    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.spines["top"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)

    # Set ticks
    ax.tick_params(axis="both", which="both", direction="in", length=6, labelsize=18)

    # Save the plot
    plt.savefig(filename, dpi=500, bbox_inches="tight")


def plot_mult_auc(
    auc_dict,
    colors_dict=None,
    title="Mean AUC with 95% confidence interval",
    group_labels=None,
    figsize=(8, 6),
):
    # Checkpoint 1: Check if all values in auc_dict have the same length
    lengths = set(len(v) for v in auc_dict.values())
    if len(lengths) != 1:
        raise ValueError("All values in auc_dict must have the same length")

    # Read in AUC data from each file in the dictionary and store them in a list
    auc_data = []
    for key, paths in auc_dict.items():
        for i, path in enumerate(paths):
            df = pd.read_csv(path, sep="\t")
            mean = df.loc[df["Run Name"] == "AUC", "Mean"].values[0]
            lower = df.loc[df["Run Name"] == "AUC", "CI_Lower"].values[0]
            upper = df.loc[df["Run Name"] == "AUC", "CI_Upper"].values[0]
            auc_data.append(
                {"label": key, "mean": mean, "lower": lower, "upper": upper, "group": i}
            )

    # Sort the data by group number
    auc_data = sorted(auc_data, key=lambda x: x["group"])

    # Set the positions and width of the bars
    pos = np.arange(len(auc_data) // len(auc_dict))
    width = 0.8 / len(auc_dict)

    if group_labels and len(group_labels) != len(auc_data) // len(auc_dict):
        raise ValueError(
            "The provided group_labels must have the same length as "
            "the number of groups"
        )

    # Checkpoint 2: Generate random colors if colors_dict is not provided or
    # is not of the same length as auc_dict
    if not colors_dict or len(colors_dict) != len(auc_dict):
        colors_dict = {
            key: f"#{random.randint(0x000000, 0xFFFFFF):06x}" for key in auc_dict.keys()
        }

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=figsize)
    xs = []
    # Plot the bars and error bars for each dataset
    for i, data in enumerate(auc_data):
        x = pos[data["group"]] + i * width
        xs.append(x)
        ax.bar(
            x,
            data["mean"],
            width,
            yerr=[[data["mean"] - data["lower"]], [data["upper"] - data["mean"]]],
            color=colors_dict[data["label"]],
            label=data["label"] if i < len(auc_dict) else "",
            capsize=5,
        )

    # Set axis labels and legend
    ax.set_xlabel("Dataset", fontsize=18, fontweight="bold")
    ax.set_ylabel("Mean AUC", fontsize=18, fontweight="bold")
    ax.set_yticks(np.arange(0.2, 1.1, 0.2))
    ax.set_ylim(0, 1.1)
    ax.set_xticks(np.mean(np.array(xs).reshape(-1, len(auc_dict)), axis=1))

    if not group_labels:
        group_labels = [f"Data {i + 1}" for i in range(len(auc_data) // len(auc_dict))]
    ax.set_xticklabels(group_labels, fontsize=18)
    ax.legend(fontsize=16, frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")

    # Set title and border thickness
    ax.set_title(title, fontsize=18, fontweight="bold")
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_linewidth(2)

    # Set ticks
    ax.tick_params(direction="in", labelsize=16, length=6)

    # Save the plot
    plt.savefig("auc_plot.jpg", dpi=500, bbox_inches="tight")


def plot_mult_enrichment(
    enrich_dict,
    colors_dict=None,
    title="Enrichment factors with percentage cutoffs",
    group_labels=None,
    hatch_patterns=None,
    figsize=(8, 6),
):
    num_datasets = len(enrich_dict.keys())
    num_groups = len(next(iter(enrich_dict.values())))

    # Read in enrichment data
    enrich_data = {
        key: [pd.read_csv(path, sep="\t") for path in paths]
        for key, paths in enrich_dict.items()
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Set the positions and width of the bars
    bar_width_fraction = 0.8
    bar_width = bar_width_fraction / num_datasets
    pos = np.arange(num_groups)

    # Generate a list of hatch patterns if not provided
    if not hatch_patterns:
        hatch_patterns = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]

    # Get the number of components in the stack
    num_colors = len(
        pd.read_csv(next(iter(enrich_dict.values()))[0], sep="\t")[
            pd.read_csv(next(iter(enrich_dict.values()))[0], sep="\t")[
                "Run Name"
            ].str.contains("% Enrichment")
        ]
    )

    # Generate colors_dict if not provided or if not provided with the
    # correct number of colors
    if not colors_dict or len(colors_dict) != num_colors:
        colors_dict = {
            i: plt.cm.viridis(i / (num_colors - 1)) for i in range(num_colors)
        }

    # Plot the stacked bars for each dataset
    type_handles = []
    factor_handles = []
    for i, (key, dfs) in enumerate(enrich_data.items()):
        bottom = np.zeros(num_groups)
        hatch_pattern = hatch_patterns[i % len(hatch_patterns)]
        type_handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="w",
                hatch=hatch_pattern,
                label=key,
                edgecolor="k",
            )
        )
        for idx, df in enumerate(dfs):
            enrichments = df[df["Run Name"].str.contains("% Enrichment")]["Mean"].values
            enrichment_labels = df[df["Run Name"].str.contains("% Enrichment")][
                "Run Name"
            ].values
            for j, enrichment in enumerate(enrichments):
                ax.bar(
                    pos + i * bar_width,
                    enrichment,
                    bar_width,
                    bottom=bottom[idx],
                    color=colors_dict[j],
                    hatch=hatch_pattern,
                )
                bottom[idx] += enrichment
                if i == 0 and idx == 0:
                    factor_handles.append(
                        plt.Rectangle(
                            (0, 0),
                            1,
                            1,
                            color=colors_dict[j],
                            label=enrichment_labels[j],
                            edgecolor="k",
                        )
                    )

    # Set axis labels and legend
    ax.set_xlabel("Dataset", fontsize=18, fontweight="bold")
    ax.set_ylabel("Enrichment Factor", fontsize=18, fontweight="bold")
    ax.set_xticks(pos + bar_width * (num_datasets - 1) / 2)
    ax.set_xticklabels(group_labels, fontsize=18)

    leg1 = ax.legend(
        handles=type_handles,
        fontsize=16,
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=factor_handles,
        fontsize=16,
        frameon=False,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
    )

    # Set title and border thickness
    ax.set_title(title, fontsize=18, fontweight="bold")
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_linewidth(2)

    # Set ticks
    ax.tick_params(direction="in", labelsize=16, length=6)

    # Save the plot
    plt.savefig("enrichment.jpg", dpi=500, bbox_inches="tight")
