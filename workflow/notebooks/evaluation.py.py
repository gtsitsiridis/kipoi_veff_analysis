# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Variant-effect prediction evaluation

# %% [markdown]
# ## Setup

# %%
import os
import yaml
import pathlib
import polars as pl
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score
import plotnine as pn
import pandas as pd
from rep.utils import get_prc_curve_df
import warnings
from scipy.stats import ranksums
import seaborn as sns
import numpy as np
import scipy
from sklearn.metrics import r2_score

# %%
pn.theme_set(pn.theme_bw())
pn.theme_update(dpi=150)


# %%
def setup_dev(predictor, run_key):
    # Set working directory
    root_dir='../../'
    os.chdir(root_dir)
    
    # Read configuration files
    config_files = ['config/components.dev.yaml',
                    'config/main.dev.yaml']
    config = dict()
    for f in config_files:
        with open(f) as stream:
            config.update(yaml.safe_load(stream))
            
    # Set default wildcards
    wildcards = dict(predictor=predictor, run_key=run_key)

    # Set input files
    input_ = {
        'benchmark_path' : pathlib.Path(config["output_path"]) / 'benchmark.parquet' / f'predictor={predictor}/run={run_key}/data.parquet'
    }

    # supress warnings
    warnings.filterwarnings('ignore', category=UserWarning, append=True)

    # Set output files
    evaluation_path = pathlib.Path(config["output_path"]) / 'evaluation'
    output = dict(
        prc_path = evaluation_path / 'prc.parquet' / f'predictor={predictor}/run={run_key}/data.parquet',
        prc_tissue_path = evaluation_path / 'prc_tissue.parquet' / f'predictor={predictor}/run={run_key}/data.parquet',
        prc_tissue_type_path = evaluation_path / 'prc_tissue_type.parquet' / f'predictor={predictor}/run={run_key}/data.parquet',
        prc_fold_path = evaluation_path / 'prc_fold.parquet' / f'predictor={predictor}/run={run_key}/data.parquet',
        r2_path = evaluation_path / 'r2.parquet' / f'predictor={predictor}/run={run_key}/data.parquet',
        r2_tissue_path = evaluation_path / 'r2_tissue.parquet' / f'predictor={predictor}/run={run_key}/data.parquet',
        r2_tissue_type_path = evaluation_path / 'r2_tissue_type.parquet' / f'predictor={predictor}/run={run_key}/data.parquet',
        r2_fold_path = evaluation_path / 'r2_fold.parquet' / f'predictor={predictor}/run={run_key}/data.parquet',
    )
    for f in output.values():
        f.parent.mkdir(parents=True, exist_ok=True)
    
    return wildcards, config, input_, output


# %%
try:
    snakemake
    wildcards = snakemake.wildcards
    config = snakemake.config
    input_ = snakemake.input
    output = snakemake.output
except NameError:
    # Setup development environment if no snakemake variable is found
    wildcards, config, input_, output = setup_dev(predictor='enformer', run_key='veff1')

# %% [markdown]
# ##  Analysis

# %%
wildcards

# %%
model_name = f'{wildcards["predictor"]}-{wildcards["run_key"]}'
model_name

# %%
# read benchmark file
benchmark_path = input_['benchmark_path']
benchmark_df = pl.read_parquet(benchmark_path, hive_partitioning=False)
benchmark_df = benchmark_df.with_columns((pl.col('outlier_state') == 'underexpressed').alias('y_true'))
benchmark_df = benchmark_df.with_columns((-pl.col('veff_score')).alias('y_pred'))


# %%
def calc_prc_df(
    df,
    groupby=None,
    true_col="y_true",
    pred_col="y_pred",
    round_decimals=4
):
    if groupby is None:
        prc_df = get_prc_curve_df(
            y_trues=[df[true_col]],
            y_preds=[df[pred_col]],
            labels=["model"],
            binary_as_point=True
        ).drop(columns="model")
    else:
        true = []
        pred = []
        models = []
        for idx, group in df.groupby(groupby):
            true.append(group[true_col])
            pred.append(group[pred_col])
            models.append(idx)

        prc_df = get_prc_curve_df(
            y_trues=true,
            y_preds=pred,
            labels=models,
            binary_as_point=True
        ).rename(columns={"model": groupby})

        
    if round_decimals is not None:
        prc_df["recall"] = prc_df["recall"].round(decimals=round_decimals)
        prc_df["precision"] = prc_df["precision"].round(decimals=round_decimals)

    compare_drop_cols = ["threshold", "auc"]
    prc_df = prc_df[
        prc_df.drop(columns=compare_drop_cols)
        .ne(
            # shift by 1
            prc_df.drop(columns=compare_drop_cols).shift()
        )
        .any(axis=1)
    ]
    
    return (prc_df)

# %% [raw]
# # Is obvious outlier
# # (f.col("`vep@transcript_ablation.max`") == True)
# # | (f.col("`vep@NMD_transcript_variant.max`") == True)
# # | (f.col("`vep@LoF_HC.max`") == True)
# # | (f.col("`vep@stop_gained.max`") == True)
# # | (f.col("`vep@frameshift_variant.max`") == True)
# # | (f.col("AbExp") <= f.lit(-1.3049192425217895))

# %% [markdown]
# ### All tissues

# %%
pred_df = benchmark_df
r2_total = r2_score(pred_df["zscore"], pred_df["y_pred"])

pred_df = benchmark_df.filter(pl.col('is_obvious_outlier'))
r2_obv = r2_score(pred_df["zscore"], pred_df["y_pred"])

pred_df = benchmark_df.filter(~pl.col('is_obvious_outlier'))
r2_notobv = r2_score(pred_df["zscore"], pred_df["y_pred"])

pd.DataFrame(dict(type_=['total', 'obvious', 'not_obvious'], r2=[r2_total, r2_obv, r2_notobv])).to_parquet(output['r2_path'])

# %%
gg1 = (pn.ggplot(benchmark_df, pn.aes(x='y_pred', y='zscore')) + 
 pn.geom_bin2d(bins=100) + 
 pn.geom_smooth(method="lm", color="red") +
 pn.scale_fill_gradient(name = "Individuals", trans = "log10", breaks=[1, 10, 100, 1000], low="lightgrey", high="black") +
 pn.labs(
     x="variant-effect score", 
     y="Z-score",
     color="",
     fill="",
     title="All tissues",
 )
)

gg2 = (pn.ggplot(benchmark_df, pn.aes(x='y_pred', y='zscore')) + 
 pn.facet_wrap('is_obvious_outlier', ncol=1, labeller='label_both') +
 pn.geom_bin2d(bins=100) + 
 pn.geom_smooth(method="lm", color="red") +
 pn.scale_fill_gradient(name = "Individuals", trans = "log10", breaks=[1, 10, 100, 1000], low="lightgrey", high="black") +
 pn.labs(
     x="variant-effect score", 
     y="Z-score",
     color="",
     fill="",
     title="All tissues",
 )
)

if benchmark_df['y_pred'].unique().count() > 1:
    gg1.show()
    gg2.show()

# %%
# PR curve
total_prc_df = calc_prc_df(benchmark_df.to_pandas(), true_col='y_true', pred_col='y_pred'). \
    assign(type_='total')
obvious_prc_df = calc_prc_df(benchmark_df.filter(pl.col('is_obvious_outlier')).to_pandas(), true_col='y_true', pred_col='y_pred'). \
    assign(type_='obvious')
notobvious_prc_df = calc_prc_df(benchmark_df.filter(~pl.col('is_obvious_outlier')).to_pandas(), true_col='y_true', pred_col='y_pred'). \
    assign(type_='not_obvious')
prc_df = pd.concat([total_prc_df, obvious_prc_df, notobvious_prc_df])

# %%
# save to parquet
prc_df.to_parquet(output["prc_path"], index=False)

# %%
(pn.ggplot(prc_df, pn.aes(x='recall', y='precision')) + 
 pn.facet_wrap('type_', ncol=1) +
 pn.geom_step(direction="hv") +
 pn.geom_point(data=prc_df.query("is_binary")) +
 pn.labs(
     x="Recall", 
     y="Precision",
     color="",
     fill="",
     title="Precision-Recall curve",
 )
)

# %% [markdown]
# ### Across tissues

# %%
groupby='tissue'
r2_obv_tissue_df = benchmark_df.to_pandas().groupby([groupby, 'is_obvious_outlier'])[['zscore', 'y_pred']].apply(lambda x: r2_score(x['zscore'], x['y_pred'])).reset_index(name='r2').assign(type_='obvious')
r2_obv_tissue_df.loc[~r2_obv_tissue_df.is_obvious_outlier, 'type_'] = 'not_obvious'
del r2_obv_tissue_df['is_obvious_outlier']
r2_tissue_df = benchmark_df.to_pandas().groupby([groupby])[['zscore', 'y_pred']].apply(lambda x: r2_score(x['zscore'], x['y_pred'])).reset_index(name='r2').assign(type_='total')
r2_tissue_df = pd.concat([r2_tissue_df, r2_obv_tissue_df])

# %%
# save to parquet
r2_tissue_df.to_parquet(output["r2_tissue_path"], index=False)

# %%
(pn.ggplot(r2_tissue_df, pn.aes(x='tissue', y='r2')) + 
 pn.facet_wrap('type_', ncol=1) +
 pn.geom_bar(stat='identity') + 
 pn.theme(
    axis_text_x=pn.element_text(angle=90),
    figure_size=(8,8)
 ) +
 pn.labs(
     x="tissue", 
     y="R2",
     color="",
     fill="",
     title=f"Across {groupby}s",
 )
)

# %%
# PR curve
groupby = 'tissue'
total_prc_df = calc_prc_df(benchmark_df.to_pandas(), true_col='y_true', pred_col='y_pred', groupby=groupby). \
    assign(type_='total')
obvious_prc_df = calc_prc_df(benchmark_df.filter(pl.col('is_obvious_outlier')).to_pandas(), true_col='y_true', pred_col='y_pred', groupby=groupby). \
    assign(type_='obvious')
notobvious_prc_df = calc_prc_df(benchmark_df.filter(~pl.col('is_obvious_outlier')).to_pandas(), true_col='y_true', pred_col='y_pred', groupby=groupby). \
    assign(type_='not_obvious')

prc_df = pd.concat([total_prc_df, obvious_prc_df, notobvious_prc_df])


# %%
def pretty_mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return f'{m:.5f} +- {h:.5f}'


# %%
prc_df[["tissue","auc", "type_"]].drop_duplicates().groupby('type_')['auc'].apply(pretty_mean_confidence_interval)

# %%
# save to parquet
prc_df.to_parquet(output["prc_tissue_path"], index=False)

# %%
(pn.ggplot(prc_df, pn.aes(x='recall', y='precision', color=groupby, fill=groupby)) + 
 pn.facet_wrap('type_', ncol=1) +
 pn.geom_step(direction="hv") +
 pn.geom_point(data=prc_df.query("is_binary")) +
 pn.labs(
     x="Recall", 
     y="Precision",
     color="",
     fill="",
     title="Precision-Recall curve",
 ) +
 pn.theme(figure_size=(12,8))
)

# %% [markdown]
# ### Across tissue type

# %%
groupby='tissue_type'
r2_obv_tissue_df = benchmark_df.to_pandas().groupby([groupby, 'is_obvious_outlier'])[['zscore', 'y_pred']].apply(lambda x: r2_score(x['zscore'], x['y_pred'])).reset_index(name='r2').assign(type_='obvious')
r2_obv_tissue_df.loc[~r2_obv_tissue_df.is_obvious_outlier, 'type_'] = 'not_obvious'
del r2_obv_tissue_df['is_obvious_outlier']
r2_tissue_df = benchmark_df.to_pandas().groupby([groupby])[['zscore', 'y_pred']].apply(lambda x: r2_score(x['zscore'], x['y_pred'])).reset_index(name='r2').assign(type_='total')
r2_tissue_df = pd.concat([r2_tissue_df, r2_obv_tissue_df])

# %%
# save to parquet
r2_tissue_df.to_parquet(output[f"r2_{groupby}_path"], index=False)

# %%
(pn.ggplot(r2_tissue_df, pn.aes(x=groupby, y='r2')) + 
 pn.facet_wrap('type_', ncol=1) +
 pn.geom_bar(stat='identity') + 
 pn.theme(
    axis_text_x=pn.element_text(angle=90),
    figure_size=(8,8)
 ) +
 pn.labs(
     x=groupby, 
     y="R2",
     color="",
     fill="",
     title=f"Across {groupby}s",
 )
)

# %%
# PR curve
groupby = 'tissue_type'
total_prc_df = calc_prc_df(benchmark_df.to_pandas(), true_col='y_true', pred_col='y_pred', groupby=groupby). \
    assign(type_='total')
obvious_prc_df = calc_prc_df(benchmark_df.filter(pl.col('is_obvious_outlier')).to_pandas(), true_col='y_true', pred_col='y_pred', groupby=groupby). \
    assign(type_='obvious')
notobvious_prc_df = calc_prc_df(benchmark_df.filter(~pl.col('is_obvious_outlier')).to_pandas(), true_col='y_true', pred_col='y_pred', groupby=groupby). \
    assign(type_='not_obvious')
prc_df = pd.concat([total_prc_df, obvious_prc_df, notobvious_prc_df])


# %%
def pretty_mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return f'{m:.5f} +- {h:.5f}'


# %%
prc_df[["tissue_type","auc", "type_"]].drop_duplicates().groupby('type_')['auc'].apply(pretty_mean_confidence_interval)

# %%
# save to parquet
prc_df.to_parquet(output["prc_tissue_type_path"], index=False)

# %%
(pn.ggplot(prc_df, pn.aes(x='recall', y='precision', color=groupby, fill=groupby)) + 
 pn.facet_wrap('type_', ncol=1) +
 pn.geom_step(direction="hv") +
 pn.geom_point(data=prc_df.query("is_binary")) +
 pn.labs(
     x="Recall", 
     y="Precision",
     color="",
     fill="",
     title="Precision-Recall curve",
 ) +
 pn.theme(figure_size=(12,8))
)

# %% [markdown]
# ### Across folds

# %%
groupby='fold'
r2_obv_tissue_df = benchmark_df.to_pandas().groupby([groupby, 'is_obvious_outlier'])[['zscore', 'y_pred']].apply(lambda x: r2_score(x['zscore'], x['y_pred'])).reset_index(name='r2').assign(type_='obvious')
r2_obv_tissue_df.loc[~r2_obv_tissue_df.is_obvious_outlier, 'type_'] = 'not_obvious'
del r2_obv_tissue_df['is_obvious_outlier']
r2_tissue_df = benchmark_df.to_pandas().groupby([groupby])[['zscore', 'y_pred']].apply(lambda x: r2_score(x['zscore'], x['y_pred'])).reset_index(name='r2').assign(type_='total')
r2_tissue_df = pd.concat([r2_tissue_df, r2_obv_tissue_df])

# %%
# save to parquet
r2_tissue_df.to_parquet(output[f"r2_{groupby}_path"], index=False)

# %%
(pn.ggplot(r2_tissue_df, pn.aes(x=groupby, y='r2')) + 
 pn.facet_wrap('type_', ncol=1) +
 pn.geom_bar(stat='identity') + 
 pn.theme(
    axis_text_x=pn.element_text(angle=90),
 ) +
 pn.labs(
     x=groupby, 
     y="R2",
     color="",
     fill="",
     title=f"Across {groupby}s",
 )
)

# %%
# PR curve
groupby = 'fold'
total_prc_df = calc_prc_df(benchmark_df.to_pandas(), true_col='y_true', pred_col='y_pred', groupby=groupby). \
    assign(type_='total')
obvious_prc_df = calc_prc_df(benchmark_df.filter(pl.col('is_obvious_outlier')).to_pandas(), true_col='y_true', pred_col='y_pred', groupby=groupby). \
    assign(type_='obvious')
notobvious_prc_df = calc_prc_df(benchmark_df.filter(~pl.col('is_obvious_outlier')).to_pandas(), true_col='y_true', pred_col='y_pred', groupby=groupby). \
    assign(type_='not_obvious')
prc_df = pd.concat([total_prc_df, obvious_prc_df, notobvious_prc_df])


# %%
def pretty_mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return f'{m:.5f} +- {h:.5f}'


# %%
prc_df[["fold","auc", "type_"]].drop_duplicates().groupby('type_')['auc'].apply(pretty_mean_confidence_interval)

# %%
# save to parquet
prc_df.to_parquet(output["prc_fold_path"], index=False)

# %%
(pn.ggplot(prc_df, pn.aes(x='recall', y='precision', color=groupby, fill=groupby)) + 
 pn.facet_wrap('type_', ncol=1) +
 pn.geom_step(direction="hv") +
 pn.geom_point(data=prc_df.query("is_binary")) +
 pn.labs(
     x="Recall", 
     y="Precision",
     color="",
     fill="",
     title="Precision-Recall curve",
 )
)
