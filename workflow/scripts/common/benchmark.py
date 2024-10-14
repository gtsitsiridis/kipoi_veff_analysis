from pathlib import Path
import polars as pl
import logging

# SNAKEMAKE SCRIPT
params = snakemake.params
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
config = snakemake.config

logger = logging.getLogger()

# variables
benchmark_name = wildcards['benchmark_name']
benchmark_config = config['benchmarks'][benchmark_name]
tissues = benchmark_config['tissues']
fdr_cutoff = benchmark_config['fdr_cutoff']
annotation_df = pl.scan_parquet(benchmark_config['annotation_path']).unique().rename({'gene': 'gene_id'})
folds_df = pl.scan_parquet(benchmark_config['folds_path'])
genotypes_df = pl.scan_parquet(benchmark_config['genotypes_path'], hive_partitioning=True).select(
    ['sampleId', 'chrom', 'start', 'end', 'ref', 'alt']).rename(
    {'sampleId': 'individual',
     'start': 'variant_start',
     'end': 'variant_end', })
variant_effects_paths = list(input_['veff_path'])
output_path = output['benchmark_path']

# benchmarking
logger.info('Benchmarking...')
# load gene specific variant effect data
gene_veff_df = pl.concat([pl.scan_parquet(path) for path in variant_effects_paths]).select(
    pl.col(['tissue', 'gene_id']),
    pl.col('strand').cast(pl.Enum(['-', '+'])),
    pl.col(['chrom', 'variant_start', 'variant_end', 'ref', 'alt', 'veff_score'])
).with_columns(
    pl.col('gene_id').str.replace(r'([^\.]+)\..+$', "${1}").alias('gene_id'),
    pl.col('tissue').cast(pl.Utf8())
)

# split gene_veff to tissue specific and non-tissue specific
tissue_gene_veff_df = gene_veff_df.filter(pl.col('tissue').is_not_null()). \
    filter(pl.col('tissue').is_in(tissues)). \
    select(['chrom', 'strand', 'gene_id', 'tissue', 'variant_start', 'variant_end', 'ref',
            'alt', 'veff_score'])

# populate non-tissue specific gene_veff with all tissues
non_tissue_gene_veff_df = gene_veff_df.filter(pl.col('tissue').is_null()). \
    select(pl.exclude('tissue')). \
    join(pl.DataFrame({'tissue': tissues}).lazy(), on=None, how='cross'). \
    select(['chrom', 'strand', 'gene_id', 'tissue', 'variant_start', 'variant_end', 'ref',
            'alt', 'veff_score'])

# concatanate tissue specific and non-tissue specific gene_veff
gene_veff_df = pl.concat([tissue_gene_veff_df, non_tissue_gene_veff_df])

# todo assert that the index is unique

# join genotypes with variant effect data
veff_df = genotypes_df.join(
    gene_veff_df, on=['chrom', 'variant_start', 'variant_end', 'ref', 'alt'], how='inner'
).with_columns(
    (pl.col('veff_score').abs() == pl.col('veff_score').abs().max())
    .over(['gene_id', 'tissue', 'individual']).alias('top')
).filter(pl.col('top')).group_by(['gene_id', 'tissue', 'individual']).agg(
    pl.exclude('top').first()
)

logger.info('Joining annotation and variant effect tables...')

# join genome annotatino with folds and variant effect data
annotation_folds_df = annotation_df.join(folds_df, on='individual', how='inner')
res_df = annotation_folds_df.join(veff_df, on=['individual', 'gene_id', 'tissue'], how='left')
# fill null with 0 for veff_score
res_df = res_df.with_columns(pl.col('veff_score').fill_null(0))
res_df = res_df.with_columns(
    (
        pl.when(pl.col('FDR') > fdr_cutoff)
        .then(pl.lit('normal'))
        .otherwise(
            pl.when(pl.col('zscore') > 0)
            .then(pl.lit('overexpressed'))
            .otherwise(
                pl.when(pl.col('zscore') < 0)
                .then(pl.lit('underexpressed'))
                # this should never be the case
                .otherwise(pl.lit('CHECK'))
            )
        )
    ).cast(pl.Enum(['underexpressed', 'normal', 'overexpressed'])).alias('outlier_state')
)
logger.info('Collecting results...')
res_df.collect().write_parquet(output_path)
