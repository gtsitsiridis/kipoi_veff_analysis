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


class VeffBenchmark:

    def __init__(self, annotation_path: str | Path, genotypes_path: str | Path,
                 folds_path: str | Path, fdr_cutoff: float = 0.2):
        self.fdr_cutoff = fdr_cutoff
        self.annotation_df = pl.scan_parquet(annotation_path).unique().rename({'gene': 'gene_id'})
        self.folds_df = pl.scan_parquet(folds_path)
        self.genotypes_df = pl.scan_parquet(genotypes_path).select(
            ['sampleId', 'chrom', 'start', 'end', 'ref', 'alt']).rename(
            {'sampleId': 'individual',
             'start': 'variant_start',
             'end': 'variant_end', })

    def run(self, variant_effects_paths: list[str | Path], output_path: str | Path):
        logger.info('Benchmarking...')
        gene_veff_df = pl.concat([pl.scan_parquet(path) for path in variant_effects_paths]).select(
            pl.col(['tissue', 'gene_id']),
            pl.col('strand').cast(pl.Enum(['-', '+'])),
            pl.col(['chrom', 'variant_start', 'variant_end', 'ref', 'alt', 'veff_score'])
        ).with_columns(
            pl.col('gene_id').str.replace(r'([^\.]+)\..+$', "${1}").alias('gene_id'))

        # todo assert that the index is unique

        veff_df = self.genotypes_df.join(
            gene_veff_df, on=['chrom', 'variant_start', 'variant_end', 'ref', 'alt'], how='inner'
        ).with_columns(
            (pl.col('veff_score').abs() == pl.col('veff_score').abs().max())
            .over(['gene_id', 'tissue', 'individual']).alias('top')
        ).filter(pl.col('top')).group_by(['gene_id', 'tissue', 'individual']).agg(
            pl.exclude('top').first()
        )

        # logger.info('Checking whether the variant effect table is empty...')
        # if veff_df.collect().is_empty():
        #     logger.warning('The variant effect table is empty... All records will be predicted as normal.')

        logger.info('Joining annotation and variant effect tables...')

        annotation_folds_df = self.annotation_df.join(self.folds_df, on='individual', how='inner')
        res_df = annotation_folds_df.join(veff_df, on=['individual', 'gene_id', 'tissue'], how='left')
        # fill null with 0 for veff_score
        res_df = res_df.with_columns(pl.col('veff_score').fill_null(0))
        res_df = res_df.with_columns(
            (
                pl.when(pl.col('FDR') > self.fdr_cutoff)
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


benchmark = VeffBenchmark(annotation_path=config['benchmark']['annotation_path'],
                          genotypes_path=config['benchmark']['genotypes_path'],
                          folds_path=config['benchmark']['folds_path'],
                          fdr_cutoff=config['benchmark']['fdr_cutoff'])

benchmark.run(list(input_['veff_path']), output_path=output['benchmark_path'])
