from pathlib import Path

import polars as pl


class VeffBenchmark:

    def __init__(self, annotation_path: str | Path, genotypes_path: str | Path):
        self.annotation_df = pl.scan_parquet(annotation_path).unique().rename({'gene': 'gene_id'})
        self.genotypes_df = pl.scan_parquet(genotypes_path).select(
            ['sampleId', 'chrom', 'start', 'end', 'ref', 'alt']).rename(
            {'sampleId': 'individual',
             'start': 'variant_start',
             'end': 'variant_end', })

    def run(self, variant_effects_paths: list[str | Path], output_path: str | Path):
        gene_veff_df = pl.concat([pl.scan_parquet(path) for path in variant_effects_paths]).select(
            pl.col(['tissue', 'gene_id']),
            pl.col('strand').cast(pl.Enum(['-', '+'])),
            pl.col(['chrom', 'variant_start', 'variant_end', 'ref', 'alt', 'log2fc'])
        ).with_columns(
            pl.col('gene_id').str.replace(r'([^\.]+)\..+$', "${1}").alias('gene_id'))

        veff_df = self.genotypes_df.join(
            gene_veff_df, on=['chrom', 'variant_start', 'variant_end', 'ref', 'alt'], how='inner'
        ).with_columns(
            (pl.col('log2fc').abs() == pl.col('log2fc').abs().max())
            .over(['gene_id', 'tissue', 'individual']).alias('top')
        ).filter(pl.col('top')).groupby(['gene_id', 'tissue', 'individual']).agg(
            pl.exclude('top').first()
        ).rename({'log2fc': f'predicted_log2fc', })
        res_df = self.annotation_df.join(veff_df, on=['individual', 'gene_id', 'tissue'], how='left').fill_null(0)
        res_df.collect().write_parquet(output_path)
