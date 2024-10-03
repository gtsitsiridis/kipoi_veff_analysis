import os
import argparse
import polars as pl
from pathlib import Path
import xarray as xr
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import logging

# os.environ['R_HOME'] = '/opt/modules/i12g/anaconda/envs/kipoi-veff-analysis/lib/R'

rbase = importr('base')
rdirichlet_reg = importr('DirichletReg')

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("sex_isoforms.log"),
                        logging.StreamHandler()
                    ])


def parse_args():
    parser = argparse.ArgumentParser(description="Process sex isoforms.")
    parser.add_argument('--gtex_annotation_path', type=str, required=True, help='Path to GTEx annotation file')
    parser.add_argument('--gtf_path', type=str, required=True, help='Path to GTF file')
    parser.add_argument('--gtex_transcript_tpm_path', type=str, required=True, help='Path to GTEx transcript TPMs')
    parser.add_argument('--genes_path', type=str, required=True, help='Path to genes file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output directory')
    parser.add_argument('--gene_index', type=int, default=0, help='Index of the gene to process')
    parser.add_argument('--start_row', type=int, default=0, help='Row from where to start reading')
    return parser.parse_args()


def run_dirichlet_reg(df, output_path):
    df = df.pivot(index=['sample', 'tissue', 'individual', 'sex'], columns='transcript', values='proportion')
    prop_columns = [f'proportion_{c}' for c in df.columns]
    df.columns = prop_columns
    df = df.reset_index()
    with (ro.default_converter + pandas2ri.converter).context():
        ro.r.assign('df', df)
        ro.r.assign('output', str(output_path))
        ro.r.assign('propColumns', ro.StrVector(prop_columns))
    logging.info('Running R script')
    ro.r(f'''
        print('Running test')
        df$proportion = DR_data(df[, propColumns])
        print('Fitting null model')
        nullModel = DirichReg(proportion ~ tissue, data = df)
        print('Fitting sex model')
        sexModel = DirichReg(proportion ~ tissue + sex, data = df)
        print('Likelihood-ratio test')
        anovaRes = anova(nullModel, sexModel)
        print(anovaRes)
        saveRDS(list(sexModelCoef=coef(sexModel), anovaRes=anovaRes), output)
    ''')


def main():
    args = parse_args()

    gtex_annotation_path = Path(args.gtex_annotation_path)
    gtf_path = Path(args.gtf_path)
    gtex_transcript_tpm_path = Path(args.gtex_transcript_tpm_path)
    genes_path = Path(args.genes_path)
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=False)
    gene_index = args.gene_index
    start_row = args.start_row

    genes = np.loadtxt(genes_path, dtype=str, skiprows=start_row)
    gene = genes[gene_index]

    output_file = output_path / f'{gene}.RDS'
    if output_file.exists():
        logging.info("Output file %s already exists. Skipping computation.", output_file)
        return

    logging.info('Processing gene %s', gene)

    transcript_ldf = pl.scan_parquet(gtf_path). \
        filter((pl.col('Feature') == 'transcript') & (pl.col('gene_type') == 'protein_coding')). \
        with_columns(gene_id=pl.col('gene_id').str.replace(r"([^\.]+)\..+$", "${1}"),
                     transcript_id=pl.col('transcript_id').str.replace(r"([^\.]+)\..+$", "${1}")). \
        select(['gene_id', 'transcript_id', 'Chromosome']). \
        unique(). \
        rename({'gene_id': 'gene', 'transcript_id': 'transcript', 'Chromosome': 'chrom'}). \
        filter(~pl.col('chrom').is_in(['chrX', 'chrY'])). \
        select(['gene', 'transcript', 'chrom'])
    transcript_df = transcript_ldf.collect().to_pandas().set_index('transcript')
    transcript_xrds = xr.Dataset.from_dataframe(transcript_df).set_coords(("gene", "chrom"))

    gtex_individual_ldf = pl.scan_csv(gtex_annotation_path, separator='\t'). \
        select(['INDIVIDUAL_ID', 'SEX']). \
        rename({'INDIVIDUAL_ID': 'individual', 'SEX': 'sex'}). \
        unique()
    gtex_individual_df = gtex_individual_ldf.collect().to_pandas().set_index('individual')
    gtex_individual_xrds = xr.Dataset.from_dataframe(gtex_individual_df).set_coords(("sex"))

    expression_xrds = xr.open_zarr(gtex_transcript_tpm_path)
    individuals = ['-'.join(x.split('-')[0:2]) for x in expression_xrds['sample'].data]
    expression_xrds = expression_xrds.assign_coords(dict(individual=('sample', individuals)))
    del expression_xrds['gene']
    expression_xrds = expression_xrds.sel(transcript=~expression_xrds.transcript.str.endswith('_PAR_Y'))
    transcripts = [x.split('.')[0] for x in expression_xrds.transcript.values]
    expression_xrds = expression_xrds.assign_coords(dict(transcript=transcripts))
    individuals = np.intersect1d(expression_xrds['individual'].values, gtex_individual_xrds['individual'].values)
    expression_xrds = expression_xrds.sel(sample=expression_xrds['individual'].isin(individuals))
    gtex_individual_xrds = gtex_individual_xrds.sel(individual=gtex_individual_xrds['individual'].isin(individuals))
    expression_xrds = expression_xrds.assign_coords(
        sex=('sample', gtex_individual_xrds['sex'].sel(individual=expression_xrds['individual']).values)
    )

    final_xrds = xr.merge([expression_xrds, transcript_xrds], join='inner')

    gene_selector = final_xrds['gene'] == gene
    xrds = final_xrds.sel(transcript=gene_selector)
    total_tpm = xrds['tpm'].groupby('sample').sum('transcript')
    xrds = xrds.assign(total_tpm=total_tpm,
                       proportion=xrds['tpm'] / total_tpm)
    df = xrds['proportion'].to_dataframe().reset_index()[
        ['transcript', 'sample', 'tissue', 'sex', 'proportion', 'individual']]
    logging.info('Running Dirichlet regression')
    run_dirichlet_reg(df, output_file)


if __name__ == "__main__":
    main()