import pyranges as pr

# SNAKEMAKE SCRIPT
params = snakemake.params
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
config = snakemake.config
genome_config = config['genomes'][wildcards['genome']]

print('Reading GTF file: %s', input_['gtf_file'])
gtf = pr.read_gtf(input_['gtf_file'], as_df=True, duplicate_attr=True)
chromosomes = genome_config['chromosomes']
gtf = gtf.query('`Chromosome`.isin(@chromosomes)')
gtf.to_parquet(output[0])
