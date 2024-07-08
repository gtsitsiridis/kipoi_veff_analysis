import yaml

# SNAKEMAKE SCRIPT
config = snakemake.config
output = snakemake.log[0]

print(config)

with open(str(output), 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
