#!/bin/bash

env_vars=("MAPPER" "ISOFORM" "AGG" "UPSTREAM_LIST" "DOWNSTREAM_LIST")

for var in "${env_vars[@]}"; do
  if [[ -z "${!var}" ]]; then
    echo "$var is not set"
    exit 1
  else
    echo "$var is set"
  fi
done

for UPSTREAM in ${UPSTREAM_LIST}; do
  for DOWNSTREAM in ${DOWNSTREAM_LIST}; do
    python generate_run_config.py --name="enformer_gtexv8_${MAPPER}_${AGG}_${UPSTREAM}_${DOWNSTREAM}" \
    --set \
    predictor="enformer" \
    alternative="GRCh37_short__gtexv8_2000-500" \
    mapper="${MAPPER}_gtexv8" \
    isoform_file="${ISOFORM}" \
    aggregation_mode="${AGG}" \
    upstream_tss="${UPSTREAM}" \
    downstream_tss="${DOWNSTREAM}"
  done
done