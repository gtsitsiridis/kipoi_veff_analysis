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
# # Enformer VCF scores

# %% [markdown]
# ## Process
# 1) For each transcript, extract all variants that fall within the given range around the TSS.
# 2) For each alternative transcript, run enformer on all alternative transcripts and extract raw enformer tracks for the region around the TSS
# 3) Aggregate bins around the TSS into TSS tracks
# 4) Use tissue mapper to map the raw enformer tracks to tissue specific scores for each TSS
# 5) Join the alternative TSS scores with the corresponding refference TSS scores and calculate the fold change
