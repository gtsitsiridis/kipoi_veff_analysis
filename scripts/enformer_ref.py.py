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
# # Enformer reference scores

# %% [markdown]
# ## Process
# 1) Run enformer on all reference transcripts and extract raw enformer tracks for the region around the TSS
# 2) Aggregate bins around the TSS into TSS tracks
# 3) Use tissue mapper to map the raw enformer tracks to tissue specific scores for each TSS
