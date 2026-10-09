#!/usr/bin/env python3
"""
Merge Category Blocks
Combines LD blocks from all categories into single files
"""

import pandas as pd
import os
import sys

print("="*80)
print("MERGING CATEGORY LD BLOCKS")
print("="*80)

# Get inputs from Snakemake
block_files = snakemake.input['block_files']
member_files = snakemake.input['member_files']

# Outputs
merged_blocks_out = snakemake.output['merged_blocks']
merged_members_out = snakemake.output['merged_members']

print(f"\nInput files:")
print(f"  Block files: {len(block_files)}")
print(f"  Member files: {len(member_files)}")

# =========================================================================
# Merge blocks
# =========================================================================

print(f"\nMerging blocks...")

all_blocks = []
block_id_offset = 0

for block_file in block_files:
    # Extract category name from path
    # Path format: .../ld_blocks_per_category/<category>/ld_blocks.txt
    category = os.path.basename(os.path.dirname(block_file))
    
    # Read blocks for this category
    try:
        df = pd.read_csv(block_file, sep='\t')
    except pd.errors.EmptyDataError:
        print(f"  WARNING: Empty file {block_file}")
        continue
    
    # Add category column
    df['category'] = category
    
    # Renumber block IDs to be globally unique
    df['block_id'] = df['block_id'] + block_id_offset
    block_id_offset = df['block_id'].max() + 1
    
    all_blocks.append(df)

if not all_blocks:
    print("ERROR: No blocks to merge!")
    sys.exit(1)

blocks_df = pd.concat(all_blocks, ignore_index=True)

print(f"  ✓ Total blocks: {len(blocks_df):,}")
print(f"  ✓ Categories: {blocks_df['category'].nunique()}")

# =========================================================================
# Merge block members
# =========================================================================

print(f"\nMerging block members...")

all_members = []
block_id_offset = 0

for member_file in member_files:
    # Extract category name from path
    category = os.path.basename(os.path.dirname(member_file))
    
    # Read members for this category
    try:
        df = pd.read_csv(member_file, sep='\t')
    except pd.errors.EmptyDataError:
        print(f"  WARNING: Empty file {member_file}")
        continue
    
    # Add category column
    df['category'] = category
    
    # Renumber block IDs to match blocks file
    # First, get the block ID mapping from the blocks file
    block_file = member_file.replace('block_members.txt', 'ld_blocks.txt')
    try:
        blocks_cat = pd.read_csv(block_file, sep='\t')
    except:
        continue
    
    # Create mapping of old to new block IDs
    old_to_new = {}
    for old_id in blocks_cat['block_id'].unique():
        old_to_new[old_id] = old_id + block_id_offset
    
    # Update block IDs in members
    df['block_id'] = df['block_id'].map(old_to_new)
    
    # Update offset for next category
    block_id_offset = max(old_to_new.values()) + 1
    
    all_members.append(df)

if not all_members:
    print("ERROR: No members to merge!")
    sys.exit(1)

members_df = pd.concat(all_members, ignore_index=True)

print(f"  ✓ Total SNPs: {len(members_df):,}")
print(f"  ✓ Categories: {members_df['category'].nunique()}")

# =========================================================================
# Write outputs
# =========================================================================

os.makedirs(os.path.dirname(merged_blocks_out), exist_ok=True)

blocks_df.to_csv(merged_blocks_out, sep='\t', index=False)
members_df.to_csv(merged_members_out, sep='\t', index=False)

print(f"\n✓ Wrote merged blocks to: {merged_blocks_out}")
print(f"✓ Wrote merged members to: {merged_members_out}")

print("="*80)