#!/usr/bin/env python3
"""
Calculate Fst using vcftools
Wrapper script with proper error handling and output verification
"""

import subprocess
import os
import shutil
import sys

# Get inputs from Snakemake
vcf = snakemake.input['vcf']
hg_samples = snakemake.input['hg_samples']
ef_samples = snakemake.input['ef_samples']
positions = snakemake.input['positions']
output_fst = snakemake.output['fst_results']

print("="*80)
print("CALCULATING FST (WEIR & COCKERHAM)")
print("="*80)

print(f"\nInputs:")
print(f"  VCF: {vcf}")
print(f"  HG samples: {hg_samples}")
print(f"  EF samples: {ef_samples}")
print(f"  Positions: {positions}")

# Create output directory
output_dir = os.path.dirname(output_fst)
os.makedirs(output_dir, exist_ok=True)
print(f"\nOutput directory: {output_dir}")

# Set output prefix for vcftools
output_prefix = os.path.join(output_dir, 'fst')

# Build vcftools command
cmd = [
    'vcftools',
    '--gzvcf', vcf,
    '--weir-fst-pop', hg_samples,
    '--weir-fst-pop', ef_samples,
    '--positions', positions,
    '--out', output_prefix
]

print(f"\nRunning vcftools...")
print(f"Command: {' '.join(cmd)}\n")

# Run vcftools from the output directory to ensure files are created there
result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)

# Print output
print(result.stdout)
if result.stderr:
    print(result.stderr, file=sys.stderr)

# Check if vcftools succeeded
if result.returncode != 0:
    print(f"\nERROR: vcftools failed with exit code {result.returncode}")
    sys.exit(1)

# Check for expected output file
expected_output = f"{output_prefix}.weir.fst"
current_dir_output = "fst.weir.fst"  # vcftools might create in current directory

print(f"\nLooking for output file...")
print(f"  Expected location: {expected_output}")
print(f"  Alternative (current dir): {current_dir_output}")

# Try to find the file in multiple locations
found_file = None

if os.path.exists(expected_output):
    found_file = expected_output
    print(f"  ✓ Found at expected location")
elif os.path.exists(current_dir_output):
    found_file = current_dir_output
    print(f"  ✓ Found in current directory")
elif os.path.exists(os.path.join(output_dir, "fst.weir.fst")):
    found_file = os.path.join(output_dir, "fst.weir.fst")
    print(f"  ✓ Found in output directory")

if found_file:
    # Move to final location
    shutil.move(found_file, output_fst)
    
    # Verify file was moved
    if os.path.exists(output_fst):
        file_size = os.path.getsize(output_fst)
        num_lines = sum(1 for _ in open(output_fst))
        
        print(f"\n✓ Fst results saved to: {output_fst}")
        print(f"  File size: {file_size:,} bytes")
        print(f"  Lines: {num_lines:,}")
        print("="*80)
    else:
        print(f"ERROR: Failed to move file to {output_fst}")
        sys.exit(1)
else:
    # List what files were actually created
    print(f"\nERROR: vcftools output file not found in any expected location")
    
    print(f"\nSearched locations:")
    print(f"  1. {expected_output}")
    print(f"  2. {current_dir_output}")  
    print(f"  3. {os.path.join(output_dir, 'fst.weir.fst')}")
    
    print(f"\nFiles in output directory ({output_dir}):")
    if os.path.exists(output_dir):
        for f in sorted(os.listdir(output_dir)):
            filepath = os.path.join(output_dir, f)
            size = os.path.getsize(filepath)
            print(f"  {f} ({size:,} bytes)")
    else:
        print(f"  Directory does not exist!")
    
    print(f"\nFiles in current working directory ({os.getcwd()}):")
    for f in sorted(os.listdir('.')):
        if 'fst' in f.lower() or 'weir' in f.lower():
            size = os.path.getsize(f) if os.path.isfile(f) else 0
            print(f"  {f} ({size:,} bytes)")
    
    print("\nPossible causes:")
    print("  1. vcftools output format changed")
    print("  2. vcftools failed silently") 
    print("  3. Permissions issue")
    print("  4. Check vcftools stdout/stderr above for clues")
    
    sys.exit(1)