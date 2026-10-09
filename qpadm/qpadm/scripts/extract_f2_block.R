library(tidyverse)
library(admixtools)
library(dplyr)
library(tidyr)
library(purrr)

### add error logging
sink(snakemake@log[[1]], append=TRUE, split=TRUE)

# Snakemake inputs
prefix <- snakemake@params$prefix
f2_dir <- snakemake@params$f2_dir
results_dir = snakemake@params$qpadm_dir
target_pop <- snakemake@params$pop_in



rightpops = read_csv(snakemake@input$rightpops, col_names=F) %>% pull(X1)
leftpops = read_csv(snakemake@input$leftpops, col_names=F) %>% pull(X1)

## output
output_file <- snakemake@output$result

### make output dirs
dir.create(results_dir, recursive = TRUE, showWarnings = TRUE)
dir.create(f2_dir, recursive = TRUE, showWarnings = FALSE)


### scripts start here
extract_f2(prefix, f2_dir, pops=c(target_pop, rightpops, leftpops), overwrite=TRUE, n_cores=1)

## load f2 blocks
blocks <- f2_from_precomp(f2_dir)

# Run qpAdm
res <- qpadm(blocks, c(leftpops), rightpops, target_pop)

save(res, file = output_file)
