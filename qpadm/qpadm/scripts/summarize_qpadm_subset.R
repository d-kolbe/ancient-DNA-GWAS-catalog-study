library(dplyr)
library(tidyr)

# Redirect output to log
log_file <- snakemake@log[[1]]
con <- file(log_file, open = "wt")
sink(con, type = "output")
sink(con, type = "message")

# Get Rdata files directly from snakemake input (already resolved by snakemake)
rdata_paths <- snakemake@input[["wait_for"]]

# Left populations (sources), in the order used for the qpAdm runs
leftpops <- trimws(readLines(snakemake@input[["leftpops"]]))
leftpops <- leftpops[leftpops != ""]

cat("Number of Rdata files to process:", length(rdata_paths), "\n")

# Check which files exist
existing_files <- rdata_paths[file.exists(rdata_paths)]
missing_files  <- rdata_paths[!file.exists(rdata_paths)]

cat("Existing:", length(existing_files), "\n")
cat("Missing: ", length(missing_files),  "\n")
if (length(missing_files) > 0) {
  cat("Missing files:\n")
  cat(paste(missing_files, collapse="\n"), "\n")
}

# Function to extract ancestry proportions from each Rdata file
extract_ancestry <- function(rdata_file) {
  tryCatch({
    env <- new.env()
    load(rdata_file, envir = env)

    # Debug: print what was loaded
    cat("Loaded from", basename(rdata_file), ":", paste(ls(env), collapse=", "), "\n")

    res <- env$res
    w   <- res$weights

    # weights are reported per left population (column 2); stop if their order
    # is not the order of the leftpops file
    if (!identical(as.character(w[[2]]), leftpops)) {
      stop("left populations in the result (", paste(w[[2]], collapse = ","),
           ") differ from leftpops (", paste(leftpops, collapse = ","), ")")
    }

    out <- data.frame(ukbpop = as.character(w[[1]][1]), stringsAsFactors = FALSE)
    for (k in seq_along(leftpops)) out[[leftpops[k]]] <- w[[3]][k]
    out
  }, error = function(e) {
    cat("ERROR in", rdata_file, ":", conditionMessage(e), "\n")
    NULL
  })
}

ancestry_list <- lapply(existing_files, extract_ancestry)
ancestry_list <- Filter(Negate(is.null), ancestry_list)

cat("Successfully parsed:", length(ancestry_list), "files\n")

if (length(ancestry_list) == 0) {
  stop("No ancestry results could be parsed - check Rdata object structure above")
}

ancestry_df <- bind_rows(ancestry_list)

# Get real EIDs from the .ind file
subset   <- snakemake@params[["sub"]]
base_path <- dirname(snakemake@output[[1]])  # derives from output path, no hardcoding
ind_file  <- file.path(base_path, "eigenstrat.ind")

cat("Reading ind file:", ind_file, "\n")
ind_df <- read.table(ind_file, stringsAsFactors = FALSE, col.names = c("eid", "sex", "groupID"))

ukb_df <- ind_df %>%
  filter(grepl("^UKB", groupID)) %>%
  rename(ukbpop = groupID)

final_df <- ukb_df %>%
  select(eid, ukbpop) %>%
  inner_join(ancestry_df, by = "ukbpop") %>%
  select(eid, all_of(leftpops))

cat("UKB individuals in subset:", nrow(ukb_df), " with qpAdm result:", nrow(final_df), "\n")

write.table(final_df, file = snakemake@output[[1]], sep = "\t", quote = FALSE, row.names = FALSE)

sink(type = "message")
sink(type = "output")
close(con)