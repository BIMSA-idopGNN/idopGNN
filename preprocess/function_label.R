# Load required libraries 
library(parallel)
library(glmnet)
library(cowplot)
library(magrittr)
library(reshape2)
library(tidyr)
library(dplyr)
library(scales)
library(viridisLite)
pacman::p_load(tidyverse, microeco, aplot, ggsci)

#data load
df_info_new = read.csv(file = "./data/df_info_new.csv",row.names = 1)
df_otu = read.csv(file = "./data/df_otu.csv",row.names = 1)
XD = read.csv(file = "./data/XD.csv",row.names = 1)
YD = read.csv(file = "./data/YD.csv",row.names = 1)
ZD = read.csv(file = "./data/ZD.csv",row.names = 1)
ED = read.csv(file = "./data/ED.csv",row.names = 1)

# Rename taxonomy columns
colnames(df_info_new)[1] <- "Kingdom"
colnames(df_info_new)[2] <- "Phylum"
colnames(df_info_new)[3] <- "Class"
colnames(df_info_new)[4] <- "Order"
colnames(df_info_new)[5] <- "Family"
colnames(df_info_new)[6] <- "Genus"

# Standardize taxonomy format using tidy_taxonomy
df_info_new %<>% tidy_taxonomy

# Read sample metadata
sample_table <- read.csv(file = "./data/phenotype.csv",
                         row.names = 1)
rownames(sample_table) = sample_table$sample
sample_table = sample_table[,-1]

# Remove samples with all missing values and reorder
sample_table = sample_table[rowSums(is.na(sample_table)) < ncol(sample_table), ]
sample_table = sample_table[order(match(rownames(sample_table), colnames(df_otu))), ]

# Prepare OTU feature table and taxonomy table
feature_table <- df_otu
rownames(df_info_new) = rownames(feature_table)
tax_table <- df_info_new

# Function to predict microbial function using microeco
predict_microbiome_func <- function(otu_df, tax_table, sample_table) {

  # Subset tax_table to OTU IDs
  otu_tax <- tax_table[rownames(otu_df), ]
  
  # Match samples in sample_table
  otu_sample <- sample_table[match(colnames(otu_df), rownames(sample_table)), ]
  
  # Create microtable object
  dataset <- microtable$new(
    sample_table = otu_sample,
    otu_table = otu_df,
    tax_table = otu_tax
  )
  
  # Create functional prediction object
  trans_obj <- trans_func$new(dataset)
  trans_obj$for_what <- 'prok'
  
  # Calculate microbial functions using FAPROTAX
  trans_obj$cal_spe_func(prok_database = "FAPROTAX")
  
  # Get functional result
  func_res <- trans_obj$res_spe_func
  
  # Remove columns with all zeros
  all_zero_col <- which(apply(func_res, 2, function(x) all(x == 0)))
  if(length(all_zero_col) > 0) {
    func_res <- func_res[, -all_zero_col]
  }
  
  return(func_res)
}

# Predict functional profiles for each batch
XD_func <- predict_microbiome_func(XD, tax_table, sample_table)
ZD_func <- predict_microbiome_func(ZD, tax_table, sample_table)
YD_func <- predict_microbiome_func(YD, tax_table, sample_table)
ED_func <- predict_microbiome_func(ED, tax_table, sample_table)

# Save OTU tables and functional prediction results

write.csv(XD_func,"./data/XD_function.csv", row.names = TRUE)
write.csv(ZD_func,"./data/ZD_function.csv", row.names = TRUE)
write.csv(YD_func,"./data/YD_function.csv", row.names = TRUE)
write.csv(ED_func,"./data/ED_function.csv", row.names = TRUE)
