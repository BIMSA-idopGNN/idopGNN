# Clear workspace 
rm(list=ls())

# Load required libraries
library(parallel)
library(glmnet)

# Read OTU table
df_bacteria = read.csv(file = "./data/16s_otu.csv",
                       row.names = 1)

# Extract metadata columns
df_info_bac = df_bacteria[,224:ncol(df_bacteria)]

# Fix taxonomy for uncultured Lactobacillus
lac = which(df_info_bac$species == "uncultured_Lactobacillus_sp.")
for (i in 1:length(lac)) {
  df_info_bac[lac[i],6:8] = c("Lactobacillales","Lactobacillaceae","Lactobacillus")
}
df_info_bac[lac,6:9]

# Fix taxonomy for uncultured Virgibacillus
vir = which(df_info_bac$species == "uncultured_Virgibacillus_sp.")
df_info_bac[vir,8] = "Virgibacillus"
df_info_bac[vir,6:9]

# Use genus annotation for aggregation
df_annotation = df_info_bac$genus

# Aggregate OTU counts by genus
df_raw = aggregate(df_bacteria[,1:223], by=list(df_annotation), FUN = 'sum')

# Aggregate metadata by genus
df_info_bac_new = aggregate(df_info_bac, by=list(df_annotation), FUN = 'unique')
df_info_bac_new = df_info_bac_new[-1,-c(1:3,ncol(df_info_bac_new))]
df_info_bac_new[] <- lapply(df_info_bac_new, function(x) {
  if (is.list(x)) {
    sapply(x, function(y) paste(unlist(y), collapse = ";"))
  } else {
    x
  }
})

# Merge "Unassigned" and blank genus entries
df_raw[which(df_raw$Group.1=="Unassigned"),-1] = df_raw[which(df_raw$Group.1=="Unassigned"),-1] + 
  df_raw[which(df_raw$Group.1==""),-1]
df_raw = df_raw[-which(df_raw$Group.1==""),]

# Set row names and reorder columns by total counts
rownames(df_raw) = df_raw[,1]
df_raw = df_raw[,-1]
df_otu_bac = df_raw[,order(colSums(df_raw))]

# Merge any remaining empty row names into "Unassigned"
if (any(rownames(df_otu_bac)=='')) {
  df_otu_bac[which(rownames(df_otu_bac)=='Unassigned'),] = df_otu_bac[which(rownames(df_otu_bac)=='Unassigned'),] +
    df_otu_bac[which(rownames(df_otu_bac)==''),]
  df_otu_bac = df_otu_bac[-which(rownames(df_otu_bac)==''),]
}

# Function to split OTU table by batch and filter rows with too many zeros
split_batch <- function(df_otu_bac, remove_nopheno = TRUE, pheno_path) {
  
  # Remove samples without phenotype if requested
  if (remove_nopheno == TRUE) {
    pheno = read.csv(file = pheno_path, row.names = 1)
    missing_pheno = which(sapply(pheno[, 2], is.na))
    missing_sample = pheno$sample[missing_pheno]
    df_otu_bac = df_otu_bac[, -match(missing_sample, colnames(df_otu_bac))]
  }
  
  # Extract first two letters from sample names to define batches
  tmp = sapply(colnames(df_otu_bac), strsplit, '')
  tmp2 = sapply(tmp, function(x) paste0(x[1:2], collapse = ''))
  
  # Identify unique batches containing "D"
  batch = unique(tmp2)
  batch = batch[grep("D", batch)]
  
  # Split OTU table by batch
  all_no = sapply(batch, function(x) which(tmp2 == x))
  df_otu_bac2 = lapply(all_no, function(x) df_otu_bac[, x])
  df_otu_bac2 = df_otu_bac2[order(names(df_otu_bac2))]
  
  # Filter rows by number of zeros
  df_otu_bac2 <- lapply(names(df_otu_bac2), function(name) {
    df <- df_otu_bac2[[name]]
    zero_counts <- rowSums(df == 0)
    if (name %in% c("XD", "ZD")) {
      df <- df[zero_counts <= ncol(df) - 3, , drop = FALSE]
    } else {
      df <- df[zero_counts < ncol(df) - 5, , drop = FALSE]
    }
    df
  })
  
  names(df_otu_bac2) <- names(all_no[order(names(all_no))])
  return(df_otu_bac2)
}

# Split OTU table into batches
df_batch = split_batch(df_otu_bac, remove_nopheno = TRUE,
                       pheno_path = "./data/phenotype.csv")

# Assign batches to variables
ED = df_batch[[1]]
XD = df_batch[[2]]
YD = df_batch[[3]]
ZD = df_batch[[4]]

# Function to select common species and filter rows with too many zeros
sample_batch <- function(df_otu, remove_nopheno = TRUE, pheno_path) {
  
  # Remove samples without phenotype if requested
  if (remove_nopheno == TRUE) {
    pheno = read.csv(file = pheno_path, row.names = 1)
    missing_pheno = which(sapply(pheno[, 2], is.na))
    missing_sample = pheno$sample[missing_pheno]
    df_otu = df_otu[, -match(missing_sample, colnames(df_otu))]
  }
  
  # Extract first two letters from sample names to define batches
  tmp = sapply(colnames(df_otu), strsplit, '')
  tmp2 = sapply(tmp, function(x) paste0(x[1:2], collapse = ''))
  
  # Identify unique batches containing "D"
  batch = unique(tmp2)
  batch = batch[grep("D", batch)]
  
  # Split OTU table by batch
  all_no = sapply(batch, function(x) which(tmp2 == x))
  df_otu2 = lapply(all_no, function(x) df_otu[, x])
  df_otu2 = df_otu2[order(names(df_otu2))]
  
  # Filter rows by number of zeros
  df_otu2 <- lapply(names(df_otu2), function(name) {
    df <- df_otu2[[name]]
    zero_counts <- rowSums(df == 0)
    df <- df[zero_counts <= ncol(df) - 3, , drop = FALSE]
  })
  
  names(df_otu2) <- names(all_no[order(names(all_no))])
  
  # Find common microbes (row names) present in all four data frames
  common_microbes <- Reduce(intersect, list(rownames(df_otu2[[1]]), rownames(df_otu2[[2]]), rownames(df_otu2[[3]]), rownames(df_otu2[[4]])))
  length(common_microbes)  # Check the number of common microbes

  # Subset each data frame to only include the common microbes
  ED_common <- df_otu2[[1]][common_microbes, , drop=FALSE]
  XD_common <- df_otu2[[2]][common_microbes, , drop=FALSE]
  YD_common <- df_otu2[[3]][common_microbes, , drop=FALSE]
  ZD_common <- df_otu2[[4]][common_microbes, , drop=FALSE]

  # Concatenate the four data frames column-wise
  all_data <- cbind(ED_common, XD_common, YD_common, ZD_common)
    
  result = list(df_otu2,all_data) 
  return(result)
}

result = sample_batch(df_otu_bac, remove_nopheno = TRUE,
                       pheno_path = "./data/phenotype.csv")

all_data = result[[2]]
                   
df_ZJcx = read.csv(file = "./data/ZJcx.csv",row.names = 1)

#Integrating microbial community data
df_info_ZJcx = df_ZJcx[,225:ncol(df_ZJcx)]

df_annotation = df_info_ZJcx$genus

df_raw = aggregate(df_ZJcx[,1:222], by=list(df_annotation), FUN = 'sum')

df_raw[which(df_raw$Group.1=="Unassigned"),-1] = df_raw[which(df_raw$Group.1=="Unassigned"),-1] + df_raw[which(df_raw$Group.1=="unidentified"),-1]

rownames(df_raw) = df_raw[,1]

df_raw = df_raw[,-1]
df_raw = df_raw[-which(rownames(df_raw)=="unidentified"),]

df_otu_ZJcx = df_raw[,order(colSums(df_raw))]

if (any(rownames(df_otu_ZJcx)=='')) {
  df_otu_ZJcx[which(rownames(df_otu_ZJcx)=='Unassigned'),] = df_otu_ZJcx[which(rownames(df_otu_ZJcx)=='Unassigned'),] +
    df_otu_ZJcx[which(rownames(df_otu_ZJcx)==''),]
  df_otu_ZJcx = df_otu_ZJcx[-which(rownames(df_otu_ZJcx)==''),]
}

result2 = sample_batch(df_otu_ZJcx, remove_nopheno = TRUE,
                       pheno_path = "./data/phenotype.csv")

ED_ZJcx = result2[[1]][[1]]
XD_ZJcx = result2[[1]][[2]]
YD_ZJcx = result2[[1]][[3]]
ZD_ZJcx = result2[[1]][[4]]

ED_all = rbind(ED,ED_ZJcx)
XD_all = rbind(XD,XD_ZJcx)
YD_all = rbind(YD,YD_ZJcx)
ZD_all = rbind(ZD,ZD_ZJcx)

#sort data and calculate the overall value
site = order(colSums(ED_all));ED_all = ED_all[,site];ED = ED[,site];ED = rbind(ED,(log10(colSums(ED_all) + 1)));rownames(ED)[nrow(ED)] = "x"
site = order(colSums(XD_all));XD_all = XD_all[,site];XD = XD[,site];XD = rbind(XD,(log10(colSums(XD_all) + 1)));rownames(XD)[nrow(XD)] = "x"
site = order(colSums(YD_all));YD_all = YD_all[,site];YD = YD[,site];YD = rbind(YD,(log10(colSums(YD_all) + 1)));rownames(YD)[nrow(YD)] = "x"
site = order(colSums(ZD_all));ZD_all = ZD_all[,site];ZD = ZD[,site];ZD = rbind(ZD,(log10(colSums(ZD_all) + 1)));rownames(ZD)[nrow(ZD)] = "x"
site = order(colSums(all_data));all_data = all_data[,site];all_data = rbind(all_data,(log10(colSums(all_data) + 1)));rownames(all_data)[nrow(all_data)] = "x"

#save clean data
save(ED, XD, YD, ZD, all_data, file = "./data/clean_data.RData")
write.csv(ED,"./data/ED.csv", row.names = TRUE)
write.csv(XD,"./data/XD.csv", row.names = TRUE)
write.csv(YD,"./data/YD.csv", row.names = TRUE)
write.csv(ZD,"./data/ZD.csv", row.names = TRUE)
write.csv(all_data,"./data/all_data.csv", row.names = TRUE)
write.csv(df_otu_bac,"./data/df_otu.csv", row.names = TRUE)
write.csv(df_info_bac_new,"./data/df_info_new.csv", row.names = TRUE)
