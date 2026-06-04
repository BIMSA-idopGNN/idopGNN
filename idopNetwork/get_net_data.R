rm(list = ls())
library(igraph)
library(Cairo)
library(dplyr)
 
args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  arg_name <- args[1]
  
  if (file.exists(arg_name)) {
    load(arg_name)
    data_name <- tools::file_path_sans_ext(basename(arg_name))
  } else if (file.exists(paste0("./data/", arg_name, "_qdODE_result.RData"))) {
    load(paste0("./data/", arg_name, "_qdODE_result.RData"))
    load(paste0("./data/", arg_name, "_fit_result.RData"))
    data_name <- arg_name
  } else {
    stop("qdODE_result.RData don't exist。Please check if there are any fitting failures in the Allometric scaling law fit graph.")
  }
  
} else {
  load("./data/XD_fit_result.RData")
  load("./data/XD_qdODE_result.RData")
  data_name <- "XD"
}

cat("Loaded data from:", data_name, "\n")
normalization <- function(x, z = 0.2){(x-min(x))/(max(x)-min(x))+z}

network_conversion <- function(result){

  n = ncol(result$fit)

  effect.mean = apply(result$fit,2,mean)[4:n]

  effect.predict.mean = apply(result$predict,2,mean)[4:n]

  effect.total = colSums(result$fit)[4:n]
  
  temp = matrix(NA,nrow = n-3, ncol=3)

  colnames(temp) = c("From", "To", "Effect")

  temp[,1] = colnames(result$fit)[4:n]

  temp[,2] = colnames(result$fit)[4]

  temp[,3] = effect.predict.mean
  if (nrow(temp)==2) {
    temp = t(data.frame(temp[-1,]))
  } else{

    temp = data.frame(temp[-1,])
  }

  temp2 = matrix(NA,nrow = n-3, ncol=3)

  colnames(temp2) = c("From", "To", "Effect")

  temp2[,1] = colnames(result$fit)[4:n]

  temp2[,2] = colnames(result$fit)[4]
 
  temp2[,3] = effect.total
  if (nrow(temp2)==2) {
    temp2 = t(data.frame(temp2[-1,]))
  } else{

    temp2 = data.frame(temp2[-1,])
  }

  output <- list(ind.name = colnames(result$fit)[4],

                 dep.name = colnames(result$fit)[5:n],

                 ODE.par = result$ODE.value,
  
                 ind.par = result$LOP_par[,3],
 
                 dep.par = result$LOP_par[,4:(n-1)],

                 effect.mean = effect.predict.mean,

                 effect.total = effect.total,

                 effect.all = result$fit,

                 edge = temp,

                 edge.total = temp2,

                 ind.effect = effect.predict.mean[1])
  return(output)
}

net_all = lapply(qdODE_result$ode_result, network_conversion)

network_maxeffect <- function(result){
  module = result[[1]]
  after <- data.frame(do.call(rbind, lapply(module, "[[", "edge.total")))
  after$Effect = as.numeric(after$Effect)
  Module.maxeffect = max(abs(after$Effect))
  
  if (length(result) == 2) {
    after <- data.frame(do.call(rbind, lapply(result[[2]], "[[", "edge")))
    after$Effect = as.numeric(after$Effect)
    res.maxeffect = max(abs(after$Effect))
  } else{
    after = lapply(2:length(result),function(c) data.frame(do.call(rbind, lapply(result[[c]], "[[", "edge"))))
    after = do.call(rbind,after)
    after$Effect = as.numeric(after$Effect)
    res.maxeffect = max(abs(after$Effect))
  }
  maxeffect = max(c(Module.maxeffect,res.maxeffect))
  return(maxeffect)
}

network_data <- function(result, maxeffect = NULL, type = NULL){

  extra <- sapply(result,"[[", "ind.effect")
  if (is.null(type)) {
    after <- data.frame(do.call(rbind, lapply(result, "[[", "edge")))
  } else{
    after <- data.frame(do.call(rbind, lapply(result, "[[", "edge.total")))
    
  }
  after$Effect = as.numeric(after$Effect)
  negative = which(after$Effect<0)
  rownames(after) = NULL
  
  nodes <- data.frame(unique(after[,2]),unique(after[,2]),extra)
  colnames(nodes) <- c("id","name","ind_effect")
  
  nodes$influence <- aggregate(Effect ~ To, data = after, sum)[,2]

  if (is.null(maxeffect)) {
    after[,3] <- normalization(abs(after[,3]))
    nodes[,3:4] <- normalization(abs(nodes[,3:4]))
  } else{
    after[,3] <- (abs(after[,3]))/maxeffect*1.5+0.1
    nodes[,3:4] <- (abs(nodes[,3:4]))/maxeffect*1.5+0.1
  }
  after[after == "Unassigned"] <- "bec_Unassigned"
  after[,3][negative] = -after[,3][negative]
  after$From = sub("^g\\_","",after$From)
  after$To = sub("^g\\_","",after$To)
  nodes$name = sub("^g\\_","",nodes$name)
  nodes$id = sub("^g\\_","",nodes$id)
  result = list(nodes = nodes, edge = after)
  return(result)
}

all_result = network_data(net_all, maxeffect = NULL, type = NULL)
end_result = all_result$edge
label_data <- read.csv(paste0("./data/", "process data/", data_name, "/", data_name, "_list.csv"))
end_result$From = gsub(" ", ".", end_result$From)
end_result$To = gsub(" ", ".", end_result$To)
label_data$s = gsub(" ", ".", label_data$s)

for (i in 1:nrow(end_result)) {
  end_result$From_numeric[i] = label_data$id[which(unique(label_data$s) == end_result$From[i])]
  end_result$To_numeric[i] = label_data$id[which(label_data$s == end_result$To[i])]
}


raw_data = as.data.frame(result_all$original_data)
data_sorted <- raw_data %>%
  mutate(order = match(rownames(raw_data), label_data$s)) %>% 
  arrange(order) %>% 
  select(-order)

if (arg_name == "all_data"){
  path = paste0("./data/","phenotype.csv")
  nodes_label = read.csv(path,row.names = 1)
  rownames(nodes_label) = nodes_label[,1]
  nodes_label = nodes_label[,-1]
  nodes_label = nodes_label[match(colnames(data_sorted),rownames(nodes_label)),]
  label_data <- read.csv(paste0("./data/", "process data/", data_name, "/", data_name, "_sample_list.csv"))
}else{
  path = paste0("./data/", data_name, "_function.csv")
  nodes_label = read.csv(path,row.names = 1)
  nodes_label = nodes_label[match(rownames(data_sorted),rownames(nodes_label)),]
  rownames(nodes_label)[which(rownames(data_sorted) != rownames(nodes_label))] = rownames(data_sorted)[which(rownames(data_sorted) != rownames(nodes_label))]
  nodes_label[is.na(nodes_label[,1]),] = rep(0,ncol(nodes_label))
}

nodes_label <- nodes_label %>%
  mutate(order = match(rownames(nodes_label), label_data$s)) %>%
  arrange(order) %>%
  select(-order)

path = paste0("./data/", data_name, "_func_nodeslabel_data.csv")
write.csv(nodes_label, file = path, row.names = TRUE)

path = paste0("./data/", data_name, "_nodesfeature_data.csv")
write.csv(data_sorted, file = path, row.names = TRUE)


path = paste0("./data/", data_name, "_row_data.csv")
write.csv(end_result, file = path, row.names = FALSE)
path = paste0("./data/", data_name, "_edgeindex_data.csv")
write.csv(end_result[,4:5], file = path, row.names = FALSE)
path = paste0("./data/", data_name, "_edgefeature_data.csv")
write.csv(end_result[,3], file = path, row.names = FALSE)

network_plot <- function(result, title = NULL, maxeffect = NULL, type = NULL,data_name,plot_type,point_size,title_size,select_type,plot_size,change_name= NULL,change_x= NULL,change_y= NULL){
  
  extra <- sapply(result,"[[", "ind.effect")
  
  
  if (is.null(type)) {
    
    after <- data.frame(do.call(rbind, lapply(result, "[[", "edge")))
  } else{
    
    after <- data.frame(do.call(rbind, lapply(result, "[[", "edge.total")))
    
  }
  
  after$Effect = as.numeric(after$Effect)
  
  rownames(after) = NULL
  
  after$edge.colour = NA
  for (i in 1:nrow(after)) {
    if(after$Effect[i]>=0){
      after$edge.colour[i] = "#FE433C"
    } else{
      after$edge.colour[i] = "#0095EF"
    }
  }
  
  nodes <- data.frame(unique(after[,2]),unique(after[,2]),extra)
  colnames(nodes) <- c("id","name","ind_effect")
  
  nodes$influence <- aggregate(Effect ~ To, data = after, sum)[,2]
  nodes$node.colour = NA
  
  
  for (i in 1:nrow(nodes)) {
    if(nodes$influence[i]>=0){
      nodes$node.colour[i] = "#FBC99A"
    } else{
      nodes$node.colour[i] = "#6FDCB5"
    }
  }
  
  
  if (is.null(maxeffect)) {
    after[,3] <- normalization(abs(after[,3]))
    nodes[,3:4] <- normalization(abs(nodes[,3:4]))
  } else{
    after[,3] <- (abs(after[,3]))/maxeffect*1.5+0.1
    nodes[,3:4] <- (abs(nodes[,3:4]))/maxeffect*1.5+0.1
  }
  
  after$From = sub("^g\\_","",after$From)
  after$To = sub("^g\\_","",after$To)
  nodes$name = sub("^g\\_","",nodes$name)
  nodes$id = sub("^g\\_","",nodes$id)
  
  net <- graph_from_data_frame( d=after,vertices = nodes,directed = T )
  
  if (plot_type == "fr"){
    l <- layout_with_fr(net)
  }
  if (plot_type == "kk"){
    l <- layout_with_kk(net)
  }
  if (plot_type == "circle"){
    l <- layout_in_circle(net)
  }
  if (plot_type == "randomly"){
    l <- layout_randomly(net)
  }
  if (plot_type == "grid"){
    l <- layout_on_grid(net)
  }
  if (plot_type == "drl"){
    l <- layout_with_drl(net)
  }
  row.names(l) = names(result)
  if(!is.null(change_name)){
    if(length(change_name) == 1){
      l[which(row.names(l)==change_name),] = c(l[which(row.names(l)==change_name),][1]+change_x,l[which(row.names(l)==change_name),][2]+change_y)
    }
    if(length(change_name) == 2){
      l[which(row.names(l)==change_name[1]),] = c(l[which(row.names(l)==change_name[1]),][1]+change_x,l[which(row.names(l)==change_name[1]),][2]+change_y)
      l[which(row.names(l)==change_name[2]),] = c(l[which(row.names(l)==change_name[2]),][1]-change_x,l[which(row.names(l)==change_name[2]),][2]+change_y)
    }
  }
  
  save_path="./data/"
  plot_name=paste(save_path,data_name,"network_plot_", plot_type,select_type ,".png", sep = "")
  CairoPNG(plot_name, width=plot_size, height=plot_size, bg="white", res=300)
  plot.igraph(net,
              vertex.label=V(net)$name,
              vertex.label.color="black",
              vertex.shape="circle",
              vertex.label.cex=V(net)$ind_effect*title_size,
              vertex.size=V(net)$ind_effect*5+point_size,
              edge.curved=0.05,
              edge.color=E(net)$edge.colour,
              edge.frame.color=E(net)$edge.colour,
              edge.width=E(net)$Effect*3,
              vertex.color=V(net)$node.colour,
              vertex.frame.color=NA,  
              layout=l,
              main=title,
              margin=c(-.05,-.05,-.05,-.05))
  
  dev.off()
  
}

network_plot(net_all, title = "Idop  Network",data_name = data_name,title_size = 2
             ,plot_type = "circle",point_size = 3,select_type = "yall",plot_size = 12000
             ,change_name = NULL,change_x = 1,change_y = -1)
