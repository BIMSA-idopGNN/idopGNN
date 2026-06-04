# Clear workspace and load required libraries 
rm(list = ls())
library(ggplot2)
library(reshape2)
library(parallel)
library(mvtnorm)
library(pbapply)
library(parallel)
library(deSolve)
library(orthopolynom)
library(glmnet)
library(ggplot2)
library(reshape2)

# Load preprocessed data
load("./data/clean_data.RData")

# Parse command line arguments for dataset selection
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) {
  arg_name <- args[1]
  if (exists(arg_name, envir = .GlobalEnv)) {
    df <- get(arg_name, envir = .GlobalEnv)
    data_name <- arg_name
  } else if (file.exists(arg_name)) {
    df <- read.csv(arg_name)
    data_name <- tools::file_path_sans_ext(basename(arg_name))
  } else {
    stop("Argument is neither an existing object nor a valid file.")
  }
} else {
  df <- all_data
  data_name <- "all_data"
}


get_nozero_otus_result <- function(otu_data){

  #' @title Process OTU data with no zero counts using power law modeling.
  #' @title  Fits power law models for species with complete data.
  #' @Args:
  #   otu_data: OTU abundance matrix (species x samples)
  #' @Returns:
  #   power_par: Power law parameters for each species
  
  #  Clean up species names
  rownames(otu_data)[which(rownames(otu_data) == "Unassigned")] = "bec_Unassigned"

   # Extract or compute habitat index (x)
  if ("x" %in% rownames(otu_data)) {
    x <- as.numeric(otu_data["x", ])
    otu_data <- otu_data[rownames(otu_data) != "x", ]
  } else if (!exists("x") || all(is.na(x))) {
    x <- log10(colSums(otu_data) + 1)
  } else {
    cat("Using existing x value.\n")
  }

  # Prepare data for modeling
  X <- otu_data
  X = log10(X+1) # Log-transform abundance data

  zero_counts1 <- rowSums(X == 0)
  first_letters <- substr(colnames(X), 1, 1)
  
  # Filter species based on zero counts and dataset type
  if(length(unique(first_letters)) == 1){
    data_n <- X[which(zero_counts1 == 0), ]
  } else {
    data_n <- X[which(zero_counts1 <= 50), ]
  }

  # Define error function for power law fitting
  error_function <- function(x, y, pars){
    y_fit = pars[1]*x^pars[2]
    error = sum((y-y_fit)^2)
    return(error)
  }

  # Power law equation for prediction
  power_equation <- function(x, power_par){ t(sapply(1:nrow(power_par),
                                                     function(c) power_par[c,1]*x^power_par[c,2] ) )}

  # Base power law fitting function                                                    
  power_equation_base <- function(x, y){

    x <- as.numeric(x)
    y <- as.numeric(y)
    
    if (length(y) == 200){
      
      lmFit <- lm( log(y+1)  ~ log(x+1))
      
    }else{
      
      lmFit <- lm( log(y)  ~ log(x))
      
    }
    
    coefs <- coef(lmFit)
    a <- exp(coefs[1])
    b <- coefs[2]

    # Refine parameters using optimization
    pars <- optim(c(a = a, b = b), error_function, x = x, y = y, method = "L-BFGS-B")$par

    # Nonlinear least squares fitting
    model <- try(nls(y~a*x^b,start = list(a = pars[1], b = pars[2]),
                     control = nls.control(maxiter = 10e4,tol = 1e-6, minFactor = 1e-200,warnOnly = TRUE)))
    if( 'try-error' %in% class(model)) {
      result = NULL
    }
    else{
      result = model
    }
    return(result)
  }

  # Robust power law fitting with retry mechanism                                                     
  power_equation_all <- function(x,y, maxit=1e4){
    result <- power_equation_base(x,y)
    iter <- 1

    # Retry fitting until success or max iterations
    while( is.null(result) && iter <= maxit) {
      iter <- iter + 1 
      try(result <- power_equation_base(x,y))
    }
    return(result)
  }

  # Parallel power law fitting for all species                                                   
  power_equation_fit <- function(data_n,x, n=30, trans = log10, thread = 12) {
    trans_data = data_n
    colnames(trans_data) = x

    # Set up parallel processing    
    core.number <- thread
    cl <- makeCluster(getOption("cl.cores", core.number))
    clusterExport(cl, c("power_equation_all", "power_equation_base", "trans_data", "X","error_function"), envir = environment())

    # Fit models in parallel    
    all_model = parLapply(cl = cl, 1:nrow(data_n), function(c) power_equation_all(x, trans_data[c,]))
    stopCluster(cl)
    
    names(all_model) = rownames(data_n)
    no = which(sapply(all_model, length)>=1)
    all_model2 = all_model[no]
    data2 = data_n[no,]
    trans_data2 = trans_data[no,]
    
    new_x = seq(min(x), max(x), length = n)
    power_par = t(vapply(all_model2, coef, FUN.VALUE = numeric(2), USE.NAMES = TRUE))
    power_fit = t(vapply(all_model2, predict, newdata = data.frame(x=new_x),
                         FUN.VALUE = numeric(n), USE.NAMES = TRUE))
    
    colnames(power_fit) = new_x

    # Return comprehensive results                          
    result = list(original_data = data2, trans_data = trans_data2,
                  power_par = power_par, power_fit = power_fit,
                  Time = x)
    return(result)
  }
  r1 = power_equation_fit(data_n,x, thread = 12)
  write.csv(r1$power_par,"./data/params.csv", row.names = T)
  return(r1$power_par)
}

get_zero_otus_result <- function(otu_data){

  #' @title Process OTU data with zero counts using two-component mixture model.
  #' @title Handles species with presence-absence patterns and continuous abundance.
  
  #' @Args:
  #   otu_data: OTU abundance matrix with zero values
  
  #' @Returns:
  #   params1: Model parameters for species with zero counts
  
  rownames(otu_data)[which(rownames(otu_data) == "Unassigned")] = "bec_Unassigned"
  if ("x" %in% rownames(otu_data)) {
    x <- as.numeric(otu_data["x", ])
    otu_data <- otu_data[rownames(otu_data) != "x", ]
  } else if (!exists("x") || all(is.na(x))) {
    x <- log10(colSums(otu_data) + 1)
  } else {
    cat("Using existing x value.\n")
  }
  
  X <- otu_data
  X = log10(X+1)
  
  zero_counts1 <- rowSums(X == 0)
  first_letters <- substr(colnames(X), 1, 1)

  if(length(unique(first_letters)) == 1){
    data_n <- X[which((zero_counts1<=length(colnames(otu_data))-3)&zero_counts1>0), ]
  } else {
    data_n <- X[which((zero_counts1<=length(colnames(otu_data))-3)&zero_counts1>50),]
  }

  colnames(data_n) <- x

  # Initialize parameter storage  
  rownum = length(row.names(data_n))
  params1 <- data.frame(matrix(nrow = rownum, ncol = 5))
  model = list()

  # Fit mixture model for each species  
  for (i in 1:rownum) {
    y=data_n[i,]

    # Split data into presence and absence components    
    y = as.numeric(y)
    y0=y[which(y==0)]
    y1=y[-which(y==0)]
    
    x0=x[which(y==0)]
    x1=x[-which(y==0)]

    # Initialize parameters    
    beta1 = c(0.1,0.1)
    beta2 = c(0.1,0.1)
    sigma = c(0.1)

    # Joint log-likelihood function for mixture model    
    JointLogLik <-  function(par,x0,x1,y0,y1){
      beta1 = par[1:2]
      beta2 = par[3:4]
      sigma = par[5]
      
      p0 = exp(beta1[1]*x0^beta1[2])/(1+exp(beta1[1]*x0^beta1[2]))
      p1 = 1/(1+exp(beta1[1]*x1^beta1[2]))
      
      LogLik1 = sum(log(p0))
      LogLik2 = sum(log(p1)) + 
        sum(dnorm(y1-beta2[1]*x1^beta2[2], mean = 0, sd = abs(sigma), log = T))
      LogLik = LogLik1 + LogLik2
      return(-LogLik)
    }
    par = c(beta1,beta2,sigma)

   # Select optimization method based on dataset characteristics    
    prefix <- unique(substr(colnames(X), 1, 2))
    
    if (all(c("XD", "ZD", "YD", "ED") %in% prefix)) {
      method <- "Nelder-Mead"
      maxit  <- 2e4
    } else {
      method <- ifelse(prefix %in% c("XD", "ZD"), "BFGS", "Nelder-Mead")
      maxit  <- ifelse(prefix == "XD", 2e4, 10e4)
    }
    
    par_hat = optim(par,JointLogLik,x0=as.numeric(x0),x1=as.numeric(x1),y0=as.numeric(y0),y1=as.numeric(y1), 
                    method = method,
                    control = list(maxit = maxit, 
                                   trace = T, 
                                   parscale = rep(1e-2,5),
                                   #factr = 1e-300,
                                   pgtol = 1e-300))
    if ('try-error' %in% class(par_hat) ){
      par_hat = optim(par,JointLogLik,x0=as.numeric(x0),x1=as.numeric(x1),y0=as.numeric(y0),y1=as.numeric(y1), 
                      method = "Nelder-Mead",
                      control = list(maxit = 8e3, 
                                     trace = T, 
                                     parscale = rep(1e-2,5),
                                     #factr = 1e-300,
                                     pgtol = 1e-300))
    }
    if (  par_hat$convergence != 0 ){
      while(par_hat$convergence != 0){
        par = par_hat$par
        par_hat = optim(par,JointLogLik,x0=as.numeric(x0),x1=as.numeric(x1),y0=as.numeric(y0),y1=as.numeric(y1), 
                        method = "Nelder-Mead", 
                        control = list(maxit = 2e3, 
                                       trace = F, 
                                       parscale = rep(1e-2,5),
                                       #factr = 1e-300,
                                       pgtol = 1e-300))
      }
    }
    
    model[[i]] <-  par_hat
    params1[i,] <- par_hat$par
    rownames(params1)[i] <- rownames(data_n)[i]
  }

  # Verify convergence for all models  
  for (i in 1:length(model)) {
    if (model[[i]]$convergence != 0){
      print(i)
      y=data_n[i,]
      y0=y[which(y==0)]
      y1=y[-which(y==0)]
      
      x0=x[which(y==0)]
      x1=x[-which(y==0)]
      while(model[[i]]$convergence != 0){
        par = model[[i]]$par
        par_hat = optim(par,JointLogLik,x0=as.numeric(x0),x1=as.numeric(x1),y0=as.numeric(y0),y1=as.numeric(y1), 
                        method = "Nelder-Mead", 
                        control = list(maxit = 2e3, 
                                       trace = F, 
                                       parscale = rep(1e-2,5),
                                       #factr = 1e-300,
                                       pgtol = 1e-300))
        model[[i]] = par_hat
      }
      model[[i]] = par_hat
      params1[i,] = par_hat$par
    }
    
  }
  write.csv(params1,"./data/params1.csv", row.names = T)
  return(params1)
}

get_all_otus_result <- function(otu_data,result1,result2){

  #' @title Combine results from zero and non-zero OTU analyses.
  #' @title Creates comprehensive power law model predictions for all species.
  
  #' @Args:
  #   otu_data: Original OTU abundance matrix
  #   result1: Parameters from non-zero OTU analysis
  #   result2: Parameters from zero OTU analysis
  
  #' @Returns:
  #   fit_result: Comprehensive modeling results for all species

  rownames(otu_data)[which(rownames(otu_data) == "Unassigned")] = "bec_Unassigned"
  
  if ("x" %in% rownames(otu_data)) {
    x <- as.numeric(otu_data["x", ])
    otu_data <- otu_data[rownames(otu_data) != "x", ]
  } else if (!exists("x") || all(is.na(x))) {
    x <- log10(colSums(otu_data) + 1)
  } else {
    cat("Using existing x value.\n")
  }
  X <- otu_data
  
  zero_counts1 <- rowSums(X == 0)
  data_n <- X[which(zero_counts1==0), ]
  data1 <- otu_data
  
  prefix <- unique(substr(colnames(X), 1, 2))  
  
  if (length(prefix) != 4){
    
    params = read.csv("./data/params.csv", row.names = 1)
    params1 = read.csv("./data/params1.csv", row.names = 1)
    
  }else{

    params = result1
    params1 = result2
  }

  
  first_letters <- substr(colnames(X), 1, 1)

  if(length(unique(first_letters)) == 1){
    data1 <- data1[c(which(zero_counts1<=0),which((zero_counts1<=length(colnames(otu_data))-3)&zero_counts1>0)),]
    X <- X[c(which(zero_counts1<=0),which((zero_counts1<=length(colnames(otu_data))-3)&zero_counts1>0)),]
  } else {
    data1 <- data1[c(which(zero_counts1<=50),which((zero_counts1<=length(colnames(otu_data))-3)&zero_counts1>50)),]
    X <- X[c(which(zero_counts1<=50),which((zero_counts1<=length(colnames(otu_data))-3)&zero_counts1>50)),]
  }
  
  # Transform abundance data
  data1 <- log10(data1+1)
  
  power_equation <- function(x, power_par){t(sapply(1:nrow(power_par),
                                                     function(c) power_par[c,1]*x^power_par[c,2] ) )}

  # Generate predictions for all species                                                    
  data_fit <- data.frame(matrix(nrow = length(row.names(X)), ncol = length(colnames(otu_data))))
  data_fit[1:nrow(params),] <- power_equation(x,params)
  data_fit[(nrow(params)+1):(nrow(params)+nrow(params1)),] <- power_equation(x,params1[,3:4])

  # Combine all parameters                                                    
  power_par <- data.frame(matrix(nrow = nrow(X), ncol = 2))
  power_par[1:nrow(params),] <- params
  power_par[(nrow(params)+1):(nrow(params)+nrow(params1)),] <- params1[,3:4]

  # Format results                                                    
  colnames(power_par) <- c("a","b")
  colnames(data1) <- as.numeric(x)
  colnames(data_fit) <- as.numeric(x)
  data_fit <- apply(data_fit, 2, as.numeric)
  rownames(data_fit) <- rownames(data1)
  power_par <- apply(power_par, 2, as.numeric)
  rownames(power_par) <- rownames(data1)

  # Return comprehensive results                                                    
  fit_result = list(original_data = X, trans_data = data1,
                    power_par = power_par, power_fit = data_fit,
                    Time =x)
}

run_fit_pipeline <- function(otu_data_bec) {

  # Execute complete power law modeling pipeline.
  
  # Args:
  #   otu_data_bec: OTU abundance matrix
  
  # Returns:
  #   result_all: Comprehensive modeling results
  
  result1 <- get_nozero_otus_result(otu_data_bec)
  result2 <- get_zero_otus_result(otu_data_bec)
  result_all <- get_all_otus_result(otu_data_bec,result1,result2)
  return(result_all)
}

result_all = run_fit_pipeline(df)

power_equation_plot <- function(result, label = 10,title){

  #' @title Create visualization of power law model fits.
  
  #' @Args:
  #   result: Modeling results from get_all_otus_result
  #   label: Base for axis label transformation
  #   title: Plot title
  
  #' @Returns:
  #   p: ggplot object with model visualization
  
  data1 = result[[2]]
  data2 = result[[4]]
  
  df_original =  reshape2::melt(as.matrix(data1))
  df_fit = reshape2::melt(as.matrix(data2))
  
  label = 10
  p <- ggplot() +
    geom_point(df_original, mapping = aes(x = Var2, y = value),colour = "#015493",
               show.legend = F, alpha = 0.5, shape = 1) +
    geom_line(df_fit, mapping = aes(x = Var2, y = value),colour = "#015493", linewidth = 1.25, show.legend = F)  +
    facet_wrap(~Var1) +
    xlab("Habitat Index") + ylab("Niche Index") + theme(axis.title=element_text(size=18)) +
    theme_bw() +
    geom_text(data = df_fit, aes(label = Var1, x = ((min(Var2)+max(Var2))/2), y = max(df_original$value) * 0.9), show.legend = FALSE, check_overlap = TRUE, size = 4)+
    theme_bw() +
    theme(axis.title=element_text(size=15),
          axis.text.x = element_text(size=10),
          axis.text.y = element_text(size=10,hjust = 0),
          panel.spacing = unit(0.0, "lines"),
          plot.margin = unit(c(1,1,1,1), "lines"),
          strip.background = element_blank(),
          plot.background = element_blank(),
          strip.text = element_blank(),
          plot.title = element_text(hjust = 0.5))+
    ggtitle(title)
  
  
  if (is.null(label)) {
    p = p
  } else {
    xlabel = ggplot_build(p)$layout$panel_params[[1]]$x.sec$breaks
    ylabel = ggplot_build(p)$layout$panel_params[[1]]$y.sec$breaks
    xlabel2 = parse(text= paste(label,"^", xlabel, sep="") )
    if (is.na(ylabel[1]) | ylabel[1] == 0) {
      ylabel2 = parse(text=c(0,paste(label,"^", ylabel[2:length(ylabel)], sep="")))
    } else{
      ylabel2 = parse(text= paste(label,"^", ylabel, sep="") )
    }
    p = p + scale_x_continuous(labels = xlabel2) + scale_y_continuous(labels = ylabel2)
  }
  return(p)
}
                                                    
# Create and save visualization
plot_name = paste0(data_name, " Power Law Model Fitting")
p <- power_equation_plot(result = result_all, label = 10,title = plot_name)
                                                    
# Save results and plot
file_name <- paste0("./data/", data_name, "_fit_result.RData")
save(result_all, file = file_name)
ggsave(p, filename = paste0("./data/", data_name, "_power_law_fit_plot.png"),
       width = 12, height = 8)
