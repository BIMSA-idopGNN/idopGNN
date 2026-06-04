rm(list = ls())
library(deSolve)
library(reshape2)
library(ggplot2)
library(orthopolynom)
library(ggplot2)
library(cowplot)
library(patchwork)
library(dplyr)
library(ggrepel)
library(parallel)
library(pbapply)
library(glmnet)
set.seed(15)
args <- commandArgs(trailingOnly = TRUE) 

if (length(args) >= 1) {
  arg_name <- args[1]
  
  if (file.exists(arg_name)) {
    load(arg_name)
    data_name <- tools::file_path_sans_ext(basename(arg_name))
  } else if (file.exists(paste0("./data/", arg_name, "_fit_result.RData"))) {
    load(paste0("./data/", arg_name, "_fit_result.RData"))
    data_name <- arg_name
  } else {
    stop("fit_result.RData don't exist！ Please check the Allometric scaling law fit process！")
  }
  
} else {
  load("./data/XD_fit_result.RData")
  data_name <- "XD"
}

cat("Loaded data from:", data_name, "\n")

power_equation1 <- function(x, power_par){power_par[1]*x^power_par[2]}

power_equation <- function(x, power_par){ t(sapply(1:nrow(power_par),
                                                   function(c) power_par[c,1]*x^power_par[c,2] )) }

get_legendre_matrix <- function(x,legendre_order){
  
  legendre_coef <- legendre.polynomials(n = legendre_order, normalized=F)
  legendre_matrix <- as.matrix(as.data.frame(polynomial.values(
    polynomials = legendre_coef, x = scaleX(x, u = -1, v = 1))))
  colnames(legendre_matrix) <- paste0("legendre_",0:legendre_order)
  return(legendre_matrix[,2:(legendre_order+1)])
}

get_legendre_par <- function(y,legendre_order,x){
  legendre_par <- as.numeric(coef(lm(y~get_legendre_matrix(x,legendre_order))))
  return(legendre_par)
}

legendre_fit <- function(par,x){
  legendre_order = length(par)
  fit <- sapply(1:length(par), function(c)
    par[c]*legendre.polynomials(n=legendre_order, normalized=F)[[c]])
  legendre_fit <- as.matrix(as.data.frame(polynomial.values(
    polynomials = fit, x = scaleX(x, u = -1, v = 1))))
  x_interpolation <- rowSums(legendre_fit)
  return(x_interpolation)
}

darken <- function(color, factor=1.2){
  col <- col2rgb(color)
  col <- col/factor
  col <- rgb(t(col), maxColorValue=255)
  col
}

qdODEmod <- function(Time, State, Pars, power_par) {
  nn = length(Pars)
  ind_effect = paste0("alpha","*",names(State)[1])
  dep_effect = sapply(2:nn, function(c) paste0(paste0("beta",c-1),"*",names(State)[c]))
  dep_effect = paste0(dep_effect, collapse = "+")
  all_effect = paste0(ind_effect, "+", dep_effect)
  expr = parse(text = all_effect)
  
  with(as.list(c(State, Pars)), {
    dx = eval(expr)
    dy <- power_par[,1]*power_par[,2]*Time^(power_par[,2]-1)
    dind = alpha*x
    for(i in c(1:(nn-1))){
      tmp = paste0(paste0("beta",i),"*",paste0("y",i))
      expr2 = parse(text = tmp)
      assign(paste0("ddep",i),eval(expr2))
    }
    return(list(c(dx, dy, dind, mget(paste0("ddep",1:(nn-1))))))
  })
}

qdODEmod_lgkt <- function(Time, State, Pars, power_par) {
  nn = length(Pars)
  ind_effect = paste0("alpha", "*", names(State)[1])
  dep_effect = sapply(2:nn, function(c) paste0(paste0("beta", c-1), "*", names(State)[c]))
  dep_effect = paste0(dep_effect, collapse = "+")
  all_effect = paste0(ind_effect, "+", dep_effect)
  expr = parse(text = all_effect)
  
  with(as.list(c(State, Pars)), {
    dx = eval(expr)
    dy <- power_par[, 1] * power_par[, 2] * Time^(power_par[, 2] - 1)
    dind = alpha * x
    ddep = sapply(1:(nn-1), function(i) {
      tmp = paste0(paste0("beta", i), "*", paste0("y", i))
      expr2 = parse(text = tmp)
      eval(expr2)
    })
    return(c(dx, dy, dind, ddep))
  })
}

qdODE_ls <- function(pars, data, Time, power_par,bc,lx){

  n = length(pars)
  power_par = as.matrix(power_par)
  if (n==2) {
    Pars = c(alpha = pars[1], beta1 = pars[2:n])
    power_par = t(power_par)
    State = c(x=data[1,1],y1 = matrix(data[-1,1], nrow = n-1, ncol=1),ind = data[1,1],dep = rep(0,n-1))
  } else{
    Pars = c(alpha = pars[1], beta = pars[2:n])
    State = c(x=data[1,1],y = matrix(data[-1,1], nrow = n-1, ncol=1),ind = data[1,1],dep = rep(0,n-1))
  }

  if(lx ==  "ode"){
    out = as.data.frame(ode(func = qdODEmod, y = State, parms = Pars,
                            times = Time, power_par = power_par))
    X = as.numeric(data[1,])
    fit = as.numeric(out[,2])
    ind = as.numeric(out[,(n+2)])

    sse = sum(crossprod(X-fit),sum((ind[ind<0])^2))
  }
  if(lx == "lgkt"){
    rk4 <- function(ode_func, State, Time, Pars, power_par, dt) {
      k1 <- ode_func(Time, State, Pars, power_par)
      k2 <- ode_func(Time + 0.5 * dt, State + 0.5 * dt * k1, Pars, power_par)
      k3 <- ode_func(Time + 0.5 * dt, State + 0.5 * dt * k2, Pars, power_par)
      k4 <- ode_func(Time + dt, State + dt * k3, Pars, power_par)
      
      new_state <- State + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
      
      return(new_state)
    }
    
    integrate_rk4 <- function(ode_func, State, Time_range, Pars, power_par, dt) {
      times = Time
      states <- matrix(NA, nrow = length(times), ncol = length(State))
      colnames(states) <- names(State)
      states[1, ] <- State
      
      current_time <- Time_range[1]
      current_state <- State
      
      for (i in 2:length(times)) {
        current_time <- times[i-1]
        current_state <- rk4(ode_func, current_state, current_time, Pars, power_par, dt)
        states[i, ] <- current_state
      }
      
      return(cbind(times,states))
    }
    Time_range <- c(min(Time), max(Time))
    result <- integrate_rk4(qdODEmod_lgkt, State, Time_range, Pars, power_par, dt=((max(Time)-min(Time))/length(Time))/bc)
    out = as.data.frame(result)
    X = as.numeric(data[1,])
    ind = as.numeric(out[,(n+2)])
    alpha=5e-3
    beta=1e-3
    sse = sum(crossprod(X-fit),sum((ind[ind<0])^2),alpha*(Pars[1]^2),beta*(sum(Pars[2:n])^2))
  }
  return(sse)
}

qdODE_fit <- function(pars, data, Time, power_par, LOP_order = 6, new_time = NULL, n_expand = 100,bc,ind_par,lx){
  n = length(pars)
  if (n==2) {
    Pars = c(alpha = pars[1], beta1 = pars[2:n])
    power_par = t(power_par)
    State = c(x=data[1,1],y1 = matrix(data[-1,1], nrow = n-1, ncol=1),ind = data[1,1],dep = rep(0,n-1))
  } else{
    Pars = c(alpha = pars[1], beta = pars[2:n])
    State = c(x=data[1,1],y = matrix(data[-1,1], nrow = n-1, ncol=1),ind = data[1,1],dep = rep(0,n-1))
  }
  if(lx ==  "ode"){
    out = as.data.frame(ode(func = qdODEmod, y = State, parms = Pars,
                            times = Time, power_par = power_par))
    out2 = data.frame(x = out[,1], y = as.numeric(data[1,]), y.fit = out[,2],
                      ind = out[,(n+2)], dep = out[,(n+3):(ncol(out))])
  }
  if(lx ==  "lgkt"){
    rk4 <- function(ode_func, State, Time, Pars, power_par, dt) {
      k1 <- ode_func(Time, State, Pars, power_par)
      k2 <- ode_func(Time + 0.5 * dt, State + 0.5 * dt * k1, Pars, power_par)
      k3 <- ode_func(Time + 0.5 * dt, State + 0.5 * dt * k2, Pars, power_par)
      k4 <- ode_func(Time + dt, State + dt * k3, Pars, power_par)
      new_state <- State + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
      
      return(new_state)
    }
    
    integrate_rk4 <- function(ode_func, State, Time_range, Pars, power_par, dt) {
      times <- seq(Time_range[1], Time_range[2], by = dt)
      states <- matrix(NA, nrow = length(times), ncol = length(State))
      colnames(states) <- names(State)
      states[1, ] <- State
      
      current_time <- Time_range[1]
      current_state <- State
      
      for (i in 2:length(times)) {
        current_time <- times[i-1]
        current_state <- rk4(ode_func, current_state, current_time, Pars, power_par, dt)
        states[i, ] <- current_state
      }
      
      return(cbind(times,states))
    }
    Time_range <- c(min(Time), max(Time))
    result <- integrate_rk4(qdODEmod_lgkt, State, Time_range, Pars, power_par, dt=((max(Time)-min(Time))/length(Time))/bc)
    out = as.data.frame(result)
    out2 = data.frame(x = out[,1], y = power_equation1(seq(Time_range[1], Time_range[2], by = ((max(Time)-min(Time))/length(Time)/bc)),ind_par), y.fit = out[,2],
                      ind = out[,(n+2)], dep = out[,(n+3):(ncol(out))])
  }
  colnames(out2)[4:ncol(out2)] = c(rownames(data)[1], rownames(data)[2:n])
  rownames(out2) = NULL
  
  all_LOP_par = sapply(2:ncol(out2),function(c)get_legendre_par(out2[,c], LOP_order, out2$x))
  

  if (is.null(new_time)) {
    time2 = seq(min(Time), max(Time), length = n_expand)
    out3 = apply(all_LOP_par, 2, legendre_fit, x = time2)
    out3 = cbind(time2, out3)
  } else{
    out3 = apply(all_LOP_par, 2, legendre_fit, x = new_time)
    out3 = cbind(new_time, out3)
  }
  colnames(out3) = colnames(out2)
  result = list(fit = out2,
                predict = data.frame(out3),
                LOP_par = all_LOP_par)
  return(result)
}

qdODE_all <- function(result, relationship, i, init_pars = 1, LOP_order = LOP_order, methods = "ls",
                      new_time = NULL, n_expand = 100, maxit = 1e3,bc = bc,lx){

  Time = as.numeric(colnames(result$power_fit))
  variable = c(relationship[[i]]$ind.name, relationship[[i]]$dep.name)
  data = result$power_fit[variable,]
  if (length(variable)<=1) {
    qdODE.est = NA
    result = NA
    return.obj <- append(result, list(ODE.value = NA,
                                      parameters = NA))
  } else{
    power_par = result$power_par[variable,][-1,]
    n = nrow(data)
    pars_int = c(init_pars,relationship[[i]]$coefficient)
    if (methods == "ls") {
      qdODE.est <- optim(pars_int, qdODE_ls, data = data, Time = Time, power_par = power_par,bc=bc,lx = lx,
                         method = "L-BFGS-B",
                         lower = c(rep(-10,(length(pars_int)))),
                         upper = c(rep(10,(length(pars_int)))),
                         control = list(trace = TRUE, maxit = maxit))
      result <- qdODE_fit(pars = qdODE.est$par,
                          data = data,
                          power_par = power_par,
                          Time = Time,
                          bc = bc,
                          ind_par = result$power_par[variable,][1,],
                          lx = lx)
      return.obj <- append(result, list(ODE.value = qdODE.est$value,
                                        parameters = qdODE.est$par))
    } else{
      qdODE.est <- optim(pars_int, qdODE_ls, data = data, Time = Time, power_par = power_par,bc=bc,lx = lx ,
                         method = "L-BFGS-B",
                         lower = c(rep(-10,(length(pars_int)))),
                         upper = c(rep(10,(length(pars_int)))),
                         control = list(trace = TRUE, maxit = maxit))
      
      result <- qdODE_fit(pars = qdODE.est$par,
                          data = data,
                          power_par = power_par,
                          Time = Time,
                          bc = bc,
                          ind_par = result$power_par[variable,][1,],
                          lx = lx)
      return.obj <- append(result, list(ODE.value = qdODE.est$value,
                                        parameters = qdODE.est$par))
    }
  }
  return(return.obj)
}

get_interaction <- function(data, col, str, gamma = 1, scaler = FALSE, reduction = TRUE ){
  if (nrow(data)==2) {
    return_obj = list(ind.name = rownames(data)[col],
                      dep.name = rownames(data)[-col],
                      coefficient = cor(t(data))[1,2])
    
  } else{
    data <- t(data); name <- colnames(data)

    n = ncol(data)-1
    
    if(scaler == T){
      data = scale(data)
    }
    
    y = as.matrix(data[,col])
    x = as.matrix(data[,-col])
    
    if (reduction == TRUE) {
      vec <- abs(apply(x, 2, cor, y))
      if (all(is.na(vec))) {
        return_obj = list(ind.name = name[col],
                          dep.name = NA,
                          coefficient = 0)
      } else{
        x = x[,order(vec, decreasing = T)[1:len]]
      }
    }
    
    if ( all(y==0) |  all(y==1) ) {
      return_obj = list(ind.name = name[col],
                        dep.name = NA,
                        coefficient = 0)
    } else{
      ridge_cv <- try(cv.glmnet(x = x, y = y,alpha = 0))
      if ('try-error' %in% class(ridge_cv)) {
        return_obj = list(ind.name = name[col],
                          dep.name = NA,
                          coefficient = 0)
        
      } else{
        ridge_cv <- cv.glmnet(x = x, y = y, type.measure = "mse", nfolds = 10, alpha = 0)
        best_ridge_coef <- abs(as.numeric(coef(ridge_cv, s = ridge_cv$lambda.1se))[-1])
        weights = 1/((best_ridge_coef)^gamma)

        fit <- cv.glmnet(x = x, y = y, alpha = alpha, family = "gaussian", type.measure = "mse",
                         penalty.factor = weights,
                         nfolds = 10, keep = TRUE, thresh=1e-10, maxit=1e6)
        lasso_coef <- coef(fit, s = fit$lambda.1se)
        return_obj = list(ind.name = name[col],
                          dep.name = lasso_coef@Dimnames[[1]][lasso_coef@i + 1][-1],
                          coefficient = lasso_coef@x[-1])
        if ( length(return_obj$dep.name)==0 ) {
          tmp = cor(x,y)
          return_obj$dep.name = rownames(tmp)[which.max(abs(tmp))]
          return_obj$coefficient = tmp[which.max(abs(tmp))]*1/3
        }
        
      }
      
    }

  }
  return(return_obj)
}

qdODE_parallel <- function(result, thread = 12, maxit = 1e3, bc=bc , LOP_order , lx , str){
  data = result$power_fit
  relationship = lapply(1:nrow(data),function(c)get_interaction(data, c, str))
  cat('Start qdODE test',sep="\n")
  core.number <- thread
  cl <- makeCluster(getOption("cl.cores", core.number))
  clusterEvalQ(cl, {require(orthopolynom)})
  clusterEvalQ(cl, {require(deSolve)})
  clusterExport(cl, c( "qdODE_ls", "qdODE_fit", "qdODE_all","get_legendre_matrix","power_equation1","qdODEmod",
                       "get_legendre_par","legendre_fit","result","relationship","maxit","power_equation","bc","qdODEmod_lgkt"), envir=environment())
  result = pblapply(1:nrow(data),function(c) qdODE_all(result = result,
                                                       relationship = relationship,
                                                       i = c,
                                                       maxit = maxit,
                                                       bc=bc,
                                                       LOP_order = LOP_order,
                                                       lx = lx
  ), cl = cl)
  stopCluster(cl)
  names(result) = rownames(data)
  names(relationship) = rownames(data)
  return_obj <- list(ode_result = result,
                     relationship = relationship)
  return(return_obj)
}

qdODE_result <- qdODE_parallel(result = result_all,
                                 thread = 12,
                                 maxit = 1e3,
                                 bc = 10,
                                 LOP_order = 6,
                                 lx = "ode",
                                 str = data_name)

save(qdODE_result, file = paste0("./data/",data_name,"_qdODE_result.RData"))
