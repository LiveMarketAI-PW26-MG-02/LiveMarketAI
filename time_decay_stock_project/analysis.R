
exponential_decay <- function(data, alpha=0.5){
  return(exp(-alpha * seq_along(data)))
}
