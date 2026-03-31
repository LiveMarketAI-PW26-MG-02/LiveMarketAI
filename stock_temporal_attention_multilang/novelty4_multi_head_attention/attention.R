compute_attention <- function(data){
  w <- seq(0.1,1,length.out=length(data))
  w <- w / sum(w)
  return(w)
}

data <- c(100,102,101,105,110)
print("R attention:")
print(compute_attention(data))
