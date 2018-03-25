#Author: William Weimin Yoo
#Project: Modeling football trajectories for 22 players plus ball

######################################################################################################
library(dplyr)
library(KFAS)

#data preprocessing
#combine player's information with movement data
m1raw <- read.csv("match1.csv") #read position data
playerid <- read.csv("players_82468_0.csv")  #read player's info
playerid <- rbind(playerid, c(-1, -1, 0 , "True", NA, as.factor("Ball"), "Ball", 82468))
colnames(playerid)[1] <- ""
colnames(playerid)[2] <- "player"
m1rawm <- merge(m1raw, playerid, by = "player") #combine 
m1 <- m1rawm[order(m1rawm$time),] #position data with player's info

#how big is the field?
right <- max(m1$x)
left <- min(m1$x)
up <- max(m1$y)
down <- min(m1$y)

#combine positional data for ball and all players
data <- c()
for(i in 1:23){
  p <- filter(m1, player == m1$player[i])
  data<- cbind(data, p$x[1:34134], p$y[1:34134]) 
  #up until 34134 timesteps because some player will be substituted after that
  #so same players playing from timestep 1 to 34134
}

ntrain <- 5  #number training samples
h <- 1  #prediction steps ahead

upperlimit <- 1000
#nrow(data) - ntrain - h + 1  #max limit of data points used

predwhere <- matrix(0, upperlimit, 46) #predicted position

delta <- 100 #diff between timesteps

#########################################################################################################
#Kalman Filter
#Construct matrices and state vectors

#observation matrix
Zt <- matrix(rep(c(1, rep(0, 92), 1, rep(0, 94) ), 46), nrow=46, ncol=92, byrow=TRUE) 

Ht <- matrix(0, 46, 46)

diag(Ht) <- NA  #diagonal cov measurement error

#latent transition matrix
Tt <- matrix(rep(c(1, 0, delta, rep(0, 90), 1, 0, delta, rep(0, 90), 1, rep(0, 92), 1, rep(0, 92)), 23), nrow=92, ncol=92, byrow=TRUE)

#acceleration matrix
Rt <- matrix(rep(c(delta^2/2, rep(0, 46), delta^2/2, rep(0, 44), delta, rep(0, 46), delta, rep(0, 46)), 23), nrow=93, ncol=46, byrow=TRUE)

Rt <- Rt[-93,]

Qt <- matrix(0, 46, 46)

diag(Qt) <- NA  #diagonal cov for acceleration

a1 <- matrix(rep(0, 92), 92, 1)  #latent state initialization

P1 <- matrix(0, 92, 92) #diffuse cov latent state initialization

P1inf <- diag(92)

par(mai = c(0.5, 0.5, 0.1, 0.1))
plot(NA, xlab="", ylab="",  ylim=c(down,up), xlim=c(left,right))

set.seed(100)
for(i in 1:upperlimit){
 datatrain <- data[i:(i+ntrain-1), ]

#set up state space object
 model_gaussian <- SSModel(datatrain ~ -1 +
  SSMcustom(Z = Zt, T = Tt, R = Rt, Q = Qt, a1 = a1, P1 = P1, P1inf = P1inf), H = Ht)

#estimate parameters by MLE
 fit_gaussian <- fitSSM(model_gaussian, inits = rep(0.1, 92), method = "BFGS")

#predict h-step ahead
 pred <- predict(fit_gaussian$model, n.ahead = h, interval = "prediction", level = 0.95)  #predict h-step ahead
  predwhere[i, ] <- c(pred$y1[h, "fit"], pred$y3[h, "fit"],pred$y5[h, "fit"],pred$y7[h, "fit"],pred$y9[h, "fit"],
  pred$y11[h, "fit"],pred$y13[h, "fit"],pred$y15[h, "fit"],pred$y17[h, "fit"],pred$y19[h, "fit"],pred$y21[h, "fit"],
  pred$y23[h, "fit"],pred$y25[h, "fit"],pred$y27[h, "fit"],pred$y29[h, "fit"],pred$y31[h, "fit"],pred$y33[h, "fit"],
  pred$y35[h, "fit"],pred$y37[h, "fit"],pred$y39[h, "fit"],pred$y41[h, "fit"],pred$y43[h, "fit"],pred$y45[h, "fit"],
  pred$y2[h, "fit"], pred$y4[h, "fit"],pred$y6[h, "fit"],pred$y8[h, "fit"],pred$y10[h, "fit"],
  pred$y12[h, "fit"],pred$y14[h, "fit"],pred$y16[h, "fit"],pred$y18[h, "fit"],pred$y20[h, "fit"],pred$y22[h, "fit"],
  pred$y24[h, "fit"],pred$y26[h, "fit"],pred$y28[h, "fit"],pred$y30[h, "fit"],pred$y32[h, "fit"],pred$y34[h, "fit"],
  pred$y36[h, "fit"],pred$y38[h, "fit"],pred$y40[h, "fit"],pred$y42[h, "fit"],pred$y44[h, "fit"],pred$y46[h, "fit"])

#true position in red 
 points(x=data[(i+ntrain-1+h), c(1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45)], 
  y=data[(i+ntrain-1+h), c(2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46)], pch = 19, col = "red") #true position in red

#predicted position in blue 
 points(x=predwhere[i, 1:23], y=predwhere[i, -(1:23)], pch=19, col="blue")  #predicted position in blue
 print(i) #monitor progress
}
