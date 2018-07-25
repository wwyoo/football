#Author: William Weimin Yoo
#Aim: Modeling football trajectories for single player

#Brief description:
#1. Modeling player movement by building state space models based on Newtonian mechanics.
#2. Reproduces Figures 1 to 5 in the paper.

######################################################################################################
#load R libraries to do data processing and run Kalman filter
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

timeinv = seq(min(m1$time), max(m1$time), by=100)  #all time points

#p <- ggplot(xyraw[1:1000,], aes(x, y, color=IdTeam, frame=time)) + 
# geom_point()

#gganimate(p, interval=.1)

#######################################################################################################
#live football 
#data visualization
par(mai = c(0.5, 0.5, 0.1, 0.1))

#how big is the field?
right <- max(m1$x)
left <- min(m1$x)
up <- max(m1$y)
down <- min(m1$y)

for(i in 1:length(timeinv)){
 #jpeg(paste("fig", i, ".jpg", sep = ""))
 xyt <- filter(m1, time == timeinv[i])
 z<- as.factor(xyt[,colnames(xyt)=="IdTeam"])
 plot(xyt[, 3:4], pch=c("o",19,19)[z], col=c("black","blue","red")[z], ylim=c(down,up), xlim=c(left,right)) 
 #ggplot(xyt, aes(x, y, color=IdTeam)) + 
 # geom_point()
 #pch=c("o",1,1)[z]
 #Sys.sleep(0.01)
 #lines(xyt[-1,], type="p", col="4")
 #dev.off()
}

#########################################################################################################
#Kalman Filter
#Construct matrices and state vectors

p1 <- filter(m1, player == 342459) #do for one player first

data <- cbind(p1$x, p1$y)  #xy coordinates

ntrain <- 10  #number training samples
h <- 1 #prediction steps ahead

upperlimit <- 2500 
#nrow(data) - ntrain - h + 1  #max limit of data points used

vpred <- rep(0, upperlimit)

predwhere <- matrix(0, upperlimit, 2) #predicted position

delta <- 100 #diff between timesteps

Zt <- matrix(c(1,0,0,1,0,0,0,0),2,4)  #observation matrix

Ht <- matrix(c(NA,0,0,NA),2,2)  #cov measurement error

Tt <- matrix(c(1,0,0,0,0,1,0,0,delta,0,1,0,0,delta,0,1),4,4) #latent transition matrix

Rt <- matrix(c(delta^2/2, 0, delta, 0, 0, delta^2/2, 0, delta), 4, 2)  #acceleration matrix

Qt <- matrix(NA, 2, 2)  #cov for acceleration

a1 <- matrix(c(0, 0, 0, 0), 4, 1)  #latent state initialization
P1 <- matrix(0, 4, 4)  #diffuse cov latent state initialization
P1inf <- diag(4) 

par(mai = c(0.5, 0.5, 0.1, 0.1))
plot(NA, xlab="", ylab="",  ylim=c(down,up), xlim=c(left,right))
#x11()

set.seed(100)
for(i in 1:upperlimit){
#uncomment here to model all players
#for(j in 1:23){
# p <- filter(m1, player == m1$player[j])
# data <- cbind(p$x, p$y)
 datatrain <- data[i:(i+ntrain-1), ]

#set up state space object
 model_gaussian <- SSModel(datatrain ~ -1 +
  SSMcustom(Z = Zt, T = Tt, R = Rt, Q = Qt, a1 = a1, P1 = P1, P1inf = P1inf), H = Ht)

#estimate parameters by MLE
 fit_gaussian <- fitSSM(model_gaussian, inits = rep(0.1,6), method = "BFGS")
 #out_gaussian <- KFS(fit_gaussian$model)
 pred <- predict(fit_gaussian$model, n.ahead = h, interval = "prediction", level = 0.95)  #predict h-step ahead
 predwhere[i, ] <- c(pred$y1[h, "fit"], pred$y2[h, "fit"])
 #dev.set(which=2)
 points(x=data[(i+ntrain-1+h), 1], y=data[(i+ntrain-1+h), 2], pch = 19, col = "red") #true position in red
 points(x=predwhere[i, 1], y=predwhere[i, 2], pch=19, col="blue")  #predicted position in blue
 rect(xleft=pred$y1[h, "lwr"], ybottom=pred$y2[h, "lwr"], xright=pred$y1[h, "upr"], ytop=pred$y2[h, "upr"], density=30, col="blue", angle=-30, border="transparent") #plot confidence rectangle

#one-step ahead only to predict latent states
 xt <- out_gaussian$a[nrow(out_gaussian$a), ]  #predicted latent states
 vpred[i] <- sqrt(xt[3]^2+xt[4]^2)  #extract the last two velocity components and compute predicted speed
# dev.set(which=3)
 plot(vpred[1:i], type="l", ylab="Predicted Speed", xlab = "Time")  #plot predicted speed vs time

# plot velocity vector field
 arrows(x0 = predwhere[i, 1], y0 = predwhere[i, 2], x1 = predwhere[i, 1] + 1000*xt[3], y1 = predwhere[i, 2] + 1000*xt[4], length=0.05, col="blue")
 segments(x0 = predwhere[i, 1], y0 = predwhere[i, 2], x1 = predwhere[i, 1] + 1000*xt[3], y1 = predwhere[i, 2] + 1000*xt[4], col="blue")
}
#}
