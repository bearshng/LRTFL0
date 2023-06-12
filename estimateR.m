function [Rn,kf]=estimateR(x,noise_type,verbose)
[w, Rn] = estNoise(x,noise_type,verbose);
[kf Ek]=hysime(x,w,Rn,verbose);  
end