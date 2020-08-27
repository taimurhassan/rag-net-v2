clc
clear all
close all

load('C:\tbme\results\quantitative\rocFundus.mat')

[X,Y,T,AUC] = perfcurve(labels_origa,scores_origa,1);

plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
grid on