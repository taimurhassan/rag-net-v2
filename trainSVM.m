clc
clear all
close all

load gradingPoints_training.mat

% 0: early glaucoma
% 1: advanced glaucoma

c0 = zeros(1,length(r1));
c1 = ones(1,length(r2));
c = [c0, c1]';

% RNFL
r = [r1;r2];

% GCC
gc = [gc1; gc2];

% GC-IPL
g = [g1; g2];


x = [r, g, gc];

idx = randperm(length(c));

for i = 1:length(idx)
    index = idx(i);
    
    temp = x(index,:);
    x(index,:) = x(i,:);
    x(i,:) = temp;
    
    temp = c(index);
    c(index) = c(i);
    c(i) = temp;
end

svm = fitcsvm(x,c);

CVSVMModel = crossval(svm);
[p,scorePred] = kfoldPredict(CVSVMModel);

accuracy = sum(p == c)/length(c)

classLoss = kfoldLoss(CVSVMModel)

sv = svm.SupportVectors;

save("svm.mat","svm");