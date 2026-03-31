clc
clear
close all
addpath(genpath('.'))
digitDatasetPath='./Data_out/';
  Database1=dir(digitDatasetPath );
Database1(1:2)=[];
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true);
Train_tar=repmat([1:8],[100,1]);
Train_tar=categorical(Train_tar(:));
imds.Labels=Train_tar;
numTrainFiles = 0.8;
% split the data for training and testing
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
% Train_Fea=[];
options = trainingOptions('sgdm','ExecutionEnvironment','auto', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',20, ...
    'Verbose',false, ...
    'Plots','training-progress');
num_net=2;
switch num_net
    case 1
[Eff_Train,Eff_test]=efficient_net(imds,imdsTrain,imdsValidation,options);
    case 2
[Resnet_Train,Resnet_test]=Res_network(imds,imdsTrain,imdsValidation,options);
end