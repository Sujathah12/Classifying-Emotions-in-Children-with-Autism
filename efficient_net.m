function [featuresTrain_eff,featuresTest_eff]=efficient_net(imds,imdsTrain,imdsValidation,options)
net= efficientnetb0;
net.Layers;

a= fullyConnectedLayer(6,'Name','efficientnet-b0|model|head|dense|MatMul');
res_net1= replaceLayer(layerGraph(net),'efficientnet-b0|model|head|dense|MatMul',a);
a = imageInputLayer([224 224 3],'Name','ImageInput');
res_net1= replaceLayer((res_net1),'ImageInput',a);
a = convolution2dLayer(3,32,'Name','efficientnet-b0|model|stem|conv2d|Conv2D','Stride',2, 'Padding' ,[0 1 0 1]);
res_net2= replaceLayer((res_net1),'efficientnet-b0|model|stem|conv2d|Conv2D',a);
b = classificationLayer('Name','classification');
res_net= replaceLayer((res_net2),'classification',b);
net = trainNetwork(imds,res_net,options);
%% feature selection 

    for i=1: length(imdsTrain.Files)
        featuresTrain_eff(i,:) = activations(net,imread(imdsTrain.Files{i}),'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool','OutputAs','rows');%

    end

    for i=1: length(imdsValidation.Files)
        featuresTest_eff(i,:) = activations(net,imread(imdsValidation.Files{i}),'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool','OutputAs','rows');%

    end
% save Fea_eff featuresTest_eff featuresTrain_eff
end
% end