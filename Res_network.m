function [featuresTrain_resnet,featuresTest_resnet]=Res_network(imds,imdsTrain,imdsValidation,options)
net= resnet50;
net.Layers;

a= fullyConnectedLayer(6,'Name','fc1000');
res_net1= replaceLayer(layerGraph(net),'fc1000',a);
a = imageInputLayer([224 224 3],'Name','input_1');
res_net1= replaceLayer((res_net1),'input_1',a);
a = convolution2dLayer(7,64,'Name','conv1','Stride',2, 'Padding' ,[3 3 3 3]);
res_net2= replaceLayer((res_net1),'conv1',a);
b = classificationLayer('Name','ClassificationLayer_fc1000');
res_net= replaceLayer((res_net2),'ClassificationLayer_fc1000',b);
net = trainNetwork(imds,res_net,options);
%% feature selection 

    for i=1: length(imdsTrain.Files)
        featuresTrain_resnet(i,:) = activations(net,imread(imdsTrain.Files{i}),'avg_pool','OutputAs','rows');%

    end

    for i=1: length(imdsValidation.Files)
        featuresTest_resnet(i,:) = activations(net,imread(imdsValidation.Files{i}),'avg_pool','OutputAs','rows');%

    end
% save Fea_resnet featuresTest_resnet featuresTrain_resnet
end
% end