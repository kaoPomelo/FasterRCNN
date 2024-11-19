%% 下載資料
unzip vehicleDatasetImages.zip
data = load ('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;
vehicleDataset
%% 分割數據集
rng(0)
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.7*height(vehicleDataset));

trainingIdx = 1:idx;
trainingDataTb1 = vehicleDataset(shuffledIndices(trainingIdx),:);

validationIdx = idx + 1: idx + 1 + floor(0.1*length(shuffledIndices));
validationDataTb1 = vehicleDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1:length(shuffledIndices);
testDataTb1 = vehicleDataset(shuffledIndices(testIdx),:);
%% 建立物件專用的Datastore
imdsTrain = imageDatastore (trainingDataTb1{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTb1(:,'vehicle'));

imdsValidation = imageDatastore (validationDataTb1{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTb1(:,'vehicle'));

imdsTest = imageDatastore (testDataTb1{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTb1(:,'vehicle'));

trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

data = read(trainingData);
I = data{1};
bbox = data {2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
%% 建立 Faster R-CNN 網路
numClasses = width (vehicleDataset)-1;
featureExtractionNetwork = resnet18;
inputSize = featureExtractionNetwork.Layers(1).InputSize;
featureLayer = 'res4b_relu';
preprocessedTrainingData = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData, numAnchors)
lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);
analyzeNetwork(lgraph)
%%
augmentedTrainingData = transform (trainingData,@augmentData);
%%
trainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
validationData = transform (validationData,@(data)preprocessData(data,inputSize));
%%
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 2, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir, ...
    'ValidationData',validationData);
%%
doTrainingAndEval = true; % 用自己的，建議要有gpu不然根本跑不動
% doTrainingAndEval = false; % 用人家訓練好的
if doTrainingAndEval
    % 訓練 Faster R-CNN 目標檢測器
    [detector, info] = trainFasterRCNNObjectDetector(trainingData, lgraph, options, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.6, 1]);
    
else
    
    % 下載預訓練模型
    pretrained = load('fasterRCNNResNet50EndToEndVehicleExample.mat');
    detector = pretrained.detector;
    
end

%% 評估網路
testData = transform(testData,@(data)preprocessData(data,inputSize));
if doTrainingAndEval
    detectionResults = detect(detector,testData,'MinibatchSize',4);
else
    % load pretrained detector for example.
    pretrained = load ('fasterRCNNResNet50EndToEndVehicleExample.mat');
    detectionResults = pretrained.detectionResults;
end
[ap, recall,precision] = evaluateDetectionPrecision(detectionResults, testData);

figure
plot(recall, precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %2f',ap))
%% 顯示分數
% 1. 載入預訓練的 Faster R-CNN 模型
% pretrained = load('fasterRCNNResNet50EndToEndVehicleExample.mat');
pretrained = load('FasterRCNN2.mat');
detector = pretrained.detector;

% 2. 讀取要檢測的圖片
I = imread('test_img5.jpg');  % 請更換為你的圖片路徑

% 3. 使用 Faster R-CNN 檢測物體
%[bboxes, scores, labels] = detect(detector, I);
[bboxes, scores] = detect (detector, I);
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

%% 顯示labels
% 1. 載入預訓練的 Faster R-CNN 模型
% pretrained = load('fasterRCNNResNet50EndToEndVehicleExample.mat');
pretrained = load('FasterRCNN2.mat');
detector = pretrained.detector;

% 2. 讀取要檢測的圖片
I = imread('test_img.jpg');  % 請更換為你的圖片路徑

% 3. 使用 Faster R-CNN 檢測物體
[bboxes, scores, labels] = detect(detector, I);
%[bboxes, scores] = detect (detector, I);
I = insertObjectAnnotation(I,'rectangle',bboxes,labels);
figure
imshow(I)