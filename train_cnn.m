function train_cnn(splitsMat, outModel)
% 3.3-2 训练论文同款 CNN（224x224x3，3个卷积模块，Adam 1e-3，10 epochs, bs=32）
% 用法：
%   step3_3_step2_train_cnn('E:\EIT_N\eit_3_3_imds_splits.mat', ...
%                           'E:\EIT_N\cnn_eit_3_3.mat');

if nargin < 1 || isempty(splitsMat)
    splitsMat = 'eit_3_3_imds_splits.mat';
end
if nargin < 2 || isempty(outModel)
    outModel  = 'cnn_eit_3_3.mat';
end
assert(isfile(splitsMat), '未找到数据索引文件：%s', splitsMat);
S = load(splitsMat);  % 含 splits.train/val/test.Files & Labels, classes
splits = S.splits; classes = S.classes;

% —— 统一读图函数（自动转三通道 & 归一化到[0,1]）——
targetSize = [224 224];
readFcn = @(f) toRGB224(imread(f), targetSize);

% —— 构建三个 datastore —— 
imdsTrain = imageDatastore(splits.train.Files, 'Labels', splits.train.Labels, 'ReadFcn', readFcn);
imdsVal   = imageDatastore(splits.val.Files,   'Labels', splits.val.Labels,   'ReadFcn', readFcn);
imdsTest  = imageDatastore(splits.test.Files,  'Labels', splits.test.Labels,  'ReadFcn', readFcn);

% ——（可选）轻量增强：小角度旋转/平移；如需最严格复现可关掉 —— 
aug = imageDataAugmenter( ...
    'RandRotation',   [-5 5], ...
    'RandXTranslation',[-4 4], ...
    'RandYTranslation',[-4 4] ...
);
auTrain = augmentedImageDatastore(targetSize, imdsTrain, 'DataAugmentation', aug, 'ColorPreprocessing','none');
auVal   = augmentedImageDatastore(targetSize, imdsVal,   'ColorPreprocessing','none');
auTest  = augmentedImageDatastore(targetSize, imdsTest,  'ColorPreprocessing','none');

% —— 论文同款 CNN：3个卷积模块，前两层后接 MaxPool，末尾 GlobalAvgPool —— 
lgraph = layerGraph([
    imageInputLayer([224 224 3], 'Name','input','Normalization','none')   % 读函数已做[0,1]归一化

    convolution2dLayer(3, 32, 'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool1')

    convolution2dLayer(3, 64, 'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Stride',2,'Name','pool2')

    convolution2dLayer(3, 128, 'Padding','same','Name','conv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')

    globalAveragePooling2dLayer('Name','gap')
    fullyConnectedLayer(numel(classes),'Name','fc')
    softmaxLayer('Name','sm')
    classificationLayer('Name','cls')
]);

% —— 训练选项：Adam, lr=1e-3, epochs=10, bs=32（与论文一致）—— 
opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', auVal, ...
    'ValidationFrequency', max(1, floor(numel(imdsTrain.Files)/32)), ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% —— 训练 —— 
net = trainNetwork(auTrain, lgraph, opts);

% —— 验证/测试 —— 
[YPred, scores] = classify(net, auTest);
yTrue = imdsTest.Labels;
acc   = mean(YPred == yTrue);
fprintf('CNN 测试集准确率 = %.2f%%\n', acc*100);

figure; confusionchart(yTrue, YPred, 'Title','CNN - 测试集混淆矩阵','FontSize',12);

% —— 保存模型与类别 —— 
save(outModel, 'net', 'classes');
fprintf('✔ 已保存训练模型到：%s\n', outModel);

end

% ====== 小工具：读图→RGB→resize→归一化到[0,1] ======
function I3 = toRGB224(I, tgtSize)
% 灰度/索引图 → RGB
if ndims(I)==2
    I3 = repmat(I,1,1,3);
elseif size(I,3)==1
    I3 = repmat(I,1,1,3);
elseif size(I,3)==3
    I3 = I;
else
    I3 = I(:,:,1:3);
end
% 转 double & 归一化到[0,1]
if ~isfloat(I3), I3 = im2double(I3); end
% 保持纵横比的紧缩填充：使用中心裁剪/letterbox到 224
I3 = imresize(I3, tgtSize);
end
