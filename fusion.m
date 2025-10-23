function my_gradcam(cnnMat, splitsMat, outDir)
% -------------------------------------------------------------------------
% Step 3.3-4 生成 Grad-CAM 热图（自动检测兼容网络层名）
% 兼容 MATLAB R2024b + 自定义 CNN（3卷积结构）
% -------------------------------------------------------------------------
if nargin < 1 || isempty(cnnMat)
    cnnMat = 'cnn_eit_3_3.mat';
end
if nargin < 2 || isempty(splitsMat)
    splitsMat = 'eit_3_3_imds_splits.mat';
end
if nargin < 3 || isempty(outDir)
    outDir = 'E:\EIT_N\gradcam_results';
end
if ~exist(outDir,'dir'), mkdir(outDir); end

fprintf('加载 CNN 模型与图像划分...\n');
S = load(cnnMat);
if isfield(S,'net')
    net = S.net;
elseif isfield(S,'trainedNet')
    net = S.trainedNet;
else
    error('模型文件中未找到 net 或 trainedNet 变量。');
end

D = load(splitsMat);
splits = D.splits;

targetSize = [224 224];
readFcn = @(f) toRGB224(imread(f), targetSize);
imdsTest = imageDatastore(splits.test.Files, ...
    'Labels', splits.test.Labels, ...
    'ReadFcn', readFcn);

% === 自动检测最后的卷积层 ===
try
    layerNames = arrayfun(@(L) string(L.Name), net.Layers);
    convLayers = layerNames(contains(layerNames,"conv",'IgnoreCase',true));
    lastConv = convLayers(end);
catch
    % 若 net.Layers 不存在 (dlnetwork)
    layerInfo = analyzeNetwork(net);
    names = string({layerInfo.Name});
    convIdx = find(contains(names,"conv",'IgnoreCase',true));
    lastConv = names(convIdx(end));
end

fprintf('使用卷积层 "%s" 生成 Grad-CAM 热图...\n', lastConv);

% === 遍历测试图像 ===
for i = 1:numel(imdsTest.Files)
    I = readimage(imdsTest,i);
    lblTrue = string(imdsTest.Labels(i));
    [lblPred,~] = classify(net,I);
    lblPred = string(lblPred);

    try
        map = gradCAM(net, I, lblPred, lastConv);
    catch
        % 某些版本不接受 layer 参数 → 自动降级调用
        try
            map = gradCAM(net, I, lblPred);
        catch
            warning('Grad-CAM 生成失败 #%d (%s)', i, imdsTest.Files{i});
            continue;
        end
    end

    % === 可视化叠加 ===
    % === 可视化叠加 ===
% === 可视化叠加（新版安全写法） ===
mapRGB = ind2rgb(uint8(255 * mat2gray(map)), jet(256));  % 热图转RGB
try
    overlayI = imfuse(I, mapRGB, 'blend');               % 普通叠加
catch
    % 如果 blend 不支持，使用简单加权叠加
    overlayI = 0.6*im2double(I) + 0.4*im2double(mapRGB);
end

outName = sprintf('%s_%04d_pred-%s.png', lblTrue, i, lblPred);
imwrite(im2double(overlayI), fullfile(outDir, outName));


    if mod(i,10)==1
        figure('Name',sprintf('Grad-CAM 样例 %d',i));
        subplot(1,3,1); imshow(I); title('原图');
        subplot(1,3,2); imshow(map,[]); colormap jet; title('热图');
        subplot(1,3,3); imshow(overlayI); title(sprintf('%s → %s',lblTrue,lblPred));
        drawnow;
    end
end

fprintf('✔ Grad-CAM 热图已导出到：%s\n', outDir);
end

% === 工具函数 ===
function I3 = toRGB224(I, tgtSize)
if ndims(I)==2
    I3 = repmat(I,1,1,3);
elseif size(I,3)==1
    I3 = repmat(I,1,1,3);
elseif size(I,3)>3
    I3 = I(:,:,1:3);
else
    I3 = I;
end
I3 = im2double(imresize(I3, tgtSize));
end
