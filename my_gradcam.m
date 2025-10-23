function my_gradcam(cnnMat, splitsMat, outDir)
% Grad-CAM 可视化（无 imfuse 依赖，R2024b 稳定）
if nargin < 1 || isempty(cnnMat),   cnnMat   = 'cnn_eit_3_3.mat'; end
if nargin < 2 || isempty(splitsMat),splitsMat= 'eit_3_3_imds_splits.mat'; end
if nargin < 3 || isempty(outDir),   outDir   = 'E:\EIT_N\gradcam_results'; end
if ~exist(outDir,'dir'), mkdir(outDir); end

% 1) 载入模型与测试集
S = load(cnnMat);
if isfield(S,'net'), net = S.net;
elseif isfield(S,'trainedNet'), net = S.trainedNet;
else, error('模型文件中未找到 net / trainedNet'); end

D = load(splitsMat); splits = D.splits;
targetSize = [224 224];
readFcn = @(f) toRGB224(imread(f), targetSize);
imdsTest = imageDatastore(splits.test.Files, 'Labels', splits.test.Labels, 'ReadFcn', readFcn);

% 2) 自动找最后一个卷积层名
lastConv = detectLastConvLayer(net);
fprintf('使用卷积层 "%s" 生成 Grad-CAM...\n', lastConv);

% 3) 遍历生成并保存
for i = 1:numel(imdsTest.Files)
    I = readimage(imdsTest,i);
    lblTrue = string(imdsTest.Labels(i));
    [lblPred,~] = classify(net, I); lblPred = string(lblPred);

    % --- Grad-CAM map ---
    map = tryGradCAM(net, I, lblPred, lastConv);   % 单通道，未归一

    % --- 归一 & 伪彩 & 叠加（不使用 imfuse）---
    mapN   = mat2gray(map);                        % [0,1]
    cmap   = jet(256);
    mapRGB = ind2rgb(uint8(mapN*255), cmap);       % HxWx3, double

    % alpha 随响应强度变化（中等透明度）
    alpha  = 0.45 * mapN;                          % HxW
    I3     = im2double(I);
    if size(I3,3)==1, I3 = repmat(I3,1,1,3); end

    % 手工 alpha 混合：Overlay = (1-alpha)*I + alpha*mapRGB
    overlay = (1 - alpha).*I3 + alpha.*mapRGB;
    overlay = min(max(overlay,0),1);

    % 保存
    outName = sprintf('%s_%04d_pred-%s.png', lblTrue, i, lblPred);
    imwrite(overlay, fullfile(outDir, outName));

    % 每 10 张预览
    if mod(i,10)==1
        figure('Name',sprintf('Grad-CAM %d',i));
        subplot(1,3,1); imshow(I3);      title('原图');
        subplot(1,3,2); imagesc(mapN);   axis image off; colormap jet; colorbar; title('热图');
        subplot(1,3,3); imshow(overlay); title(sprintf('%s → %s', lblTrue, lblPred));
        drawnow;
    end
end
fprintf('✔ Grad-CAM 叠加图已导出到：%s\n', outDir);
end

% ---------- 工具函数们 ----------
function I3 = toRGB224(I, tgt)
if ndims(I)==2, I3 = repmat(I,1,1,3);
elseif size(I,3)==1, I3 = repmat(I,1,1,3);
elseif size(I,3)>3, I3 = I(:,:,1:3);
else, I3 = I;
end
I3 = im2double(imresize(I3, tgt));
end

function lastConv = detectLastConvLayer(net)
% 尝试 Series/DAG；若失败用 analyzeNetwork
try
    L = net.Layers;
    names = string({L.Name});
    convIdx = find(contains(lower(names),"conv"));
    lastConv = names(convIdx(end));
catch
    info = analyzeNetwork(net);
    names = string({info.Name});
    convIdx = find(contains(lower(names),"conv"));
    lastConv = names(convIdx(end));
end
end

function map = tryGradCAM(net, I, classLabel, lastConv)
% 兼容不同 gradCAM 签名
try
    map = gradCAM(net, I, classLabel, lastConv);
catch
    try
        map = gradCAM(net, I, classLabel); % 有的版本无需层名
    catch ME
        error('gradCAM 调用失败：%s', ME.message);
    end
end
end
