function prepare_cnn_dataset(rootDir, outMat)
% 准备 3.3 图像数据集（S1/S2/S3）：
% 1) 从文件夹载入图像
% 2) 移除损坏/不可读文件
% 3) 按类别分层划分 Train/Val/Test
% 4) 预览样本并保存索引，供后续训练使用
%
% 用法：
%   step3_3_step1_prepare_dataset('E:\EIT_N\reconstruction1', ...
%                                 'eit_3_3_imds_splits.mat');

if nargin < 1 || isempty(rootDir)
    rootDir = 'E:\EIT_N\reconstruction1';
end
if nargin < 2 || isempty(outMat)
    outMat = 'eit_3_3_imds_splits.mat';
end

assert(isfolder(fullfile(rootDir,'S1')) && ...
       isfolder(fullfile(rootDir,'S2')) && ...
       isfolder(fullfile(rootDir,'S3')), ...
       '根目录下需要包含 S1 / S2 / S3 三个子文件夹');

% 允许的图片扩展名
exts = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'};

% 1) 载入（按文件夹自动打标签）
imds = imageDatastore( ...
    {fullfile(rootDir,'S1'), fullfile(rootDir,'S2'), fullfile(rootDir,'S3')}, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', exts);

% 基本统计
tbl = countEachLabel(imds);
disp('各类样本数：'); disp(tbl);

% 2) 移除不可读/损坏图片（稳妥扫描）
badIdx = false(numel(imds.Files),1);
for i = 1:numel(imds.Files)
    f = imds.Files{i};
    try
        I = imread(f); %#ok<NASGU>
    catch
        warning('移除不可读文件：%s', f);
        badIdx(i) = true;
    end
end
if any(badIdx)
    imds.Files(badIdx) = [];
    imds.Labels(badIdx) = [];
end
fprintf('清洗后总样本数：%d\n', numel(imds.Files));
tbl2 = countEachLabel(imds);
disp('清洗后各类样本数：'); disp(tbl2);

% 3) 分层划分（建议：70% 训练，15% 验证，15% 测试）
rng(2025); % 可复现
[imdsTrain, imdsRest] = splitEachLabel(imds, 0.70, 'randomized');
[imdsVal,   imdsTest] = splitEachLabel(imdsRest, 0.50, 'randomized'); % 0.5 of rest → 15%/15%

fprintf('Train/Val/Test = %d / %d / %d\n', numel(imdsTrain.Files), numel(imdsVal.Files), numel(imdsTest.Files));
disp('训练集分布：'); disp(countEachLabel(imdsTrain));
disp('验证集分布：'); disp(countEachLabel(imdsVal));
disp('测试集分布：'); disp(countEachLabel(imdsTest));

% 4) 预览：每类各取若干张做一页拼图
figure('Name','Dataset preview');
subplot(1,3,1); preview_montage(imdsTrain, 'S1'); title('Train S1 Sample');
subplot(1,3,2); preview_montage(imdsTrain, 'S2'); title('Train S2 Sample');
subplot(1,3,3); preview_montage(imdsTrain, 'S3'); title('Train S3 Sample');

% 5) 保存拆分结果（存储文件路径与标签；训练时再构建增强/尺寸）
splits.train.Files = imdsTrain.Files;  splits.train.Labels = imdsTrain.Labels;
splits.val.Files   = imdsVal.Files;    splits.val.Labels   = imdsVal.Labels;
splits.test.Files  = imdsTest.Files;   splits.test.Labels  = imdsTest.Labels;
classes = categories(imds.Labels);

save(outMat, 'splits', 'classes', 'rootDir', '-v7.3');
fprintf('✔ 已保存拆分索引到：%s\n', outMat);

end

% ====== 小工具：按类名从datastore取样并montage ======
function preview_montage(imds, className)
idx = find(imds.Labels == className);
idx = idx(randperm(numel(idx), min(16, numel(idx))));
t = imageDatastore(imds.Files(idx));
try
    montage(t, 'Size', [4 4]);
catch
    % 部分图像尺寸不一致也能montage，若异常则逐张显示
    tile = min(9, numel(idx));
    for k=1:tile
        subplot(3,3,k);
        imshow(imread(t.Files{k}),[]);
        axis off
    end
end
end
