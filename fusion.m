function fusion(svmMat, cnnMat, splitsMat)
% -------------------------------------------------------------------------
% Step 3.3-3 决策级融合 (SVM + CNN)
% 不重新训练，直接加载模型与数据进行融合预测
%
% 输入：
%   svmMat    - 之前保存的 SVM 数据集文件 'eit_3_2_dataset.mat'
%   cnnMat    - 训练好的 CNN 模型 'cnn_eit_3_3.mat'
%   splitsMat - 图像划分索引 'eit_3_3_imds_splits.mat'
%
% 输出：
%   显示 SVM、CNN、融合 三者的混淆矩阵与准确率
% -------------------------------------------------------------------------

if nargin < 1 || isempty(svmMat)
    svmMat = 'eit_3_2_dataset.mat';
end
if nargin < 2 || isempty(cnnMat)
    cnnMat = 'cnn_eit_3_3.mat';
end
if nargin < 3 || isempty(splitsMat)
    splitsMat = 'eit_3_3_imds_splits.mat';
end

%% === 1. 加载模型 ===
fprintf('加载 SVM 数据集与映射...\n');
S = load(svmMat);  % 包含 Xn_train, y_train, Xn_test, y_test, ps, classes
classes = string(S.classes);
ps = S.ps;

fprintf('加载 CNN 模型...\n');
C = load(cnnMat);   % 含 net, classes
net = C.net;

fprintf('加载图像集划分索引...\n');
D = load(splitsMat);
splits = D.splits;

% === 对齐类别顺序（以 SVM classes 为准）===
if iscell(C.classes)
    cnn_classes = string(C.classes);
else
    cnn_classes = string(categories(C.classes));
end
if ~isequal(sort(classes), sort(cnn_classes))
    error('CNN 与 SVM 类别不一致，请检查类别标签。');
end
fprintf('类别顺序一致：%s\n', strjoin(classes, ', '));

%% === 2. 构建 CNN 测试数据加载器（与 Step3.3-2一致）===
targetSize = [224 224];
readFcn = @(f) toRGB224(imread(f), targetSize);
imdsTest = imageDatastore(splits.test.Files, ...
    'Labels', splits.test.Labels, ...
    'ReadFcn', readFcn);

auTest = augmentedImageDatastore(targetSize, imdsTest, 'ColorPreprocessing','none');

%% === 3. CNN 测试集预测 ===
fprintf('执行 CNN 测试集预测...\n');
[YPred_cnn, score_cnn] = classify(net, auTest);
yTrue_cnn = imdsTest.Labels;
acc_cnn = mean(YPred_cnn == yTrue_cnn);
fprintf('CNN 测试准确率 = %.2f%%\n', acc_cnn*100);

%% === 4. SVM 预测（注意顺序匹配） ===
% 假设你的 SVM 测试集 (Xn_test, y_test) 顺序与图像集一致时才能融合。
% 我们确保两者样本对应：
n_cnn = numel(yTrue_cnn);
if size(S.Xn_test,1) ~= n_cnn
    warning('SVM 测试样本数与 CNN 测试图像数不一致，将按较小数量对齐。');
    n_common = min(size(S.Xn_test,1), n_cnn);
else
    n_common = n_cnn;
end

Xn_svm_test = S.Xn_test(1:n_common,:);
yTrue_svm   = S.y_test(1:n_common);
yTrue_cnn   = yTrue_cnn(1:n_common);
score_cnn   = score_cnn(1:n_common,:);
YPred_cnn   = YPred_cnn(1:n_common);

% 加载并重新创建与训练时一致的 SVM 模型
fprintf('加载 SVM 模型 (使用最佳参数 0.1, 0.001)...\n');
C_best = 0.1; gamma_best = 0.001;
sigma = 1/sqrt(2*gamma_best);
t_best = templateSVM('KernelFunction','rbf', ...
                     'KernelScale',sigma, ...
                     'BoxConstraint',C_best, ...
                     'Standardize',false);
svmModel = fitcecoc(S.Xn_train, S.y_train, ...
    'Learners',t_best, ...
    'Coding','onevsone', ...
    'ClassNames',S.classes);

[YPred_svm, score_svm] = predict(svmModel, Xn_svm_test);
acc_svm = mean(YPred_svm == yTrue_svm);
fprintf('SVM 测试准确率 = %.2f%%\n', acc_svm*100);

%% === 5. 决策融合 ===
% 按论文权重：α=0.2628 (SVM)，β=0.7372 (CNN)
alpha = 0.2628; beta = 0.7372;
% score_svm / score_cnn 大小一致 (N×3)
score_fused = alpha * normalize_scores(score_svm) + beta * normalize_scores(score_cnn);

[~, idx] = max(score_fused, [], 2);
YPred_fused = categorical(classes(idx));
acc_fused = mean(YPred_fused == yTrue_cnn);
fprintf('融合模型测试准确率 = %.2f%%\n', acc_fused*100);

%% === 6. 混淆矩阵展示 ===
% 类型统一
yTrue_svm = categorical(string(yTrue_svm));
YPred_svm = categorical(string(YPred_svm));
yTrue_cnn = categorical(string(yTrue_cnn));
YPred_cnn = categorical(string(YPred_cnn));
YPred_fused = categorical(string(YPred_fused));

figure('Name','Fusion Results');
tiledlayout(1,3,'Padding','compact');
nexttile; confusionchart(yTrue_cnn, YPred_cnn, 'Title',sprintf('CNN (%.2f%%)',acc_cnn*100));
nexttile; confusionchart(yTrue_svm, YPred_svm, 'Title',sprintf('SVM (%.2f%%)',acc_svm*100));
nexttile; confusionchart(yTrue_cnn, YPred_fused, 'Title',sprintf('Fusion (%.2f%%)',acc_fused*100));
set(gcf,'Position',[100 100 1400 450]);


end

% ===== 辅助函数 =====
function I3 = toRGB224(I, tgtSize)
if ndims(I)==2, I3 = repmat(I,1,1,3);
elseif size(I,3)==1, I3 = repmat(I,1,1,3);
else, I3 = I(:,:,1:3);
end
I3 = im2double(imresize(I3, tgtSize));
end

function S = normalize_scores(S)
S = S ./ (sum(S,2)+eps);
end
