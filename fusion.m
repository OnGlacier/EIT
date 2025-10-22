function fusion(cnnModelMat, svmDatasetMat, outFusionMat)
% 3.3-3 复现论文动态融合（SVM 电压 + CNN 图像）
% 需要：
%   cnnModelMat    – 已训练 CNN 模型 (step3_3_step2_train_cnn 输出)
%   svmDatasetMat  – 电压特征数据集 (step3_2_prepare_dataset 输出)
% 输出：
%   outFusionMat   – 保存融合结果结构体

if nargin<1, cnnModelMat = 'cnn_eit_3_3.mat'; end
if nargin<2, svmDatasetMat = 'eit_3_2_dataset.mat'; end
if nargin<3, outFusionMat = 'fusion_result.mat'; end

%% === 1. 载入 CNN 模型与 SVM 数据 ===
load(cnnModelMat, 'net', 'classes');   % CNN 网络
load(svmDatasetMat, 'Xn_train','y_train','Xn_test','y_test','ps'); % 电压数据

%% === 2. 加载 CNN 的测试图像集 ===
splitsMat = replace(cnnModelMat,'cnn_eit_3_3.mat','eit_3_3_imds_splits.mat');
S = load(splitsMat);
imdsTest = imageDatastore(S.splits.test.Files, ...
                          'Labels', S.splits.test.Labels, ...
                          'ReadFcn', @(f)toRGB224(imread(f),[224 224]));

%% === 3. 获取两种模型的概率输出 ===
% CNN 概率
[YPred_cnn, scores_cnn] = classify(net, imdsTest);
probs_cnn = scores_cnn;                % [N×3]
true_cnn  = imdsTest.Labels;

% SVM 概率：重新拟合 SVM 概率模型
template = templateSVM('KernelFunction','rbf','KernelScale','auto','BoxConstraint',1);
svmMdl = fitcecoc(Xn_train, y_train, 'Learners',template,'Coding','onevsone');
[YPred_svm,~,~,posterior_svm] = predict(svmMdl, Xn_test);
probs_svm = posterior_svm;             % [N×3]
true_svm  = y_test;

% 对齐类别顺序
[~,idxCnn] = ismember(string(classes), string(categories(y_train)));
probs_svm = probs_svm(:,idxCnn);
probs_cnn = probs_cnn(:,idxCnn);

%% === 4. 动态融合 ===
alpha = 0.2628; beta = 0.7372;  % 论文给定权重
probs_fuse = alpha*probs_svm + beta*probs_cnn;

[~,idxMax] = max(probs_fuse,[],2);
YPred_fuse = categorical(classes(idxMax), classes);
yTrue = true_cnn;  % 与 CNN 的测试集对应

acc_svm  = mean(YPred_svm == true_svm);
acc_cnn  = mean(YPred_cnn == true_cnn);
acc_fuse = mean(YPred_fuse == yTrue);

fprintf('SVM  测试准确率：%.2f%%\n', acc_svm*100);
fprintf('CNN  测试准确率：%.2f%%\n', acc_cnn*100);
fprintf('融合 测试准确率：%.2f%%\n', acc_fuse*100);

figure;
confusionchart(yTrue, YPred_fuse, ...
    'Title',sprintf('动态融合 混淆矩阵  α=%.4f β=%.4f',alpha,beta), ...
    'FontSize',12);

%% === 5. 保存结果 ===
save(outFusionMat, 'alpha','beta','probs_svm','probs_cnn','probs_fuse', ...
                   'YPred_fuse','yTrue','acc_svm','acc_cnn','acc_fuse');
fprintf('✔ 已保存融合结果到：%s\n', outFusionMat);
end

% ====== 辅助函数 ======
function I3 = toRGB224(I, tgtSize)
if ndims(I)==2, I3=repmat(I,1,1,3);
elseif size(I,3)==1, I3=repmat(I,1,1,3);
else, I3=I(:,:,1:3); end
if ~isfloat(I3), I3=im2double(I3); end
I3 = imresize(I3, tgtSize);
end
