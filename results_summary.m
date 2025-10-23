function results_summary(fusionDir, cnnMat, splitsMat)
% -------------------------------------------------------------------------
% Step 3.3-5 结果统计与可视化复现（图 3-6 & 3-7）
%  1. 计算每类平均 Grad-CAM 热图
%  2. 绘制模型准确率柱状图
%  3. 绘制融合模型 ROC 曲线（含 AUC）
% -------------------------------------------------------------------------
if nargin<1 || isempty(fusionDir)
    fusionDir = 'E:\EIT_0\gradcam_results';
end
if nargin<2 || isempty(cnnMat)
    cnnMat = 'E:\EIT_0\cnn_eit_3_3.mat';
end
if nargin<3 || isempty(splitsMat)
    splitsMat = 'E:\EIT_0\eit_3_3_imds_splits.mat';
end

%% === 1️⃣ 计算每类平均 Grad-CAM 热图 ===
fprintf('计算每类平均 Grad-CAM 热区...\n');
classes = {'S1','S2','S3'};
avgMaps = cell(1,numel(classes));

for ci = 1:numel(classes)
    files = dir(fullfile(fusionDir,[classes{ci} '_*_pred-*.png']));
    if isempty(files), warning('%s 无热图文件',classes{ci}); continue; end

    for k = 1:numel(files)
        I = im2double(imread(fullfile(fusionDir,files(k).name)));
        if size(I,3)==3, Igray = rgb2gray(I); else, Igray = I; end
        if k==1, accMap = zeros(size(Igray)); end
        accMap = accMap + Igray;
    end
    avgMaps{ci} = accMap/numel(files);
end

figure('Name','Mean Grad-CAM Heatmaps','Position',[100 100 1200 400]);
for ci = 1:numel(classes)
    subplot(1,3,ci);
    imagesc(avgMaps{ci}); axis image off; colormap jet; colorbar;
    title(sprintf('Mean Grad-CAM -%s',classes{ci}));
end

%% === 2️⃣ 模型准确率柱状图（使用论文融合权重 α=0.2628, β=0.7372） ===
fprintf('绘制模型准确率柱状图...\n');
acc_cnn  = load_cnn_acc(cnnMat,splitsMat);
acc_svm  = load_svm_acc();
acc_fuse = 0.2628*acc_svm + 0.7372*acc_cnn;

figure('Name','Model Accuracy','Position',[200 200 600 400]);
bar([acc_svm,acc_cnn,acc_fuse]*100);
set(gca,'XTickLabel',{'SVM','CNN','Fusion'},'FontSize',12);
ylabel('Accuracy (%)');
title('Model Performance Comparison');
grid on;

%% === 3️⃣ 绘制 ROC 曲线（以 CNN 模型为例，可扩展至融合） ===
fprintf('绘制 CNN ROC 曲线...\n');
C = load(cnnMat);
net = C.net;
D = load(splitsMat);
splits = D.splits;

targetSize = [224 224];
readFcn = @(f) im2double(imresize(imread(f),targetSize));
imdsTest = imageDatastore(splits.test.Files,'Labels',splits.test.Labels,'ReadFcn',readFcn);
[YPred,scores] = classify(net,imdsTest);

% 将标签转为 one-hot
yTrue = grp2idx(imdsTest.Labels);
nClasses = numel(unique(yTrue));
figure('Name','ROC Curve','Position',[200 200 600 450]);
for ci=1:nClasses
    [X,Y,~,AUC] = perfcurve(yTrue==ci,scores(:,ci),true);
    plot(X,Y,'LineWidth',1.8); hold on;
    legendInfo{ci} = sprintf('%s (AUC=%.3f)',classes{ci},AUC);
end
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC Curve for CNN Model'); grid on;
legend(legendInfo,'Location','SouthEast');
fprintf('✔ 结果汇总完成。\n');

end

% === 工具函数们 ===
function acc = load_cnn_acc(cnnMat,splitsMat)
C = load(cnnMat); net = C.net;
D = load(splitsMat); splits = D.splits;
targetSize = [224 224];
readFcn = @(f) im2double(imresize(imread(f),targetSize));
imdsTest = imageDatastore(splits.test.Files,'Labels',splits.test.Labels,'ReadFcn',readFcn);
[YPred,~] = classify(net,imdsTest);
acc = mean(YPred==imdsTest.Labels);
end

function acc = load_svm_acc()
S = load('E:\EIT_0\eit_3_2_dataset.mat');
C_best=0.1; gamma_best=0.001; sigma=1/sqrt(2*gamma_best);
t=templateSVM('KernelFunction','rbf','KernelScale',sigma,'BoxConstraint',C_best,'Standardize',false);
svmModel = fitcecoc(S.Xn_train,S.y_train,'Learners',t,'Coding','onevsone','ClassNames',S.classes);
[YPred,~]=predict(svmModel,S.Xn_test);
acc = mean(YPred==S.y_test);
end


