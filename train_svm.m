%% STEP 3.2 - 训练 SVM (RBF) + ECOC 多分类模型
clear; clc;

%% === 1. 载入准备好的数据 ===
load('eit_3_2_dataset.mat');  % 其中包含 Xn_train, y_train, Xn_test, y_test, ps, classes

fprintf('训练集大小: %d×%d\n', size(Xn_train,1), size(Xn_train,2));
fprintf('测试集大小:  %d×%d\n', size(Xn_test,1), size(Xn_test,2));

%% === 2. 构建 ECOC 模型模板 ===
% 使用 one-vs-one, 基分类器为 RBF 核 SVM
template = templateSVM( ...
    'KernelFunction','rbf', ...
    'KernelScale','auto', ...  % gamma = 1/(2*sigma^2)，MATLAB会自动估计
    'BoxConstraint',1, ...     % 即惩罚系数C
    'Standardize',false);      % 已经标准化过，无需重复

%% === 3. 定义交叉验证参数搜索空间 ===
C_range     = logspace(-1, 3, 9);   % 0.1 → 1000
gamma_range = logspace(-3, 1, 9);   % 0.001 → 10

bestAcc = 0;
bestC = NaN; bestGamma = NaN;

fprintf('=== 5折交叉验证调参中 ===\n');
for Ci = 1:numel(C_range)
    for Gi = 1:numel(gamma_range)
        C = C_range(Ci);
        gamma = gamma_range(Gi);

        % 定义子分类器模板
        t = templateSVM('KernelFunction','rbf', ...
                        'KernelScale',1/sqrt(2*gamma), ...
                        'BoxConstraint',C, ...
                        'Standardize',false);

        % ECOC多分类模型
        model = fitcecoc(Xn_train, y_train, ...
            'Learners',t, ...
            'Coding','onevsone', ...
            'KFold',5, ...
            'Verbose',0);

        % 交叉验证准确率
        cvLoss = kfoldLoss(model);
        cvAcc  = 1 - cvLoss;

        if cvAcc > bestAcc
            bestAcc = cvAcc;
            bestC = C;
            bestGamma = gamma;
        end
    end
end

fprintf('最优参数: C = %.3f, gamma = %.4f, 5折CV准确率 = %.2f%%\n', ...
    bestC, bestGamma, bestAcc*100);

%% === 4. 用最优参数训练最终模型 ===
t_best = templateSVM('KernelFunction','rbf', ...
                     'KernelScale',1/sqrt(2*bestGamma), ...
                     'BoxConstraint',bestC, ...
                     'Standardize',false);

svmModel = fitcecoc(Xn_train, y_train, ...
    'Learners',t_best, ...
    'Coding','onevsone', ...
    'ClassNames',classes);

%% === 5. 在测试集上评估 ===
y_pred = predict(svmModel, Xn_test);

% 🔧 统一类型：把 y_pred 转成与 y_test 相同的 categorical
y_pred = categorical(string(y_pred), categories(y_test), 'Ordinal',false);

acc_test = mean(y_pred == y_test);
fprintf('测试集准确率 = %.2f%%\n', acc_test*100);

%% === 6. 混淆矩阵 ===
figure;
confusionchart(y_test, y_pred, ...
    'Title','SVM-ECOC 混淆矩阵', ...
    'FontSize',12);

%% === 7. 模型体检：支持向量 & 可分性可视化 ===

% ---------- A. 支持向量统计 ----------
% svmModel 是 fitcecoc 返回的 ECOC 模型，里面每个二分类 Learner 是一个 SVM
learners = svmModel.BinaryLearners;
nLearners = numel(learners);
fprintf('\n支持向量统计：\n');
totalSV = 0;
for i = 1:nLearners
    sv = learners{i}.SupportVectors;
    fprintf('  子分类器 %d ：支持向量数 = %d\n', i, size(sv,1));
    totalSV = totalSV + size(sv,1);
end
fprintf('  平均每个SVM使用 %.1f 个支持向量\n', totalSV/nLearners);
fprintf('  训练样本总数 = %d\n', size(Xn_train,1));
fprintf('  支持向量占比约 %.2f%%\n', 100*(totalSV/nLearners)/size(Xn_train,1));

% ---------- B. PCA 降维可视化 ----------
[coeff, score, latent, tsq, explained] = pca(Xn_train);
figure;
gscatter(score(:,1), score(:,2), y_train, 'rgb', 'o^s');
xlabel(sprintf('PC1 (%.1f%%方差)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%方差)', explained(2)));
title('训练集 PCA 可视化');
grid on; axis equal;

% ---------- C. 测试集投影 ----------
score_test = (Xn_test - mean(Xn_train)) * coeff(:,1:2);
figure;
gscatter(score_test(:,1), score_test(:,2), y_test, 'rgb', 'o^s');
xlabel('PC1'); ylabel('PC2');
title('测试集 PCA 投影');
grid on; axis equal;
