function [best_lambda, mse_vs_lam] = select_lambda_cv_dataresidual( ...
            imdl, fmdl_inv, data_ref, V_all, lams, K)
% -------------------------------------------------------------------------
% 通过 K 折交叉验证 (data-domain) 选择最优正则化系数 λ
%
% 输入:
%   imdl       - inverse model (已含 solve 与 prior)
%   fmdl_inv   - inverse 模型对应的 fwd_model
%   data_ref   - 背景电压结构体 (fwd_solve(img_bkg))
%   V_all      - N×M 电压矩阵 (每行为一个样本的40维电压)
%   lams       - λ 扫描向量，如 logspace(-3,1,10)
%   K          - 折数 (建议5)
%
% 输出:
%   best_lambda - 交叉验证最优 λ
%   mse_vs_lam  - 每个 λ 对应的平均 MSE 向量
%
% -------------------------------------------------------------------------

if nargin < 6, K = 5; end
N = size(V_all,1);

% 随机划分K折
idx = randperm(N);
fold_sizes = repmat(floor(N/K),1,K);
fold_sizes(1:mod(N,K)) = fold_sizes(1:mod(N,K))+1;
folds = mat2cell(idx,1,fold_sizes);

mse_vs_lam = zeros(numel(lams),1);
fprintf('\n=== Data-domain K-fold CV for λ ===\n');

for li = 1:numel(lams)
    lam = lams(li);
    imdl.hyperparameter.value = lam;
    fold_mse = zeros(K,1);

    for k = 1:K
        te_idx = folds{k};
        se_acc = 0; cnt = 0;

        for n = te_idx
            % 构造 data 结构体
            data_meas = data_ref;
            data_meas.meas = V_all(n,:)';

            % 差分重建
            img_rec = inv_solve(imdl, data_ref, data_meas);

            % 预测电压 (forward)
            data_pred = fwd_solve(img_rec);

            % 数据域残差
            err = data_pred.meas - V_all(n,:)';
            se_acc = se_acc + mean(err.^2);
            cnt = cnt + 1;
        end

        fold_mse(k) = se_acc / max(cnt,1);
    end

    mse_vs_lam(li) = mean(fold_mse);
    fprintf('  λ = %.4g  →  CV-MSE = %.3e\n', lam, mse_vs_lam(li));
end

% 选择最优 λ
[~,best_i] = min(mse_vs_lam);
best_lambda = lams(best_i);
fprintf('>>> 最优 λ = %.4f\n', best_lambda);

% 可视化 λ 曲线
figure('Name','CV Curve (Data-domain)');
semilogx(lams, mse_vs_lam, '-o', 'LineWidth',1.4);
grid on; xlabel('\lambda'); ylabel('CV-MSE (data-domain)');
title('Cross-Validation for Regularization Coefficient \lambda');
hold on;
plot(best_lambda, mse_vs_lam(best_i), 'ro', 'MarkerFaceColor','r');
text(best_lambda, mse_vs_lam(best_i)*1.1, sprintf('\\lambda^*=%.3f',best_lambda), ...
     'HorizontalAlignment','center', 'FontSize',10, 'Color','r');
hold off;
end

