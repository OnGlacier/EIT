function export_recon_and_forward_results(fmdl, imdl, ...
    V_all, y_all, data_ref, sigma_bkg, GT_sigma, base_path)
% -------------------------------------------------------------------------
% 功能：
%   1. 批量导出单个重建图（分类存放 E:\EIT_N\S1\S2\S3，300 dpi）
%   2. 导出三类平均重建图（show_slices）
%   3. 导出三类平均正模型图（show_fem）
% 参数：
%   fmdl, imdl, V_all, y_all, data_ref, sigma_bkg, GT_sigma
%   base_path : 根目录，例如 'E:\EIT_N'
% -------------------------------------------------------------------------
fprintf('\n=== 导出重建结果与正模型均值图到 %s ===\n', base_path);
if ~exist(base_path,'dir'), mkdir(base_path); end
class_names = {'S1','S2','S3'};
for c = 1:3
    folder = fullfile(base_path, class_names{c});
    if ~exist(folder,'dir'), mkdir(folder); end
end

N = size(V_all,1);
% --------- 1) 导出每张重建结果 ---------
for n = 1:N
    data_meas      = data_ref;
    data_meas.meas = V_all(n,:)';
    img_rec = inv_solve(imdl, data_ref, data_meas);

    fig = figure('Visible','off');
    show_slices(img_rec); axis equal tight; axis off;
    title(sprintf('Sample #%d  %s', n, class_names{y_all(n)}), 'FontSize',10);
    caxis([sigma_bkg 0.70]);
    fname = sprintf('%s_%04d.png', class_names{y_all(n)}, n);
    print(fig, fullfile(base_path, class_names{y_all(n)}, fname), '-dpng', '-r300');
    close(fig);

    if mod(n,100)==0
        fprintf('  已导出 %d/%d 张重建图...\n', n, N);
    end
end

% --------- 2) 平均重建图 (show_slices) ---------
fprintf('--- 计算三类平均重建图 ---\n');
avg_rec_imgs = cell(1,3);
for c = 1:3
    idx_c = find(y_all==c);
    acc = 0; cnt = 0;
    for n = idx_c'
        data_meas      = data_ref;
        data_meas.meas = V_all(n,:)';
        img_rec = inv_solve(imdl, data_ref, data_meas);
        acc = acc + img_rec.elem_data;
        cnt = cnt + 1;
    end
    mean_ed = acc / max(cnt,1);
    avg_rec_imgs{c} = mk_image(fmdl, mean_ed);

    fig = figure('Visible','off');
    show_slices(avg_rec_imgs{c}); axis equal tight; axis off;
    title(sprintf('Mean Reconstruction - %s', class_names{c}), 'FontSize',10);
    caxis([sigma_bkg 0.70]);
    print(fig, fullfile(base_path, sprintf('Mean_Recon_%s.png', class_names{c})), ...
          '-dpng', '-r300');
    close(fig);
end

% --------- 3) 平均正模型图 (show_fem) ---------
fprintf('--- 计算三类平均正模型图 ---\n');
avg_true_imgs = cell(1,3);
for c = 1:3
    idx_c = find(y_all==c);
    acc = 0; cnt = 0;
    for n = idx_c'
        acc = acc + GT_sigma{n};
        cnt = cnt + 1;
    end
    mean_ed = acc / max(cnt,1);
    avg_true_imgs{c} = mk_image(fmdl, mean_ed);

    fig = figure('Visible','off');
    show_fem(avg_true_imgs{c}); axis equal tight; axis off;
    title(sprintf('Mean Forward Model - %s', class_names{c}), 'FontSize',10);
    caxis([sigma_bkg 0.70]);
    print(fig, fullfile(base_path, sprintf('Mean_Forward_%s.png', class_names{c})), ...
          '-dpng', '-r300');
    close(fig);
end

fprintf('✓ 已导出全部单张重建图及三类平均正模型/重建图到 %s\n', base_path);
end
