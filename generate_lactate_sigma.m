function ed = generate_lactate_sigma(class_id, fmdl, theta_rad, rho_norm, sigma_bkg, seed, cfg)
% -------------------------------------------------------------------------
% generate_lactate_sigma(class_id, fmdl, theta_rad, rho_norm, sigma_bkg, seed, cfg)
%
% 生成 S1/S2/S3 的 elem_data，可通过 cfg.dataset_difficulty 调节“可分性难度”
%
% 参数:
%   class_id : 1=S1, 2=S2, 3=S3
%   fmdl      : EIDORS forward model
%   theta_rad : 元素角度（rad）
%   rho_norm  : 元素归一化半径
%   sigma_bkg : 背景导电率
%   seed      : 随机种子（可选）
%   cfg       : 可选结构体
%       .dataset_difficulty ∈ [0,1] (默认 0.3)
%       .S2.left_r, right_r, sr, sth
%       .S3.color
%
% MATLAB R2024b + EIDORS v3.12-ng
% -------------------------------------------------------------------------
if nargin < 7, cfg = struct(); end
if nargin >= 6 && ~isempty(seed), rng(seed); end
nElem = numel(rho_norm);
ed = sigma_bkg * ones(nElem,1);
deg = @(x) x*pi/180;

% ---------- 全局控制参数 ----------
difficulty = getfield_def(cfg,'dataset_difficulty',0.3);  % 0=易分, 1=难分
% 根据难度线性调节三个关键因子
contrast_scale = 1 - 0.6*difficulty;   % 导电率对比度缩放
noise_level    = 0.01 + 0.03*difficulty; % 噪声水平 (±%)
pos_jitter     = 3 + 10*difficulty;    % 角度扰动范围 (度)

% ---------- 工具函数 ----------
function w = gaussian_blob(rho, th, r0, th0, sr, sth)
    dth = wrapToPi(th - th0);
    rr  = (rho - r0)./sr;
    tt  = dth./sth;
    w   = exp(-0.5*(rr.^2 + tt.^2));
end

function ed_out = pour_blobs(ed_in, blobs, rho_norm, theta_rad, sigma_bkg)
    ed_acc = ed_in;
    for i = 1:numel(blobs)
        bb = blobs{i};
        w = gaussian_blob(rho_norm, theta_rad, bb.r0, bb.th0, bb.sr, bb.sth);
        ed_acc = ed_acc + w .* (bb.sigma_peak - sigma_bkg);
    end
    sigma_max = max(cellfun(@(b)b.sigma_peak, blobs));
    ed_out = min(max(ed_acc, sigma_bkg), sigma_max);
end

% ---------- 各类定义 ----------
switch class_id
    % ==========================================================
    case 1 % S1：前外侧小亮块 + 扰动
        th0 = deg(75 + pos_jitter*randn);
        r0  = 0.75 + 0.05*randn;
        sigma_peak = sigma_bkg + contrast_scale*(0.70 - sigma_bkg);
        blobs = {struct('r0',r0,'th0',th0,'sr',0.08,'sth',deg(12),'sigma_peak',sigma_peak)};
        ed = pour_blobs(ed, blobs, rho_norm, theta_rad, sigma_bkg);

    % ==========================================================
    case 2 % S2：后侧两块 + 前侧小块，加入类间重叠
        if ~isfield(cfg,'S2'), cfg.S2 = struct(); end
        left_r  = getfield_def(cfg.S2,'left_r',  0.45);
        right_r = getfield_def(cfg.S2,'right_r', 0.75);
        sr_s2   = getfield_def(cfg.S2,'sr',      0.14);
        sth_s2d = getfield_def(cfg.S2,'sth',     25);
        sth_s2  = deg(sth_s2d);

        th1 = deg(210 + pos_jitter*randn); r1 = left_r  + 0.03*randn;
        th2 = deg(300 + pos_jitter*randn); r2 = right_r + 0.03*randn;
        th3 = deg(75  + pos_jitter*randn); r3 = 0.80     + 0.03*randn;

        sig1 = sigma_bkg + contrast_scale*(0.75 - sigma_bkg);
        sig2 = sigma_bkg + contrast_scale*(0.70 - sigma_bkg);
        sig3 = sigma_bkg + contrast_scale*(0.68 - sigma_bkg);

        blobs = {
            struct('r0',r1,'th0',th1,'sr',sr_s2,'sth',sth_s2,'sigma_peak',sig1), ...
            struct('r0',r2,'th0',th2,'sr',sr_s2,'sth',sth_s2,'sigma_peak',sig2), ...
            struct('r0',r3,'th0',th3,'sr',0.10,'sth',deg(12),'sigma_peak',sig3)
        };
        ed = pour_blobs(ed, blobs, rho_norm, theta_rad, sigma_bkg);

    % ==========================================================
    case 3 % S3：前亮点 + 后侧大片（缩小+淡黄）
        if ~isfield(cfg,'S3'), cfg.S3 = struct(); end
        back_color = getfield_def(cfg.S3,'color', 0.20 + 0.05*rand); % 随机微变

        % 前亮点
        thF = deg(75 + pos_jitter*randn);
        rF  = 0.80 + 0.03*randn;
        sigF = sigma_bkg + contrast_scale*(0.72 - sigma_bkg);
        blob_front = struct('r0',rF,'th0',thF,'sr',0.09,'sth',deg(14),'sigma_peak',sigF);
        ed = pour_blobs(ed, {blob_front}, rho_norm, theta_rad, sigma_bkg);

        % 后侧大片（淡黄色、较小）
        rot = deg((rand-0.5)*16);
        th_a = deg(215)+rot; th_b = deg(335)+rot;
        r_lo = 0.55; r_hi = 0.85;
        mask_ang = in_sector(theta_rad, th_a, th_b);
        mask_rad = (rho_norm>=r_lo)&(rho_norm<=r_hi);
        mask_back = mask_ang & mask_rad;
        ed(mask_back) = back_color;
end

% ---------- 加入噪声扰动 ----------
ed = ed + noise_level * sigma_bkg * randn(size(ed));
ed = max(ed, sigma_bkg*0.8);   % 防止出现负值
ed = min(ed, max(ed)*1.2);

end

% ======= 工具函数 =======
function v = getfield_def(s, name, def)
  if isstruct(s) && isfield(s,name), v = s.(name);
  else, v = def; end
end

function m = in_sector(th,a,b)
 th = mod(th,2*pi); a=mod(a,2*pi); b=mod(b,2*pi);
 if a<=b, m=(th>=a)&(th<=b);
 else, m=(th>=a)|(th<=b);
 end
end
