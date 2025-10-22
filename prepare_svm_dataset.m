function prepare_svm_dataset(X, y, out_mat)
% STEP 3.2 - 数据准备脚本
% X: N×40 电压数据
% y: N×1 标签 (数值 1/2/3 或 分类 {'S1','S2','S3'})
% out_mat: 输出文件名 (例如 'eit_3_2_dataset.mat')

if nargin < 3, out_mat = 'eit_3_2_dataset.mat'; end

% —— 基本检查 ——
assert(size(X,2)==40, 'X 应该是 N×40（相邻激励-相邻测量共40个通道）');
assert(size(X,1)==numel(y), 'X 行数与 y 长度不一致');

% 标签统一为分类类型，保持顺序 S1,S2,S3
if isnumeric(y)
    cats = {'S1','S2','S3'};
    y = categorical(y, [1 2 3], cats);
elseif iscellstr(y) || isstring(y) || iscategorical(y)
    y = categorical(string(y));
    % 映射到标准顺序
    y = reordercats(y, intersect({'S1','S2','S3'}, categories(y), 'stable'));
else
    error('y 类型不支持，请用 1/2/3 或 ''S1''/''S2''/''S3''');
end

% —— 按类别分层抽样：每类测试 10 组，共 30 组；其余为训练 1200 组 ——
rng(2025); % 可复现
classes = categories(y);
idxTrain = false(size(y));
idxTest  = false(size(y));

for c = 1:numel(classes)
    idxc = find(y==classes{c});
    % 随机打乱
    idxc = idxc(randperm(numel(idxc)));
    % 取前 10 个作为测试，其余为训练（论文：每类410→测试10，其余400训练）:contentReference[oaicite:2]{index=2}
    n_test = min(10, numel(idxc));
    idxTest(idxc(1:n_test)) = true;
    idxTrain(idxc(n_test+1:end)) = true;
end

% —— 特征标准化（mapminmax 到 [0,1]），只用训练集拟合映射 —— 
[Xn_train, ps] = mapminmax(X(idxTrain,:)', 0, 1); % 注意 mapminmax 期望特征为列
Xn_train = Xn_train';
Xn_test  = mapminmax('apply', X(idxTest,:)', ps)';  % 用同一映射作用在测试集

y_train = y(idxTrain);
y_test  = y(idxTest);

% 统计信息
fprintf('数据概览：\n');
for c = 1:numel(classes)
    ntr = sum(y_train==classes{c});
    nte = sum(y_test==classes{c});
    fprintf('  %s -> 训练 %d, 测试 %d\n', classes{c}, ntr, nte);
end
fprintf('总计：训练 %d，测试 %d\n', numel(y_train), numel(y_test));

% —— 保存到 .mat，后续训练直接加载 —— 
save(out_mat, 'Xn_train','y_train','Xn_test','y_test','ps','classes');
fprintf('已保存到 %s\n', out_mat);
end

