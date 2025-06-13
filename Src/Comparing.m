%% === Setup ===
package_path = fullfile('..', 'ExternalPackages', 'PH-STAT-main');
addpath(package_path);
%%
%% === Input Subject IDs ===
male_subject   = '100610';   
female_subject = '106521';
%%
data_dir = '../Usedata/rfMRI_ts/';
file_male   = fullfile(base_path, [male_subject '_REST1_LR_AAL_corr_vec.txt']);
file_female = fullfile(base_path, [female_subject '_REST1_LR_AAL_corr_vec.txt']);
win_size = 60;
%%
%% === Function: Load correlation vectors to 3D tensor ===
function D_tensor = load_corrvec_to_tensor(txt_path, win_size)
    V = readmatrix(txt_path);
    n_series = size(V, 2);
    n_points = n_series - win_size + 1;
    D_all = cell(1, n_points);

    for t = 1:n_points
        X = V(:, t:t+win_size-1)';
        D = squareform(pdist(X, 'euclidean'));
        D = double(D);
        D(isnan(D) | isinf(D)) = 999;
        D(1:end+1:end) = 0; % 대각 원소를 0으로 설정
        D = (D + D') / 2;   % 대칭성 보장
        D_all{t} = D;
    end

    D_tensor = cat(3, D_all{:});
end
%%
%% === Load and process both subjects ===
D_male   = load_corrvec_to_tensor(file_male, win_size);
D_female = load_corrvec_to_tensor(file_female, win_size);
%%
%% === Calculate Integrated Distance ===
fprintf('Calculating Integrated Wasserstein Distance between two subjects...\n');

n_windows_min = min(size(D_male, 3), size(D_female, 3));

total_integrated_distance = 0;

for t = 1:n_windows_min
    
    D_male_t   = D_male(:,:,t);
    D_female_t = D_female(:,:,t);
    
    [Wb_male_t, Wd_male_t]     = WS_decompose(D_male_t);
    [Wb_female_t, Wd_female_t] = WS_decompose(D_female_t);
    
    dist_0D_t = sqrt(sum((Wb_male_t(:,3) - Wb_female_t(:,3)).^2));
    dist_1D_t = sqrt(sum((Wd_male_t(:,3) - Wd_female_t(:,3)).^2));
    
    window_dist = dist_0D_t + dist_1D_t;

    total_integrated_distance = total_integrated_distance + window_dist;
end

fprintf('--------------------------------------------------\n');
fprintf('Total Integrated Distance between %s and %s: %.4f\n', ...
        male_subject, female_subject, total_integrated_distance);
fprintf('--------------------------------------------------\n');
%%
%% === Plot Side-by-Side Trajectories ===
z_male   = WS_embed(D_male);
z_female = WS_embed(D_female);

figure;
subplot(1,2,1);
scatter(z_male.x, z_male.y, 10, 1:length(z_male.x), 'filled');
colormap(jet); colorbar;
xlabel('Deviation in birth'); ylabel('Deviation in death');
title(sprintf('Male Subject: %s', male_subject));
axis square; grid on;

subplot(1,2,2);
scatter(z_female.x, z_female.y, 10, 1:length(z_female.x), 'filled');
colormap(jet); colorbar;
xlabel('Deviation in birth'); ylabel('Deviation in death');
title(sprintf('Female Subject: %s', female_subject));
axis square; grid on;

sgtitle('Topological Trajectories Comparison');