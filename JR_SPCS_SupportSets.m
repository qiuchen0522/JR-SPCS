function [g_out, s_out, obj_history] = JR_SPCS_SupportSets(WW, LL, Nb, fNum, Nc, y, Phi, fai_w, fai_t, R, opts)
% JR_SPCS_SupportSets - Joint Reconstruction via Structural-Prior-Guided Compressive Sensing

%% --- 1. Parameter and Step Size Initialization ---
maxIter  = opts.maxIter;
tol      = opts.tol;

alpha = opts.alpha;  % Step size for wall component (g)
beta  = opts.beta;   % Step size for target component (s)

lambda_w = opts.lambda_w; % Regularization for wall nuclear norm
lambda_g = opts.lambda_g; % Regularization for wall L1 norm
lambda_s = opts.lambda_s; % Regularization for target L1 norm

%% --- 2. Variable Initialization ---
Qw_full = Nb * LL;
Qt_full = WW * LL;
g_out = zeros(Qw_full, 1);
s_out = zeros(Qt_full, 1);

% Track active support sets
idx_g = (1:Qw_full).';
idx_s = (1:Qt_full).';

% Current coefficients
g = g_out(idx_g);
s = s_out(idx_s);

% Backup full dictionaries for support set recovery
faiw_full = fai_w;
fait_full = fai_t;

% Initialize obj_history to store maxIter + 1 values (including Iter 0)
obj_history = zeros(maxIter + 1, 1);

%% --- 3. Initial Objective Calculation (Iteration 0) ---
res_0 = Phi * (faiw_full(:, idx_g) * g + fait_full(:, idx_s) * s) - y;
obj_history(1) = norm(res_0, 2)^2 ;

fprintf('Iter | Objective    | Delta Obj  | Wall Supp | Tar Supp\n');
fprintf('-------------------------------------------------------\n');
fprintf('%4d | %.4e | %10s | %9d | %8d\n', 0, obj_history(1), '---', numel(idx_g), numel(idx_s));

%% --- 4. Main Iteration Loop (Proximal Gradient Descent) ---
for it = 1:maxIter

    % --- Step A: Wall Component Update (g-update) ---
    % 1. Gradient Descent step
    current_fai_t = fait_full(:, idx_s);
    res = Phi * (fai_w * g + current_fai_t * s) - y;
    grad_g = fai_w' * (Phi' * res);
    a = g - alpha * grad_g;

    % 2. Singular Value Thresholding (SVT)
    Ya = reshape(faiw_full(:, idx_g) * a, [fNum, Nc]);
    [U, Sv, V] = svd(Ya, 'econ');
    d = max(diag(Sv) - alpha * lambda_w, 0);
    Ya_svt = U * diag(d) * V';

    % 3. Back-projection from low-rank echo to coefficients
    gstar = lsqminnorm(fai_w, reshape(Ya_svt, [], 1), 1e-1);

    % 4. Soft-thresholding
    g = sign(gstar) .* max(abs(gstar) - alpha * lambda_g, 0);

    % --- Step B: Target Component Update (s-update) ---
    % 1. Gradient Descent step
    res = Phi * (faiw_full(:, idx_g) * g + fait_full(:, idx_s) * s) - y;
    grad_s = current_fai_t' * (Phi' * res);
    b = s - beta * grad_s;

    % 2. Soft-thresholding
    s = sign(b) .* max(abs(b) - beta * lambda_s, 0);

    % --- Step C: Dynamic Support Set Pruning ---
    g_out(idx_g) = g;
    s_out(idx_s) = s;

    idx_g = find(abs(g_out) > 1e-6);
    idx_s = find(abs(s_out) > 1e-6);
    
    if isempty(idx_g), idx_g = 1; end % Safety check
    if isempty(idx_s), idx_s = 1; end

    % Update dictionaries and coefficients for the next iteration
    fai_w = faiw_full(:, idx_g);
    g = g_out(idx_g);
    s = s_out(idx_s);

    % --- Step D: Objective Function Calculation (Iteration it) ---
    res_final = Phi * (faiw_full(:, idx_g) * g + fait_full(:, idx_s) * s) - y;

    curr_obj =  norm(res_final, 2)^2;

    obj_history(it + 1) = curr_obj;

    % Monitor relative change
    delta_obj = (obj_history(it) - obj_history(it + 1)) / (obj_history(it) + eps);

    % Log progress every 5 iterations or at the first iteration
    if mod(it, 5) == 0 || it == 1
        fprintf('%4d | %.4e | %.4e | %9d | %8d\n', it, curr_obj, abs(delta_obj), numel(idx_g), numel(idx_s));
    end

    % Convergence check
    if it > 5 && abs(delta_obj) < tol
        fprintf('Converged at iteration %d\n', it);
        obj_history = obj_history(1:it + 1);
        break;
    end
end
end