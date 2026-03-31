function FS = jMSGFS(feat, label, opts)
    % MSG-FS: Mutual-Information-weighted Shapley Graph Feature Selection
    % Proposed Novelty: Graph Topology + Game Theory + Information Theory
    
    rng default;
    [num_samples, num_features] = size(feat);
    
    % Parameters
    if isfield(opts,'N'), N = opts.N; else N = 10; end
    if isfield(opts,'T'), Max_iter = opts.T; else Max_iter = 20; end
    if isfield(opts,'lambda'), lambda = opts.lambda; else lambda = 0.5; end

    fprintf('\n--- Step 1: Building Feature Interaction Graph (ShapG Logic) ---\n');
    % We use Correlation as a proxy for Mutual Information Topology
    C = corr(feat); 
    % Map information-theoretic distances: MI approx: -0.5 * log(1 - R^2)
    MI_Graph = -0.5 * log(1 - C.^2 + eps); 
    MI_Graph(isinf(MI_Graph)) = 10; % Cap infinity

    fprintf('--- Step 2: Estimating Graph-Shapley Influence ---\n');
    % Topological Importance: Features connected to most information have highest Shapley Influence
    Shapley_Influence = sum(MI_Graph, 2); 

    fprintf('--- Step 3: Information Reward Optimization (MI-SHAP Logic) ---\n');
    % Initialize selection mask
    pop = rand(N, num_features) > 0.8; 
    fit = zeros(N, 1);

    % Initial Fitness Evaluation
    for i = 1:N
        fit(i) = calculate_reward_score(feat, label, pop(i,:), Shapley_Influence, MI_Graph, opts, lambda);
    end

    [gBestScore, idx] = min(fit);
    gBest = pop(idx, :);

    % Iterative Refinement
    for t = 1:Max_iter
        for i = 1:N
            % Mutation based on Graph Influence weights 
            % (Nodes with high Shapley values are preserved)
            mut_rate = 1 ./ (1 + exp(Shapley_Influence' / max(Shapley_Influence)));
            mask = rand(1, num_features) < (0.05 * mut_rate);
            new_sol = xor(pop(i,:), mask);
            
            % Evaluate new information reward
            new_fit = calculate_reward_score(feat, label, new_sol, Shapley_Influence, MI_Graph, opts, lambda);
            
            if new_fit < fit(i)
                pop(i,:) = new_sol;
                fit(i) = new_fit;
            end
            
            if new_fit < gBestScore
                gBestScore = new_fit;
                gBest = new_sol;
            end
        end
        fprintf('MSG-FS Iteration %d: Best Reward Fit = %f\n', t, gBestScore);
    end

    % Result Formatting
    Sf = find(gBest == 1);
    sFeat = feat(:, Sf);

    FS.sf = Sf;
    FS.ff = sFeat;
    FS.nf = length(Sf);
    FS.f  = feat;
    FS.l  = label;
 % --- ADD THESE TWO LINES FOR VISUALIZATION ---
    FS.Shapley_Influence = Shapley_Influence; % Needed for dot size
    FS.MI_Graph = MI_Graph;                 % Needed for drawing lines
end

% --- MI-SHAP Reward Function (The Core Scientific Novelty) ---
function score = calculate_reward_score(feat, label, mask, Shapley, MI_G, opts, lambda)
    if sum(mask) == 0
        score = 1e10; return;
    end
    
    % 1. Utility (Classification Error)
    error = jFitnessFunction1(feat, label, mask, opts);
    
    % 2. Information Redundancy (MI Penalty)
    idx = find(mask == 1);
    sub_graph = MI_G(idx, idx);
    redundancy = sum(sum(sub_graph)) / (length(idx)^2);
    
    % 3. Graph-Shapley Importance (Information Reward)
    importance = mean(Shapley(idx));
    
    % Combined Fitness: Minimize (Error + Redundancy) and Maximize (Importance)
    % Framed as minimization:
    score = (0.7 * error) + (lambda * redundancy) - (0.1 * importance);
end