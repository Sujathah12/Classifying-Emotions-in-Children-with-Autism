clc; clear; close all;
addpath(genpath('.'));
rng default

% Load your extracted features
load Fea_resnet; load Fea_eff; load fea;

for N1=1:1 
    %% 1. Dataset Setup
    digitDatasetPath = './Data_out/';
    imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders',true);
    
    % Ensure labels are set correctly
    Train_tar = repmat([1:8],[100,1]);
    Train_tar = categorical(Train_tar(:));
    imds.Labels = Train_tar;
    
    features = feature; % Total 3328 features
    disp(['Feature length of network is: ' num2str(size(features,2))])

    %% 2. Split Data
    [imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomize');
    
    % Prepare Training and Testing sets
    Train1 = feature(161:end,:);
    Test1 = feature(1:160,:);
    Train_tar_num = double(imdsTrain.Labels);
    Test_tar_num = double(imdsValidation.Labels); 

    %% 3. PROPOSED MSG-FS NOVELTY (Feature Selection)
    opts.N = 10; opts.T = 20; opts.lambda = 0.5;
    
    tic
    fprintf('\n--- Phase 3: Executing Proposed MSG-FS Novelty ---\n');
    FS = jMSGFS(Train1, Train_tar_num, opts); 
    toc
    
    sf_idx = FS.sf;
    Train_selected = Train1(:, sf_idx);
    Test_selected = Test1(:, sf_idx);

   %% ========================================================================
%% PHASE 5: NOVELTY VISUALIZATION (Facial Muscle Interaction Graph)
%% ========================================================================
fprintf('\nPHASE 5: Generating Topological Interaction Maps for TL...\n');

% DYNAMICALLY pick 4 samples from your validation set to avoid "Index Error"
val_total = length(imdsValidation.Files);
sample_indices = round(linspace(1, val_total, 4)); % Automatically picks 4 safe indices

cat_label = {'Anger','Contempt','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise'};

% Anatomical Landmark positions for the graph (Eyes, Nose, Mouth)
face_landmarks = [
    70, 80; 90, 75; 110, 80;  % Eyebrows
    140, 80; 160, 75; 180, 80; 
    80, 105; 100, 105;         % Eyes
    150, 105; 170, 105;        
    125, 120; 125, 140; 125, 155; % Nose
    90, 180; 110, 175; 130, 175; 150, 180; % Mouth
    100, 195; 125, 200; 150, 195 
];

for img_count = 1:4
    idx_to_show = sample_indices(img_count);
    img = imread(imdsValidation.Files{idx_to_show});
    img = imresize(img, [224 224]);
    
    % Get the top 20 most important features from our MSG-FS algorithm
    [~, top_v_idx] = sort(FS.Shapley_Influence, 'descend');
    selected_nodes = top_v_idx(1:20);
    
    % Build the interaction graph (The "Handshake")
    Adj = FS.MI_Graph(selected_nodes, selected_nodes);
    Adj(Adj < mean(Adj(:))*1.1) = 0; % Filter weak connections
    G_viz = graph(Adj);
    
    % Create Figure
    figure('Name', ['Proposed MSG-FS Map ', num2str(img_count)], 'Color', 'w');
    imshow(img); hold on;
    
    % Plot the Graph OVER the facial landmarks
    p_viz = plot(G_viz, 'XData', face_landmarks(1:20,1), 'YData', face_landmarks(1:20,2), ...
                'LineWidth', 2, 'NodeLabel', {});
            
    % Color the dots based on Shapley Importance (Red = High, Blue = Low)
    p_viz.NodeCData = FS.Shapley_Influence(selected_nodes); 
    p_viz.MarkerSize = 12; 
    p_viz.EdgeColor = [0 1 1]; % Cyan lines for Interaction
    p_viz.EdgeAlpha = 0.6;
    
    colormap(jet); h = colorbar;
    ylabel(h, 'Shapley Contribution (Importance Score)');
    
    % Get actual label
    actual_cat = double(imdsValidation.Labels(idx_to_show));
    title(['MSG-FS Muscle Map: ', cat_label{actual_cat}]);
    hold off;
end 


    %% 5. Classification (Remaining code stays exactly the same)
    % ... (Your existing KNN, RF, SVM, Ensemble code) ...
end