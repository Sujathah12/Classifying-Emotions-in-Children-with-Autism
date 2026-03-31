function Error = jFitnessFunction1(feat, label, X, opts)
    % X is the binary mask (1 = selected, 0 = not selected)
    if sum(X == 1) == 0
        Error = 1; % Return maximum error if no features selected
        return;
    end
    
    % Select the features
    xtrain = feat(:, X == 1);
    
    % Perform 5-Fold Cross Validation using KNN (K=5)
    % This is standard and ensures the error is stable
    try
        % We create the model and calculate loss in one step to avoid 'Model' errors
        Model = fitcknn(xtrain, label, 'NumNeighbors', 5, 'KFold', 5);
        Error = kfoldLoss(Model);
    catch
        Error = 1; % Fallback for math errors
    end
end
