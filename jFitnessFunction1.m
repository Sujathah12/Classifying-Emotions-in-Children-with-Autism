function Error = jFitnessFunction1(feat, label, mask, opts)
    % Uses a fast KNN to evaluate the quality of the selected subset
    xtrain = feat(:, mask == 1);
    
    % Ensure valid data
    if isempty(xtrain)
        Error = 1;
        return;
    end
    
    % 5-Fold Cross Validation
    k = 5;
    cv = cvpartition(label, 'KFold', k);
    err = zeros(k, 1);
    
    for i = 1:k
        tr_idx = cv.training(i);
        te_idx = cv.test(i);
        
        % Train KNN (K=5)
        mdl = fitcknn(xtrain(tr_idx,:), label(tr_idx), 'NumNeighbors', 5);
        pred = predict(mdl, xtrain(te_idx,:));
        
        % Calculate loss
        err(i) = sum(~strcmp(categorical(pred), categorical(label(te_idx)))) / length(te_idx);
    end
    
    Error = mean(err);
end