function [Parameter]=parameter_finder(Train_tar,b,Test_len)
[confMatrix,order] = confusionmat(Train_tar,b);
num_classes=length(order);
% Initialize TP, TN, FP, FN
TP = zeros(1, num_classes);
TN = zeros(1, num_classes);
FP = zeros(1, num_classes);
FN = zeros(1, num_classes);

% Calculate TP, TN, FP, FN for each class
for i = 1:num_classes
    TP(i) = confMatrix(i, i); % True Positives
    FN(i) = sum(confMatrix(i, :)) - TP(i); % False Negatives
    FP(i) = sum(confMatrix(:, i)) - TP(i); % False Positives
    TN(i) = sum(confMatrix(:)) - (TP(i) + FN(i) + FP(i)); % True Negatives
end
% Initialize metrics
Precision = zeros(1, num_classes);
Recall = zeros(1, num_classes);
F1Score = zeros(1, num_classes);
MCC = zeros(1, num_classes);
FDR = zeros(1, num_classes);
FOR = zeros(1, num_classes);
MissRate = zeros(1, num_classes);
Specificity = zeros(1, num_classes);

% Total metrics
Total_TP = sum(TP);
Total_FP = sum(FP);
Total_FN = sum(FN);
Total_TN = sum(TN);

% Accuracy
Accuracy = (Total_TP + Total_TN) / (Total_TP + Total_FP + Total_FN + Total_TN);

% Compute metrics per class
for i = 1:num_classes
    Precision(i) = TP(i) / (TP(i) + FP(i));
    Recall(i) = TP(i) / (TP(i) + FN(i));
    F1Score(i) = 2 * (Precision(i) * Recall(i)) / (Precision(i) + Recall(i));
    MCC(i) = (TP(i) * TN(i) - FP(i) * FN(i)) / sqrt((TP(i) + FP(i)) * (TP(i) + FN(i)) * (TN(i) + FP(i)) * (TN(i) + FN(i)));
    FDR(i) = FP(i) / (FP(i) + TP(i));
    FOR(i) = FN(i) / (FN(i) + TN(i));
    MissRate(i) = FN(i) / (FN(i) + TP(i));
    Specificity(i) = TN(i) / (TN(i) + FP(i));
end
Precision(isnan(Precision))=0;
Recall(isnan(Recall))=0;
F1Score(isnan(F1Score))=0;
MCC(isnan(MCC))=0;
FDR(isnan(FDR))=0;
FOR(isnan(FOR))=0;
MissRate(isnan(MissRate))=0;
Specificity(isnan(Specificity))=0;
% Error Rate
ErrorRate = (Total_FP + Total_FN) / (Total_TP + Total_FP + Total_FN + Total_TN);

Parameter=struct;
Parameter.Accuracy=(Accuracy);
Parameter.Recall=mean(Recall);
Parameter.Specificity=mean(Specificity);
Parameter.F1Score=mean(F1Score);
Parameter.MCC=mean(abs(MCC));
Parameter.FDR=mean(FDR);
Parameter.FOR=mean(FOR);
Parameter.Precision=mean(Precision);
Parameter.MissRate=mean(MissRate);
Parameter.ErrorRate=mean(ErrorRate);

Parameter.TP=TP;
Parameter.FP=FP;
Parameter.FN=FN;
Parameter.TN=TN;
Parameter.Classaccuracy=(TP/Test_len)';
% AUC using perfcurve (for binary classifiers or per class)
% Example for class 1:
% [X, Y, T, AUC_class1] = perfcurve(true_labels == 1, predicted_probs(:,1), 1);

end