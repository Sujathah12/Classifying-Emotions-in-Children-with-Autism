function [Parameter]=Finding_parameter1(Train_tar,b,Test_len)


[C,order] = confusionmat((Train_tar),b);
Acc1=[sum(diag(C))/sum(C(:))];
Erate=abs(sum(C(:))-sum(diag(C)))/sum(C(:));

for i=1:length(C)
    
    TP(i)=C(i,i);
    TN1=C(i,:);
    TN1(i)=[];
    FP(i)=sum(TN1);
    TN2=diag(C);
    TN2(i)=[];
    TN(i)=sum(TN2);
    TN3=C;
     TN3(i,:)=[];
     
    FN(i)=sum(TN3(:))-sum(TN2);
%     TN(i)=sum(C(:))-(sum(C(:,i))+sum(C(i,:))-C(i,i));
    Pre(i)=C(i,i)/sum(C(:,i));
    Recal(i)=C(i,i)/sum(C(i,:));
    Pre1(i)=TP(i)/(TP(i)+FP(i));
    Recal1(i)=TP(i)/(TP(i)+FN(i));
    P_P_V(i)=TP(i)/(TP(i)+FP(i));
    N_P_V(i)=TN(i)/(TN(i)+FN(i));
    Sen(i)=TP(i)/(TP(i)+FN(i));
    Spec(i)=TN(i)/(TN(i)+FP(i));
    F_score(i)=2*TP(i)/(2*TP(i)+FP(i)+FN(i));
    M_CC(i)=((TP(i)*TN(i))-(FP(i)*FN(i)))/sqrt((TP(i)+FP(i))*(TP(i)+FN(i))*(TN(i)+FP(i))*(TN(i)+FN(i)));
    F_DR(i)=FP(i)/(FP(i)+TP(i));
    F_OR(i)=FN(i)/(FN(i)+TN(i));
    C_SI(i)=TP(i)/(TP(i)+FN(i)+FP(i));
    Miss_rate(i)=FN(i)/(TP(i)+FN(i));
    PPV(i)=TP(i)/(TP(i)+FP(i));

end
Sen(isnan(Sen))=0;
Spec(isnan(Spec))=0;
F_OR(isnan(F_OR))=0;
F_DR(isnan(F_DR))=0;
PPV(isnan(PPV))=0;
Miss_rate(isnan(Miss_rate))=0;

Parameter=struct;
Parameter.Accuracy=(Acc1);
Parameter.sensitivity=mean(Sen);
Parameter.specificity=mean(Spec);
Parameter.F1_score=mean(F_score);
Parameter.MCC=mean(abs(M_CC));
Parameter.FDR=mean(F_DR(1));
Parameter.FOR=mean(F_OR(1));
% Parameter.CSI=C_SI(1);
% Parameter.PPV=Recal1(1);
Parameter.PPV=mean(PPV(1));
Parameter.TP=TP;
Parameter.FP=FP;
Parameter.FN=FN;
Parameter.TN=TN;
Parameter.Classaccuracy=(TP/Test_len)';

end