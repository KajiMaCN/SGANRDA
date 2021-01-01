%fold-1
B1=TreeBagger(80,Yeast_f1_train_feature,Yeast_f1_train_label,'Method', 'classification');
predict_f1=predict(B1,Yeast_f1_test_feature);

[ACC,SN,SP,PPV,NPV,F1,MCC] = roc1(str2num(char(predict_f1)),Yeast_f1_test_label);
aa = [ACC,SN,SP,PPV,NPV,F1,MCC];

%fold-2
B2=TreeBagger(80,Yeast_f2_train_feature,Yeast_f2_train_label,'Method', 'classification');
predict_f2=predict(B2,Yeast_f2_test_feature);

[ACC,SN,SP,PPV,NPV,F1,MCC] = roc1(str2num(char(predict_f2)),Yeast_f2_test_label);
bb = [ACC,SN,SP,PPV,NPV,F1,MCC];

%fold-3
B3=TreeBagger(80,Yeast_f3_train_feature,Yeast_f3_train_label,'Method', 'classification');
predict_f3=predict(B3,Yeast_f3_test_feature);

[ACC,SN,SP,PPV,NPV,F1,MCC] = roc1(str2num(char(predict_f3)),Yeast_f3_test_label);
cc = [ACC,SN,SP,PPV,NPV,F1,MCC];

%fold-4
B4=TreeBagger(80,Yeast_f4_train_feature,Yeast_f4_train_label,'Method', 'classification');
predict_f4=predict(B4,Yeast_f4_test_feature);

[ACC,SN,SP,PPV,NPV,F1,MCC] = roc1(str2num(char(predict_f4)),Yeast_f4_test_label);
dd = [ACC,SN,SP,PPV,NPV,F1,MCC];

%fold-5
B5=TreeBagger(80,Yeast_f5_train_feature,Yeast_f5_train_label,'Method', 'classification');
predict_f5=predict(B5,Yeast_f5_test_feature);

[ACC,SN,SP,PPV,NPV,F1,MCC] = roc1(str2num(char(predict_f5)),Yeast_f5_test_label);
ee = [ACC,SN,SP,PPV,NPV,F1,MCC];

R=[];
R=[aa;bb;cc;dd;ee];  