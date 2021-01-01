

%---- fold_1 ----------
mdl1 = ClassificationKNN.fit(Yeast_f1_train_feature,Yeast_f1_train_label,'NumNeighbors', 2);
predict_label1 = predict(mdl1, Yeast_f1_test_feature);
[ACC,SN,SP,PPV,NPV,F1,MCC] = roc1(predict_label1,Yeast_f1_test_label );
aa = [ACC,SN,SP,PPV,NPV,F1,MCC];

%---- fold_2 ----------
mdl2 = ClassificationKNN.fit(Yeast_f2_train_feature,Yeast_f2_train_label,'NumNeighbors', 2);
predict_label2 = predict(mdl2, Yeast_f2_test_feature);
[ACC,SN,SP,PPV,NPV,F1,MCC] = roc1(predict_label2,Yeast_f2_test_label );
bb = [ACC,SN,SP,PPV,NPV,F1,MCC];

%---- fold_3 ----------
mdl3 = ClassificationKNN.fit(Yeast_f3_train_feature,Yeast_f3_train_label,'NumNeighbors', 2);
predict_label3 = predict(mdl3, Yeast_f3_test_feature);
[ACC,SN,SP,PPV,NPV,F1,MCC] = roc1(predict_label3,Yeast_f3_test_label );
cc = [ACC,SN,SP,PPV,NPV,F1,MCC];

%---- fold_4 ----------
mdl4 = ClassificationKNN.fit(Yeast_f4_train_feature,Yeast_f4_train_label,'NumNeighbors', 2);
predict_label4 = predict(mdl4, Yeast_f4_test_feature);
[ACC,SN,SP,PPV,NPV,F1,MCC] = roc1(predict_label4,Yeast_f4_test_label );
dd = [ACC,SN,SP,PPV,NPV,F1,MCC];

%---- fold_5 ----------
mdl5 = ClassificationKNN.fit(Yeast_f5_train_feature,Yeast_f5_train_label,'NumNeighbors', 2);
predict_label5 = predict(mdl5, Yeast_f5_test_feature);
[ACC,SN,SP,PPV,NPV,F1,MCC] = roc1(predict_label5,Yeast_f5_test_label );
ee = [ACC,SN,SP,PPV,NPV,F1,MCC];

R=[aa;bb;cc;dd;ee];
