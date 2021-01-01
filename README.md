# SGANRDA
Emerging research shows that circular RNA (circRNA) plays a crucial role in the diagnosis, occurrence and prognosis of complex human diseases. Compared with traditional biological experiments, the computational method of fusing multi-source biological data to identify the association between circRNA and disease can effectively reduce costs and save time. Considering the limitations of existing computational models, we propose a semi-supervised Generative Adversarial Networks (GAN) model SGANRDA for predicting circRNA-disease association. This model first fused the natural language features of the circRNA sequence and the features of disease semantics, circRNA and disease Gaussian interaction profile (GIP) kernel, and then used all circRNA-disease pairs to pre-train the GAN network, and fine-tune the network parameters through labeled samples. Finally, the Extreme Learning Machine (ELM) classifier is employed to obtain the prediction result. Compared with the previous supervision model, SGANRDA innovatively introduced circRNA sequences and utilized all the information of circRNA-disease pairs during the pre-training process. This step can increase the information content of the feature to some extent and reduce the impact of too few known associations on the model performance. SGANRDA obtained AUC scores of 0.9411 and 0.9223 in leave-one-out cross-validation (LOOCV) and five-fold cross-validation, respectively. Prediction results on the benchmark dataset show that SGANRDA outperforms other existing models. In addition, 25 of the top 30 circRNA-disease pairs with the highest scores of SGANRDA in case studies were verified by recent literature. These experimental results demonstrate that SGANRDA is a useful model to predict the circRNA-disease association, and can provide reliable candidates for biological experiments.
