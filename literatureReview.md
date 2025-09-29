Chapter 2 – Literature Review
2.1 Introduction

Fraud detection has been an active area of research for several decades, motivated by the increasing complexity of fraudulent activities and the significant financial losses they cause. Traditional rule-based approaches, while interpretable and easy to implement, are increasingly insufficient in modern environments where fraudsters adapt dynamically. Machine learning (ML) methods have thus gained prominence, as they can automatically learn patterns from large datasets and provide more robust detection mechanisms. This chapter reviews prior research on fraud detection, with an emphasis on machine learning techniques, challenges in data representation, and applications of tree-based ensemble methods such as XGBoost and LightGBM.

2.2 Traditional Approaches to Fraud Detection

Early fraud detection systems were primarily rule-based, relying on domain experts to specify suspicious patterns (e.g., unusually large transactions or mismatched billing addresses). Bolton and Hand (2002) demonstrated the use of statistical models and outlier detection techniques to capture abnormal behaviors in transaction streams. While effective in small-scale scenarios, these methods often suffer from high false positive rates and limited adaptability, since new fraud strategies can bypass predefined rules.

2.3 Machine Learning for Fraud Detection

With the growth of digital payments and large-scale transaction datasets, machine learning approaches have become dominant in fraud detection research. Ngai et al. (2011) provided an early systematic review, categorizing machine learning techniques into supervised, unsupervised, and hybrid methods. Supervised learning, in particular, has been widely used when labeled data is available, as it can exploit historical fraud records to train predictive models.

Among supervised models, tree-based ensemble methods such as Random Forests, Gradient Boosted Decision Trees (GBDT), and their optimized variants (XGBoost, LightGBM) have consistently achieved strong performance. Carcillo et al. (2018) highlighted their effectiveness in handling high-dimensional data, nonlinear feature interactions, and imbalanced distributions—characteristics that align closely with fraud detection challenges.

2.4 Imbalanced Learning in Fraud Detection

One of the most significant issues in fraud detection is class imbalance: fraudulent transactions typically represent less than 1% of all records. Dal Pozzolo et al. (2015) emphasized the importance of resampling methods (e.g., SMOTE, undersampling) and cost-sensitive learning to mitigate imbalance effects. More recent studies (e.g., Bahnsen et al., 2016) have shown that ensemble methods combined with imbalance-aware strategies yield higher detection rates without sacrificing precision.

2.5 Feature Engineering and Representation Learning

Feature engineering plays a critical role in fraud detection. Researchers have developed methods to extract temporal patterns, user behavior profiles, and network-based relationships. Fu et al. (2016) introduced deep feature representation learning, showing how neural networks can automatically learn high-level abstractions of transaction data. However, interpretability remains a limitation for deep learning, making tree-based models more attractive for real-world deployment.

The IEEE-CIS dataset, in particular, includes anonymized features such as transaction amounts, device information, and engineered Vesta variables. Chen et al. (2020) emphasized the need for careful preprocessing, including missing value treatment, scaling of time-dependent features, and encoding of categorical variables. This aligns with the methodological challenges addressed in the present thesis.

2.6 XGBoost and LightGBM in Fraud Detection

Recent studies highlight XGBoost and LightGBM as state-of-the-art methods for structured fraud detection tasks. XGBoost, introduced by Chen and Guestrin (2016), optimizes gradient boosting with regularization and efficient parallelization, while LightGBM (Ke et al., 2017) employs histogram-based algorithms and leaf-wise tree growth for faster training and improved scalability.

Wang et al. (2019) demonstrated the superior performance of LightGBM in fraud detection, particularly in large datasets with many categorical variables. Similarly, Liu et al. (2020) compared XGBoost and LightGBM across various domains, concluding that both achieve high accuracy but LightGBM provides computational advantages. These findings reinforce the motivation for their selection in this thesis.

2.7 Summary and Research Gap

The reviewed literature demonstrates that machine learning methods—particularly tree-based ensembles—are highly effective in fraud detection. Traditional approaches, while interpretable, are insufficient for large-scale, evolving fraud patterns. Supervised ML methods such as XGBoost and LightGBM address key challenges, including high dimensionality and class imbalance, while maintaining interpretability through feature importance analysis.

However, gaps remain. Few studies have comprehensively analyzed the IEEE-CIS dataset using both XGBoost and LightGBM with systematic preprocessing, feature engineering, and evaluation under realistic conditions. This thesis addresses this gap by conducting a comparative study of these two models, providing insights into their performance, feature contributions, and practical applicability to real-world fraud detection systems.