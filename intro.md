Chapter 1 – Introduction
1.1 Background and Motivation

In recent years, the rapid expansion of e-commerce and digital payment systems has transformed the way financial transactions are conducted. While these innovations offer convenience and efficiency, they have also created new opportunities for fraudulent activities. Fraudulent transactions can cause substantial financial losses for both businesses and consumers, and they undermine trust in online payment ecosystems. According to industry reports, global losses due to payment fraud continue to increase annually, despite significant investments in prevention systems.

Detecting fraudulent activities in real-time is a particularly challenging task. Fraudsters frequently adapt their techniques to bypass existing detection mechanisms, making static, rule-based systems increasingly ineffective. As a result, there is a growing demand for intelligent, data-driven solutions that can learn from past behavior and generalize to unseen fraudulent patterns. Machine learning approaches have emerged as a promising alternative, capable of capturing complex relationships between transaction features and identifying anomalies that may indicate fraud.

1.2 Problem Statement

Fraud detection presents unique difficulties that distinguish it from many other machine learning applications. First, fraudulent transactions are relatively rare compared to legitimate ones, creating a highly imbalanced dataset. This imbalance makes it difficult to train models that do not simply default to predicting all transactions as legitimate. Second, the available data often contains high levels of missingness, anonymized fields, and a mixture of numerical and categorical variables. Third, fraudsters continuously evolve their tactics, requiring models that are robust and adaptable.

Traditional detection approaches, such as rule-based systems or simple statistical models, are unable to capture the subtle patterns of fraudulent behavior in large-scale, high-dimensional datasets. This gap highlights the need for more advanced techniques, such as gradient boosting models, which are well-suited to handling structured data with many heterogeneous features.

1.3 Research Objectives

The primary objective of this thesis is to evaluate the effectiveness of machine learning models in detecting fraudulent transactions using the IEEE-CIS fraud detection dataset. This dataset, released as part of a Kaggle competition, provides a large and complex collection of anonymized transaction and identity information, making it an ideal benchmark for studying fraud detection techniques.

Specifically, the thesis aims to:

Conduct exploratory data analysis (EDA) to understand the characteristics and challenges of the dataset.

Apply preprocessing and feature engineering techniques to handle missing values, categorical encoding, and temporal features.

Train and evaluate two advanced machine learning models—XGBoost and LightGBM—for fraud detection.

Compare model performance using key evaluation metrics, such as ROC AUC, Precision, Recall, F1-score, and Precision-Recall AUC.

Analyze feature importance to identify which variables most strongly contribute to fraud detection.

Through these objectives, the study seeks to contribute insights into the practical application of machine learning models in fraud detection tasks.

1.4 Research Questions

To achieve the stated objectives, the thesis will address the following research questions:

What preprocessing and feature engineering strategies are most effective for handling missing and high-dimensional data in fraud detection?

How do XGBoost and LightGBM compare in terms of predictive performance on the IEEE-CIS dataset?

Which features provide the most useful signals for distinguishing fraudulent from legitimate transactions?

What are the trade-offs between model interpretability and accuracy in the context of fraud detection?

1.5 Structure of the Thesis

The remainder of this thesis is organized as follows:

Chapter 2: Literature Review – Discusses previous research on fraud detection, including traditional approaches, statistical methods, and machine learning-based solutions. It also reviews related studies using the IEEE-CIS dataset and similar benchmarks.

Chapter 3: Data and Methodology – Presents the dataset in detail, describes preprocessing steps, and outlines the machine learning models and evaluation metrics used.

Chapter 4: Results and Analysis – Provides experimental results, compares model performance, and presents feature importance analyses.

Chapter 5: Discussion – Interprets the findings, considers their implications for real-world fraud detection systems, and reflects on limitations of the study.

Chapter 6: Conclusion and Future Work – Summarizes the key contributions, highlights the significance of the results, and proposes directions for future research.