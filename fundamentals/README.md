# MLOps Fundamentals - Interview Questions

## Q1: Walk me through the end-to-end ML lifecycle as you've implemented it in production.
**Context:** We had 25+ models with fragmented ownership.
**Answer:** The lifecycle: Data Ingestion (Kafka/S3) -> Data Validation (Great Expectations) -> Feature Eng (Spark/Flink -> Feast) -> Experimentation (Jupyter/MLflow) -> Training Pipeline (Airflow -> K8s) -> Validation Gate -> Model Registry -> CI/CD -> Deployment (KServe) -> Monitoring (Datadog/Evidently). 
**Key Failures:** Upstream schema changes breaking training, training-serving skew, skipped validation gates.

## Q2: Explain the difference between a feature store's offline and online layer. 
**Context:** Feature definitions drifted between Pandas (training) and Java (serving).
**Answer:** Offline store (S3/BigQuery) serves historical data for training with *point-in-time correctness* (no data leakage). Online store (Redis/DynamoDB) serves low-latency (<5ms) feature vectors for real-time inference. Tools like Feast manage both from a single definition.

## Q3: How do you detect and handle data drift in production?
**Context:** A credit model failed when Q4 brought a new demographic.
**Answer:** Monitor at three levels: 1. Feature Drift (Input) using PSI (Population Stability Index). Alert if PSI > 0.2. 2. Prediction Drift (Output) measuring shift in prediction distribution. 3. Concept Drift (Outcome) when ground truth labels arrive. Use Airflow DAGs to compute PSI daily and push to Datadog.

## Q4: What's training-serving skew, and how have you dealt with it?
**Context:** 0.94 offline AUC, but worse than random in production.
**Answer:** Skew occurs when features are computed differently in training vs serving (e.g., sliding window logic differences). 
**Fix:** 1. Use a Feature Store to unify logic. 2. Log serving-time features, recompute them offline asynchronously, and compare for exact matches. 

## Q5: Batch inference vs real-time inference - how do you decide?
**Context:** A team wanted real-time churn predictions for a weekly email.
**Answer:** Does the business need < 1 sec latency? Yes -> Real-time (K8s/Redis). If < 1 hour -> Micro-batch (Spark Streaming). Otherwise -> Batch (Airflow). Batch is cheaper, runs at 100% GPU utilization, and handles failures via retries. Real-time requires 24/7 compute and complex HPA.

## Q6: How do you design an effective Model Registry workflow?
**Context:** A junior dev deployed an untested model.
**Answer:** Registry is a state machine. Training registers as `None`. CI pipeline tests against a golden dataset, promotes to `Staging`. A human reviews metrics and promotes to `Production`. The CD pipeline *only* listens to this transition to deploy. Humans have no raw K8s access.

## Q7: Concept Drift vs Data Drift. What is the difference and how do you fix each?
**Context:** Fraud patterns changed during COVID (Concept). The upstream data team changed a column format (Data).
**Answer:** Data Drift: The input feature distribution changes (e.g., users are suddenly older). Fix: Retrain the model on new data. Concept Drift: The relationship between input and target changes (e.g., buying 10 masks used to mean hospital, now means normal user). Fix: Retrain, but weight recent data heavily or change model architecture.

## Q8: What is Target Leakage and how do you prevent it in training pipelines?
**Context:** A model had 99% accuracy offline, 50% online.
**Answer:** Target leakage is including data from the future in your training set. (e.g., using `cancellation_fee` to predict `will_churn`). 
**Fix:** Implement Point-in-Time Correctness (PITC) in the feature store. Drop features heavily correlated with the target variable during EDA.

## Q9: How do you handle severe class imbalance in production fraud models?
**Context:** 99.9% of transactions are legitimate.
**Answer:** 1. Evaluation: Never use Accuracy. Use PR-AUC (Precision-Recall AUC). 2. Training: Use SMOTE for minority oversampling, or scale_pos_weight in XGBoost. 3. Inference: Adjust the probability threshold based on business cost (Cost of False Positive vs False Negative) rather than defaulting to 0.5.

## Q10: What are degenerate feedback loops in ML systems?
**Context:** A recommender system only showed popular items, making them more popular.
**Answer:** When a model's outputs influence its future training data. A recommendation engine biases user clicks, so next month's training data only contains what the model recommended. 
**Fix:** "Epsilon-Greedy" exploration. Dedicate 5% of traffic to random/untested recommendations to gather unbiased training labels.

## Q11: How do you validate a time-series model? Why is standard cross-validation bad?
**Context:** Standard k-fold CV leaked future data into the training set for a stock model.
**Answer:** Standard CV randomly shuffles data. In time-series, predicting yesterday using tomorrow's data is target leakage. 
**Fix:** Time-Series Split (Walk-forward optimization). Train on Jan-Mar, test on Apr. Train on Jan-Apr, test on May.

## Q12: How do you handle missing data (nulls) safely in production pipelines?
**Context:** A new mobile app version stopped sending the `device_type` field.
**Answer:** 1. Strict validation (Great Expectations) to catch nulls early. 2. Inference fallback: If a critical feature is null, circuit break and return a safe default. 3. Imputation: Use median/mode from the feature store, but ensure the imputation logic is identical in training and serving.

## Q13: What is Model Calibration and why does it matter for business rules?
**Context:** Business rules said "Review if probability > 80%". The new model output 80% but was actually only correct 50% of the time.
**Answer:** Uncalibrated models (like Random Forests or Neural Nets) output scores between 0-1, but they aren't true probabilities. 
**Fix:** Use Platt Scaling or Isotonic Regression post-training to align model scores with actual observed frequencies, so a 0.8 score means an 80% chance of the event.

## Q14: Online vs Offline Evaluation Metrics. Why do they disagree?
**Context:** F1 score improved by 10%, but Click-Through Rate (CTR) dropped.
**Answer:** Offline metrics (F1, RMSE) measure predictive accuracy on historical data. Online metrics (CTR, Revenue) measure business impact in the real world. They disagree because historical data doesn't capture user behavioral shifts. Always trust online A/B tests over offline metrics.

## Q15: A/B Testing vs Interleaving for recommender systems.
**Context:** A/B tests were taking 4 weeks to reach statistical significance.
**Answer:** A/B testing splits users. Interleaving shows one user recommendations from both Model A and Model B in the same list, alternating them. Interleaving removes user-level bias and reaches statistical significance 10x faster, but requires complex frontend tracking.

## Q16: How do you handle categorical variables with high cardinality in production?
**Context:** User IDs or Zip Codes taking up massive memory in one-hot encoding.
**Answer:** Never use one-hot encoding for cardinality > 100. 
**Fix:** 1. Target Encoding (mean encoding), but requires strict smoothing to prevent leakage. 2. Entity Embeddings (training a neural net layer to learn a dense vector representation of the category). 3. Hashing trick (fastest for real-time serving).

## Q17: What is the curse of dimensionality and how does it impact MLOps?
**Context:** A team added 5,000 new sparse features. Inference latency spiked and accuracy dropped.
**Answer:** As features increase, the data space grows exponentially, making the data sparse. 
**Fix:** 1. PCA/UMAP for dimensionality reduction. 2. L1 Regularization (Lasso) to force irrelevant feature weights to zero. 3. Feature importance pruning using SHAP values.

## Q18: Cold Start Problem in ML. How do you serve predictions for a brand new user?
**Context:** A new user signs up, the model has no history, and recommends irrelevant items.
**Answer:** 1. Fallback rules: Serve the "Global Top 10 Trending" items. 2. Onboarding questionnaire to gather explicit signals. 3. Content-based filtering (matching based on item metadata rather than user history) until enough behavior is logged.

## Q19: Precision vs Recall. How do you explain the trade-off to a Product Manager?
**Context:** PM wants to catch 100% of fraud without blocking any real users.
**Answer:** Precision: "When the model fires, how often is it right?" Recall: "Out of all the bad guys, how many did we catch?" 
**Explain:** "If we want to catch every fraudster (100% recall), we have to cast a wide net, which means blocking some good users (lower precision). We have to set a threshold based on the cost of a false positive vs false negative."

## Q20: What are Data Version Control (DVC) best practices?
**Context:** Data scientists were saving data as `train_final_v3_really_final.csv`.
**Answer:** DVC tracks large files using git-like semantics. 
**Best Practice:** Store large artifacts in S3. DVC generates a `.dvc` hash file. Commit the `.dvc` file to Git alongside the training code. This links the exact code commit to the exact dataset byte-hash.

## Q21: Model Explainability in Production (SHAP/LIME). When and how?
**Context:** A loan applicant sued for discrimination. We needed to explain the model.
**Answer:** SHAP calculates the exact contribution of each feature to the final prediction. 
**Implementation:** Running SHAP is too slow for real-time inference (adds hundreds of ms). We compute SHAP values asynchronously in a background queue after the prediction is made, and store them in PostgreSQL for compliance audits.

## Q22: What is the Bias-Variance tradeoff in the context of Deep Learning?
**Context:** A massive neural net perfectly memorized the training data but failed in production.
**Answer:** High Bias = Underfitting. High Variance = Overfitting. Deep Learning models naturally have incredibly high variance because they have millions of parameters. 
**Fix:** Early stopping, Dropout layers, Weight Decay (L2 regularization), and acquiring more training data.

## Q23: Propensity Modeling in MLOps. What is it?
**Context:** Marketing wanted to send 10% discount codes to users likely to churn.
**Answer:** Propensity models predict the likelihood of an action. 
**The trap:** Don't send discounts to people who were going to stay anyway (wasted money), and don't send to people who will definitely churn regardless. Use Uplift Modeling to target the "Persuadables" (users who will only stay *if* they get the discount).

## Q24: How do you optimize Hyperparameter Tuning (HPO) to save compute costs?
**Context:** A grid search on an XGBoost model ran for 5 days on an A100.
**Answer:** 1. Abandon Grid Search. Use Bayesian Optimization (Optuna/Hyperopt) which learns from previous trials to narrow the search space. 2. Use Hyperband / Successive Halving: Start 100 trials, but aggressively terminate the worst-performing 50% after a few epochs, saving massive compute.

## Q25: Challenges of Transfer Learning in production.
**Context:** Fine-tuning BERT for domain-specific text classification.
**Answer:** Catastrophic Forgetting. As the model learns your specific domain, it abruptly forgets the general language representation it learned during pre-training. 
**Fix:** Use a very small learning rate (e.g., 2e-5), freeze the base layers, and only train the top classification head first, then unfreeze and fine-tune end-to-end.

## Q26: Online vs Offline metrics for RAG (Retrieval-Augmented Generation).
**Context:** Wait, skipping RAG as per rules (No LLMOps/GenAI). Let's replace with: What is a Shadow Deployment?
**Answer:** Deploying a new model alongside the old one. The new model receives production traffic, makes predictions, and logs them to a database, but the *user never sees the output*. 
**Why:** Safest way to measure latency, CPU usage, and output distributions on real data without risking user experience.

## Q27: How do you handle unstructured data (images/text) in a Feature Store?
**Context:** Trying to put 2MB images into Redis.
**Answer:** Feature stores are for structured, low-latency vectors. Do not put raw images in Redis. 
**Fix:** Use a deep learning model to extract a dense vector embedding (e.g., a 512-float array) from the image offline. Store the *embedding* in the feature store or a Vector DB.

## Q28: Federated Learning concepts.
**Context:** Mobile team wanted to train a keyboard predictor without sending user texts to the cloud.
**Answer:** Instead of moving data to the model, move the model to the data. Send the base model to the user's phone. Train locally on the phone's data. Send only the *gradient updates* back to the cloud. Aggregate gradients (FedAvg) and update the global model.

## Q29: What is SMOTE and why shouldn't you use it blindly?
**Context:** A data scientist used SMOTE before cross-validation, leading to data leakage.
**Answer:** Synthetic Minority Over-sampling Technique. It generates fake minority class examples. 
**Mistake:** If you SMOTE the entire dataset and *then* split into train/test, synthetic examples in the test set bleed information from the training set. Always split train/test *first*, then apply SMOTE only to the training set.

## Q30: How do you implement robust retries for ML API calls?
**Context:** Our ML microservice timed out, causing the upstream web app to crash.
**Answer:** ML endpoints have high tail latency. 
**Fix:** Use Exponential Backoff with Jitter. If request fails, wait 1s, then 2s, then 4s, but add random jitter (+/- 500ms) so thousands of failing clients don't retry at the exact same millisecond and accidentally DDoS the inference server.
