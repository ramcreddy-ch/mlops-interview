# CI/CD for ML Systems - Interview Questions

## Q1: Explain your ML CI/CD pipeline. How is it different from a traditional software CI/CD pipeline?

**Real-world context:**
When I joined, the team was using Jenkins to trigger ML training, but they were treating models like software binaries. Every commit triggered a 4-hour training job, backing up the Jenkins queue for days.

**Answer:**

**The Core Difference:**
Traditional software CI/CD is triggered by one thing: **Code Changes**.
ML CI/CD is triggered by three things: **Code Changes, Data Changes, and Model Degradation**.

Furthermore, compiling software takes minutes; training a model takes hours or days. You cannot rebuild the "artifact" (the model) on every PR.

**My Production ML CI/CD Architecture (using Jenkins & GitHub):**

**1. Continuous Integration (CI) - Triggered on PR to `main`:**
*   **What it does:** Lints Python code, runs unit tests on data transformation logic (e.g., testing that the feature engineering function handles nulls correctly).
*   **Crucially:** It does *not* train the full model. It might run a "dummy" training job on 100 rows of data just to ensure the ML framework code compiles and runs without syntax errors.
*   **Output:** Validated code merged to `main`.

**2. Continuous Training (CT) - Triggered by Schedule (Airflow) or Data Drift Alert:**
*   **What it does:** This is the heavy lifting. It pulls the latest code from `main`, pulls the latest data from the feature store, provisions a GPU node, and runs the training job.
*   **Validation:** Once trained, the model is evaluated against a holdout dataset. If it beats the current production model (Champion vs. Challenger), it is registered in MLflow and marked as "Staging".
*   **Output:** A new model artifact in the Model Registry.

**3. Continuous Deployment (CD) - Triggered by Model Registry Stage Change:**
*   **What it does:** When a model is approved in MLflow, Jenkins detects this event. It packages the model artifact, the inference code, and dependencies into a Docker image.
*   **Deployment:** It updates the Kubernetes manifests (via GitOps/ArgoCD) to point to the new image tag. The model is deployed to K8s using a Canary strategy.
*   **Output:** Model serving live traffic.

**Trade-offs:**
Separating CI (code) from CT (training) requires a robust orchestrator (like Airflow or Kubeflow) to manage the training state separately from Jenkins. The trade-off is slightly more complex architecture, but it's the only way to scale without wasting thousands of dollars on unnecessary GPU training runs.

---

## Q2: How do you handle model versioning and rollback strategies in production?

**Real-world context:**
A bad model passed offline validation but started generating garbage predictions in production, costing us revenue. The team panicked because rolling back meant manually finding an old S3 path and restarting pods. It took 45 minutes to restore service.

**Answer:**

**Versioning Strategy:**
We version three things tightly coupled together: Code, Data, and Model.
1.  **Code:** Git commit hash.
2.  **Data:** DVC (Data Version Control) hash or an S3 prefix with a date partition.
3.  **Model:** MLflow run ID and version number.

In MLflow, every model artifact is tagged with the Git commit hash and the data S3 path used to generate it. This ensures 100% reproducibility.

**Deployment & Rollback Strategy (GitOps with ArgoCD):**

We use **ArgoCD** to manage our Kubernetes deployments. The state of our cluster is defined entirely in a Git repository (separate from our application code).

1.  **The Manifest:** Our KServe `InferenceService` YAML file lives in Git. It points to a specific model version (e.g., `s3://models/fraud/v4.tar.gz`).
2.  **The Rollout:** When CD updates the model, it makes a commit to this GitOps repo: `Update fraud model to v5`. ArgoCD detects the change and applies it to K8s.
3.  **The Rollback (The Fix):** If v5 is bad, rolling back is literally a `git revert` command on the GitOps repository. ArgoCD sees the desired state is back to `v4`, and it instantly updates K8s to pull the older model. K8s handles the rolling update, ensuring zero downtime during the rollback.

**Common Mistakes:**
*   **Overwriting artifacts:** Never name a model `model_latest.pkl`. Always append the version or hash. If you overwrite, you cannot roll back.
*   **Ignoring dependency versions:** If model v4 used `scikit-learn==1.0` and v5 uses `1.2`, rolling back just the pickle file will crash the inference server. The Docker image must be versioned and rolled back entirely.

---

## Q3: Describe a time your CI/CD pipeline failed during a deployment. What went wrong and how did you fix it?

**Real-world context:**
During a deployment of a large NLP model to our K8s cluster, the Jenkins pipeline timed out and failed, leaving the cluster in a degraded state.

**Answer:**

**Root cause:**
The new model artifact was 4GB. The Jenkins pipeline was configured to build the Docker image, push it to ECR, and then run `kubectl rollout status` to wait for the new pods to become healthy.
However, pulling a 4GB image onto the K8s nodes took 5 minutes. Loading the model into GPU memory took another 2 minutes. The Jenkins deployment step had a hardcoded timeout of 5 minutes. Jenkins marked the build as failed and aborted, while K8s was still slowly trying to start the pods.

**Debugging steps:**
1. Looked at Jenkins logs: `Timeout waiting for deployment to complete`.
2. Checked K8s: `kubectl get pods` showed pods in `ContainerCreating` state.
3. `kubectl describe pod` showed `Pulling image...`.

**The Fix (Immediate):**
Manually increased the timeout in the Jenkinsfile to 15 minutes to allow the current deployment to finish.

**Prevention strategy (Long-term):**
1.  **Decouple CD from K8s status:** Migrated from Jenkins pushing to K8s to ArgoCD pulling from Git. Jenkins just updates the Git repo and finishes immediately (takes 5 seconds). ArgoCD handles the asynchronous waiting and syncing of the K8s state.
2.  **Image Caching:** Implemented K8s node image caching (e.g., using a daemonset to pre-pull large base images) to speed up pod startup times.
3.  **Liveness/Readiness Probes:** Ensured the readiness probe had a high `initialDelaySeconds` (e.g., 120s) so K8s wouldn't kill the pod while it was slowly loading the 4GB model into RAM.

---
*[Back to README](../README.md)*
