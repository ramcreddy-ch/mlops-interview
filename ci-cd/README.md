# CI/CD for ML Systems - Interview Questions

## Q1: Explain the difference between Software CI/CD and ML CI/CD.
**Context:** Team was training models inside Jenkins PR builders.
**Answer:** Software CI/CD builds binaries. ML CI/CD has three triggers: Code (Git PR), Data (Drift alert), and Model (Registry promotion). You cannot train a model on every PR (takes too long). Separate CI (code testing) from CT (Continuous Training - triggered by Airflow/Kubeflow).

## Q2: How do you handle model versioning and rollback strategies?
**Context:** Bad model deployed, took 45 minutes to rollback.
**Answer:** Version Code (Git), Data (DVC), and Model (MLflow run ID). Use ArgoCD for GitOps deployment. The K8s manifest pointing to `s3://model/v4` lives in Git. To rollback, you simply `git revert` the manifest repo. ArgoCD instantly syncs K8s to pull the older model.

## Q3: How do you handle large model (4GB+) timeouts during deployment?
**Context:** Jenkins pipelines timing out waiting for K8s pods to become healthy.
**Answer:** Decouple CD from K8s status. Jenkins should only update the GitOps repo and exit immediately. ArgoCD handles the async pulling and syncing. Ensure K8s readiness probes have a high `initialDelaySeconds` (e.g., 120s) so K8s doesn't kill the pod while loading weights into VRAM.

## Q4: How do you implement automated A/B testing in your CD pipeline?
**Context:** Need to test models safely in production.
**Answer:** CD pipeline updates Istio/KServe routing rules. Deploy Model B alongside Model A. CD script sets Istio to route 10% traffic to B. Wait for metrics collection. If CTR improves, CD script automatically updates Istio to 100%.

## Q5: How do you test ML code in CI without running a full training job?
**Context:** CI pipeline took 8 hours to run on every commit.
**Answer:** Use a "dummy" dataset (100 rows). CI runs the training code on the dummy data just to verify syntax, shape compatibility, and that the loss function doesn't throw a `NaN`. Full training only happens in the CT pipeline.

## Q6: How do you manage infrastructure as code (IaC) alongside ML code?
**Context:** K8s manifests drifting from actual deployed state.
**Answer:** Keep Terraform/K8s manifests in a separate Git repository from the ML training code. This enforces separation of concerns. Data scientists PR the model code; Platform engineers PR the IaC.

## Q7: What are the best practices for using GitHub Actions for MLOps?
**Context:** Moving away from Jenkins.
**Answer:** 1. Use OIDC instead of long-lived AWS/GCP secrets. 2. Use reusable workflows (e.g., `ml-training-workflow.yml`) so multiple teams share the same CI/CD standards. 3. Use self-hosted runners for heavy jobs to avoid GitHub's per-minute billing.

## Q8: How do you implement data quality gates in CI/CD?
**Context:** Corrupt data broke the nightly retraining job.
**Answer:** Before the training step begins, the CT pipeline runs a Great Expectations suite against the S3 dataset. If the schema changed or nulls exceed 5%, the pipeline fails and halts *before* wasting GPU hours on training.

## Q9: How do you handle database migrations linked to ML model deployments?
**Context:** Model required a new feature, which required a new DB column.
**Answer:** Decouple them. Phase 1: Deploy DB schema migration (adds column). Phase 2: Deploy data pipeline (populates column). Phase 3: Deploy ML model (uses column). Never bundle schema migrations and ML deployments in the same CD step.

## Q10: How do you ensure reproducibility in your CI/CD pipelines?
**Context:** Model trained in CI had different accuracy than model trained locally.
**Answer:** Dockerize the training environment. The exact `Dockerfile` (with pinned `poetry.lock` and CUDA versions) must be used locally and in the CI pipeline. Use MLflow to track the Git commit hash of the code that triggered the run.

## Q11: What is a Shadow Deployment in the context of CD?
**Context:** Evaluating a high-risk medical model.
**Answer:** CD pipeline deploys the model, but sets Istio to "mirror" traffic. The model receives 100% of prod requests, makes predictions, and logs them to Kafka, but the responses are discarded. Allows safe validation of latency and accuracy.

## Q12: How do you test inference endpoints in CI?
**Context:** Model deployed but failed due to JSON serialization errors.
**Answer:** CI pipeline builds the inference Docker image, spins it up using `docker-compose` or Minikube, and fires synthetic payload tests (`pytest`) against the REST API to ensure input/output schemas match expectations.

## Q13: How do you manage multi-environment progression?
**Context:** Pushing from Dev -> Staging -> Prod.
**Answer:** Model artifact is built *once* in Dev. The exact same Docker image/S3 artifact is promoted through environments. Environment-specific configs (DB URLs) are injected via K8s ConfigMaps. Never rebuild the image for Prod.

## Q14: How do you use Feature Flags in ML deployments?
**Context:** Need to instantly disable a new model feature.
**Answer:** Wrap the model invocation or specific preprocessing steps in a feature flag (LaunchDarkly/Split). If the model fails, toggle the flag to bypass it without needing a full K8s rollback.

## Q15: How do you handle CI/CD for edge devices (IoT/Mobile)?
**Context:** Deploying models to 10k cameras.
**Answer:** CI pipeline converts the model (ONNX/TensorRT). CD pipeline pushes the artifact to an IoT Device Management service (AWS IoT Greengrass), which handles OTA (Over-The-Air) updates to the fleet asynchronously.

## Q16: Explain "Continuous Training" (CT).
**Context:** Models were decaying in production.
**Answer:** An Airflow DAG scheduled weekly, or triggered by an Evidently AI drift alert. It pulls the latest data, runs the training code, evaluates against a holdout set, and if better, registers the new model version automatically.

## Q17: How do you track model lineage in CI/CD?
**Context:** Audit required knowing exactly how a model was built.
**Answer:** MLflow tracks the S3 URI of the dataset, the Git commit of the code, the hyperparameter dictionary, and the Docker image digest used for training. This creates an immutable chain of custody.

## Q18: Using Spinnaker vs ArgoCD for ML CD.
**Context:** Choosing a CD tool.
**Answer:** Spinnaker is powerful for complex, multi-cloud imperative pipelines (e.g., deploying to AWS and GCP simultaneously). ArgoCD is declarative GitOps, deeply integrated with Kubernetes. For K8s-native MLOps, ArgoCD is generally preferred.

## Q19: How do you optimize Docker image builds in CI?
**Context:** Jenkins taking 20 minutes to build PyTorch images.
**Answer:** Use Docker build caching (`--cache-from`). Separate `requirements.txt`/`poetry.lock` copy steps from the rest of the code. Put heavy base images (CUDA) in a lower layer so they are cached.

## Q20: How do you handle secrets in ML CI/CD?
**Context:** AWS keys hardcoded in Jenkinsfiles.
**Answer:** Use HashiCorp Vault. Jenkins authenticates to Vault via an AppRole, fetches a short-lived, dynamically generated AWS STS token valid for 15 minutes, deploys the model, and the token expires.

## Q21: Implementing Blue/Green Deployments.
**Context:** Zero-downtime ML updates.
**Answer:** Deploy Model V2 alongside Model V1. Both are fully scaled. Switch the K8s Service selector (or Ingress route) to point 100% of traffic to V2 instantly. Keep V1 running for 10 minutes as a fast fallback, then terminate V1.

## Q22: What is the role of DVC in CI/CD?
**Context:** Versioning datasets.
**Answer:** DVC creates a `.dvc` file containing the MD5 hash of the dataset in S3. You commit the `.dvc` file to Git. In the CI pipeline, `dvc pull` fetches the exact data version tied to that Git commit.

## Q23: How do you handle configuration drift?
**Context:** Someone manually edited a K8s deployment via `kubectl`.
**Answer:** ArgoCD detects that the live cluster state differs from the Git state. It marks the app as "OutOfSync" and can be configured to automatically overwrite the manual changes, enforcing Git as the source of truth.

## Q24: Automating hyperparameter tuning in CI/CD.
**Context:** Need to tune automatically on retrain.
**Answer:** The CT pipeline spawns an Optuna/Ray Tune job. It runs 50 trials in parallel on a K8s GPU cluster. The orchestrator tracks the best trial in MLflow and promotes that specific run to the Model Registry.

## Q25: How do you handle schema evolution in ML endpoints?
**Context:** V2 of the model requires a new input field.
**Answer:** Version the API route (`/v1/predict` vs `/v2/predict`). Deploy V2. Have clients migrate to `/v2`. Once V1 traffic hits zero, deprecate V1. Never introduce breaking schema changes to a live endpoint.

## Q26: Testing Data Pipelines in CI.
**Context:** Spark job logic was flawed.
**Answer:** Use Pytest with local Spark sessions. Create small, hardcoded Parquet files with known edge cases (nulls, extreme values). The CI pipeline runs the Spark transformations locally to assert the output matches expected results.

## Q27: How do you integrate model cards into CI/CD?
**Context:** Compliance team needed documentation for every deployment.
**Answer:** The CI pipeline automatically generates a markdown Model Card containing training metrics, dataset distributions, and bias test results, and attaches it as an artifact to the MLflow registry.

## Q28: Canary Analysis automation.
**Context:** Waiting 24 hours to manually check a canary is slow.
**Answer:** Use tools like Kayenta or Flagger. They automatically query Datadog metrics (latency, error rate) comparing the canary vs baseline. If metrics deviate significantly, it automatically triggers a rollback.

## Q29: Deploying multiple models to a single endpoint.
**Context:** Triton inference server managing 5 models.
**Answer:** The CD pipeline doesn't deploy a new container. It uploads the ONNX model to an S3 bucket and triggers a Triton API call to dynamically load the new model into GPU memory without restarting the pod.

## Q30: Security scanning in the ML CI pipeline.
**Context:** Preventing CVEs from reaching production.
**Answer:** In Jenkins, run Trivy or Clair against the built Docker image. If CRITICAL or HIGH vulnerabilities are found in the Python packages (e.g., a vulnerable version of `requests`), the build fails and blocks deployment.
