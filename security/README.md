# Security & Compliance for ML - Interview Questions

## Q1: How do you manage secrets (API keys, DB credentials) in your ML training pipelines?

**Real-world context:**
A data scientist accidentally committed an AWS access key to a GitHub repository while trying to download a dataset for a Jupyter notebook. A bot scraped it, and we had a $15,000 EC2 bill for crypto mining by the time we caught it.

**Answer:**

Managing secrets in ML is harder than standard software because data scientists often run ad-hoc scripts locally, in notebooks, and in distributed K8s clusters.

**My Architecture (Vault + External Secrets Operator):**

We absolutely forbid `.env` files or hardcoded credentials anywhere in the code. We use HashiCorp Vault as the single source of truth.

1.  **For local/notebook dev:** Data scientists authenticate to Vault via OIDC (using their corporate Okta login). A short-lived token is issued. A Python helper library intercepts calls to `get_secret("DB_PASSWORD")` and fetches it from Vault dynamically into memory. Nothing is ever written to disk.
2.  **For Kubernetes Training Jobs:** We use the **External Secrets Operator (ESO)** on K8s.
    *   The data scientist defines an `ExternalSecret` custom resource in their Helm chart.
    *   ESO connects to Vault (using a K8s Service Account bound to an IAM role), retrieves the secret, and injects it as a native Kubernetes `Secret`.
    *   The training pod mounts this native K8s Secret as an environment variable or volume mount.
3.  **For CI/CD (Jenkins/GitHub Actions):** We use the Vault plugin for Jenkins or OIDC federation for GitHub Actions. The pipeline requests temporary, scoped credentials to pull the Docker image or deploy the model, valid for only 15 minutes.

**Common Mistakes:**
*   Logging secrets in MLflow. I've seen `mlflow.log_params(config)` where the config dict contained a database password. I implemented a regex-based secret scrubber in our logging wrapper to prevent this.
*   Baking secrets into the Docker image during the Jenkins build phase.

---

## Q2: How do you handle Data Privacy and PII in ML feature stores and model training?

**Real-world context:**
We were building a churn prediction model. The raw data contained customer names, emails, and plain-text phone numbers. If a model inadvertently memorized this data, or if a data scientist dumped a sample to a CSV, we would have a massive GDPR violation.

**Answer:**

My approach to PII (Personally Identifiable Information) in MLOps is defense-in-depth: Data Masking at rest, RBAC, and Audit Logging.

**1. Data Masking (The ELT Phase):**
PII should never enter the Feature Store in plain text.
*   In our Spark data pipelines (before data hits Feast/Redis), we apply hashing. If a model needs a "user ID" to join features, we use a salted SHA-256 hash of the ID. The model doesn't care if the ID is "john.doe" or "a7x9b...", it just needs a unique identifier.
*   For categorical features (e.g., Zip Code), we generalize them (e.g., truncate to the first 3 digits to represent a region rather than a specific neighborhood) to prevent re-identification.

**2. RBAC (Role-Based Access Control):**
*   Data scientists do not get direct access to production databases.
*   They get access to an anonymized "Data Lakehouse" view.
*   In Feast, we use project-level isolation.

**3. Model Artifact Security:**
*   A trained model artifact (the `.pkl` or `.pt` file) can sometimes memorize training data (especially true in deep learning/NLP, though less so in XGBoost).
*   We treat the model artifact itself as sensitive data. It is stored in an S3 bucket encrypted with AWS KMS customer-managed keys. The inference pods must have the specific IAM role to decrypt the model at startup.

**The GDPR "Right to be Forgotten" Challenge:**
If a user requests their data be deleted, we must delete it from the raw databases. But what about the model trained on their data? Technically, you are supposed to retrain the model without their data. In practice, retraining daily/weekly on a rolling window of data naturally phases out deleted users.

---

## Q3: How do you secure your ML CI/CD pipelines against supply chain attacks?

**Real-world context:**
The PyTorch dependency confusion attack happened, where a malicious package was uploaded to public PyPI with the same name as an internal private package. A naive `pip install` pulled the malicious package.

**Answer:**

ML pipelines are incredibly vulnerable to supply chain attacks because `pip install -r requirements.txt` routinely pulls hundreds of transitive dependencies from the internet.

**My Pipeline Security Strategy:**

1.  **Private Package Repository (Artifactory/Nexus):**
    *   Our Jenkins build servers *never* connect directly to public PyPI or DockerHub.
    *   They connect to a private JFrog Artifactory. Artifactory acts as a proxy, caching approved packages. If a package isn't in Artifactory, the build fails.
2.  **Dependency Pinning and Hashing:**
    *   We forbid loose requirements (e.g., `pandas>=1.0`). We require `poetry.lock` or `pip-compile` generated files that contain the exact version AND the cryptographic hash of every package.
    *   If the hash of the package on PyPI changes, Jenkins will refuse to install it.
3.  **Container Scanning (Trivy):**
    *   In the Jenkins pipeline, after building the ML model serving Docker image, we run Trivy.
    *   If Trivy detects any CRITICAL or HIGH CVEs in the base OS packages or Python libraries, the pipeline fails and the deployment is blocked.
    *   We use minimal base images (like distroless or Alpine, though Alpine is notoriously tricky with Python C-extensions; we usually stick to `python:3.9-slim`).
4.  **Least Privilege CI/CD:**
    *   The Jenkins worker node running the training job does not have permissions to deploy to the production K8s cluster. It only has permissions to write to the MLflow Model Registry and S3 artifact bucket. A separate CD pipeline (ArgoCD) handles the actual K8s deployment.

---
*[Back to README](../README.md)*
