# Security & Compliance for ML - Interview Questions

## Q1: How do you manage secrets in your ML training pipelines?
**Context:** AWS key leaked in a Jupyter notebook.
**Answer:** Never use `.env` files. Use HashiCorp Vault. Data scientists authenticate via Okta (OIDC) to get a short-lived token. In K8s, use External Secrets Operator to inject Vault secrets directly into pod memory.

## Q2: How do you handle Data Privacy and PII in feature stores?
**Context:** Churn model memorized user phone numbers.
**Answer:** 1. Data Masking (hash PII using salted SHA-256 before it hits Feast). 2. Generalization (truncate Zip Codes to 3 digits). 3. RBAC (Data scientists get anonymized views, not raw DB access).

## Q3: How do you secure CI/CD pipelines against supply chain attacks?
**Context:** PyTorch dependency confusion attack.
**Answer:** Never connect Jenkins directly to public PyPI. Use JFrog Artifactory as a caching proxy. Enforce `poetry.lock` with cryptographic hashes. Run Trivy container scanning before pushing ML images.

## Q4: What is a Model Inversion Attack and how do you prevent it?
**Context:** Competitor reconstructed training data from the API.
**Answer:** Attackers query a model repeatedly to map the decision boundary and extract sensitive training data (e.g., facial recognition). Prevent by rate-limiting APIs, adding noise to confidence scores (Differential Privacy), and restricting API access.

## Q5: What is an Adversarial Attack and how do you defend against it?
**Context:** Fraudsters altered 1 pixel to bypass an image filter.
**Answer:** Attackers add imperceptible noise to inputs to force a misclassification. Defend via Adversarial Training (train the model on adversarial examples), input sanitization, and ensemble models.

## Q6: Why is `pickle` a security risk?
**Context:** Deserializing untrusted `.pkl` files.
**Answer:** `pickle` allows arbitrary code execution via the `__reduce__` method. If an attacker uploads a malicious `.pkl`, loading it executes a shell script on the server. Always use ONNX or Safetensors in production.

## Q7: How do you implement RBAC for Model Registries?
**Context:** Junior engineer deleted a production model.
**Answer:** Integrate MLflow/Vertex with corporate SSO (Okta/AD). Assign roles: `Data Scientist` (can read all, write to staging), `Lead DS` (can promote to production), `Platform Engineer` (admin). 

## Q8: How do you secure Jupyter Notebook environments in K8s?
**Context:** Notebooks had raw internet access.
**Answer:** Run JupyterHub. Enforce OIDC login. Apply K8s NetworkPolicies: block outbound internet access by default, only allow traffic to internal S3 buckets and vetted artifact repositories. Run pods as non-root.

## Q9: How do you handle GDPR "Right to be Forgotten" in ML?
**Context:** User requested deletion, but their data is in the model weights.
**Answer:** Hard deleting data from weights (Machine Unlearning) is an unsolved research problem. In practice, delete the raw data, and retrain the model frequently on a rolling window (e.g., last 90 days) so deleted data naturally phases out.

## Q10: Securing S3 buckets containing training data.
**Context:** S3 bucket accidentally made public.
**Answer:** Enable AWS Macie to scan for PII. Enforce KMS Customer Managed Keys (CMK) for encryption at rest. Apply Bucket Policies blocking `Principal: *`. Use VPC Endpoints so data is only accessible from within the AWS VPC.

## Q11: How do you secure ML endpoints exposed to the public internet?
**Context:** DDoS attack on an expensive GPU inference endpoint.
**Answer:** Never expose KServe/FastAPI directly. Put an API Gateway (AWS API Gateway or Kong) in front. Implement Web Application Firewall (WAF) to block SQLi/XSS, and aggressive IP-based rate limiting to prevent GPU exhaustion.

## Q12: What is Differential Privacy in ML?
**Context:** Training a medical model on patient data.
**Answer:** Adding statistical noise to the training process (e.g., DP-SGD) or the dataset so that the output model does not memorize any single individual's data, mathematically guaranteeing privacy.

## Q13: How do you perform vulnerability scanning on ML Docker images?
**Context:** Base Python images containing CVEs.
**Answer:** Integrate Trivy or Snyk into Jenkins/GitHub Actions. Scan the `ubuntu` base OS and the Python `requirements.txt`. If Critical/High CVEs are found (e.g., Log4j, outdated Requests library), fail the build.

## Q14: Managing SSH access to ML training nodes.
**Context:** Developers SSHing into production GPUs to debug.
**Answer:** Disable SSH. Use AWS Systems Manager (SSM) Session Manager or `kubectl exec`. This provides a secure, audited, and RBAC-controlled shell without opening port 22 or managing SSH keys.

## Q15: Securing internal ML APIs (Service-to-Service).
**Context:** Any microservice could call the fraud model.
**Answer:** Implement Istio Service Mesh. Enforce mTLS (Mutual TLS) between all pods. Configure authorization policies so only the `payment-gateway` pod is cryptographically allowed to call the `fraud-model` pod.

## Q16: Auditing and logging for compliance (SOC2/HIPAA).
**Context:** Auditors need proof of model changes.
**Answer:** Centralize logs. AWS CloudTrail logs all infrastructure changes. MLflow logs all model code/param changes. Git logs all deployment manifest changes. This provides an immutable audit trail of who changed what and when.

## Q17: Data poisoning attacks. What are they?
**Context:** Internet trolls feeding bad data to a chatbot.
**Answer:** Attackers inject malicious data into the training set to subtly change the model's behavior (e.g., making a spam filter allow a specific spammer). Defend via strict data validation, anomaly detection on inputs, and human-in-the-loop review.

## Q18: Model theft/extraction.
**Context:** Protecting proprietary IP (the model weights).
**Answer:** If deploying to Edge devices, encrypt the ONNX model and use hardware enclaves (TrustZone). If deploying to the cloud, strictly limit API query rates to prevent attackers from querying millions of times to train a proxy model.

## Q19: Using OIDC for Cloud Authentication in CI/CD.
**Context:** Long-lived AWS access keys in GitHub Actions.
**Answer:** Hardcoded keys leak. Use OpenID Connect (OIDC). GitHub Actions proves its identity to AWS, AWS issues a temporary STS token valid only for the duration of the job.

## Q20: Securing the Feature Store.
**Context:** Redis instance exposed without a password.
**Answer:** Redis natively lacks strong RBAC. Run Redis inside a private VPC subnet. Enforce AUTH passwords. Use TLS encryption for data in transit between the inference pod and Redis.

## Q21: What is Federated Learning from a security perspective?
**Context:** Training on sensitive mobile data.
**Answer:** It enhances privacy. Raw data never leaves the user's device. The model is sent to the device, trained locally, and only the mathematically encrypted gradients are sent back to the cloud for aggregation.

## Q22: Handling malware in user-uploaded datasets.
**Context:** Data scientist uploaded a CSV containing a macro virus.
**Answer:** Route all file uploads through a staging S3 bucket. Trigger an AWS Lambda function that runs ClamAV (antivirus scanning). Only move the file to the "clean" bucket if it passes.

## Q23: Zero Trust Architecture in MLOps.
**Context:** Moving away from perimeter-based security.
**Answer:** "Never trust, always verify." Every interaction (User to Jupyter, Pod to S3, Jenkins to K8s) must be authenticated and authorized. No internal network is inherently trusted.

## Q24: Data Lineage for Compliance.
**Context:** Proving a model doesn't use prohibited features.
**Answer:** Use tools like Apache Atlas or MLflow. Map exactly which S3 columns fed into which Feast features, which fed into which MLflow model version.

## Q25: Mitigating Prompt Injection (Context: NLP models).
**Context:** Users bypassing filters in text generation.
**Answer:** Use strict input validation. Separate user input from system prompts. Use an intent classification model to scan inputs for malicious patterns before passing them to the main NLP model.

## Q26: Securing Distributed Training Jobs (MPI/NCCL).
**Context:** Data moving between nodes in plaintext.
**Answer:** Distributed training nodes communicate heavily over the network. If running across AZs or public networks, configure NCCL to encrypt data in transit or use WireGuard VPNs between worker nodes.

## Q27: Security implications of using pre-trained public models.
**Context:** Downloading models from HuggingFace.
**Answer:** Public models can be poisoned or contain malicious `pickle` code. Never load untrusted models directly into production. Scan them with tools like `safetensors`, verify publisher signatures, and run them in sandboxed environments.

## Q28: How do you enforce ML infrastructure compliance?
**Context:** Ensuring all S3 buckets are encrypted.
**Answer:** Use AWS Config or OPA Gatekeeper (K8s). Define policies as code (e.g., "All KServe inferenceservices must have resource limits"). If a deployment violates the policy, the API server rejects it.

## Q29: Handling sensitive logs.
**Context:** Model inputs containing passwords accidentally logged.
**Answer:** Train data scientists on secure coding. Implement a centralized log scrubber daemon (Fluentd) that uses regex to mask PII/passwords before forwarding logs to Elasticsearch.

## Q30: Incident response for a compromised ML model.
**Context:** Discovered a backdoor in a live model.
**Answer:** 1. Containment: Instantly route traffic to a safe fallback or older version via Istio. 2. Revoke all access tokens used by the compromised model's pipeline. 3. Forensic analysis: Preserve the compromised pod and logs for investigation. 4. Patch and redeploy from a known clean state.
