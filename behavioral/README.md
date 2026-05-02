# Behavioral & Leadership - Interview Questions

## Q1: Tell me about a time you handled a critical production incident involving an ML model.
**Context:** High decline rate on the fraud model.
**Answer:** Triage: I stopped the bleeding first by invoking a kill switch in ArgoCD to fall back to a legacy rules engine. Investigation: Traced ELK logs, found upstream Spark job failed, causing Redis features to expire. Resolution: Restarted Spark job, added alerts for "Feature Missing Rate > 5%".

## Q2: How do you drive adoption of MLOps practices across a resistant data science team?
**Context:** Devs thought CI/CD was "red tape".
**Answer:** Don't force governance; solve a pain point. I built a self-service JupyterHub on K8s so they could get GPUs in 30 seconds instead of waiting 3 weeks for IT. Once they loved the platform, I introduced MLflow as a "convenience tool" to track experiments. Adoption hit 100%.

## Q3: Explain a complex MLOps architecture to a non-technical executive.
**Context:** VP of Product asked why deploying takes 3 weeks.
**Answer:** "A data scientist writes a great recipe (the model). MLOps builds the factory to produce that recipe 10,000 times a day perfectly. We have to build the supply chain (data engineering), quality control (CI/CD), and health inspectors (monitoring). It takes time to build the factory, but once built, new recipes launch in days."

## Q4: Describe a time you disagreed with a Data Scientist on model deployment.
**Context:** DS wanted to deploy a massive deep learning model for a simple tabular dataset.
**Answer:** I showed them the math: the DL model got 0.5% better AUC, but cost $10k/month in GPU serving costs. An XGBoost model got 99.5% of the performance but ran on cheap CPUs for $500/month. We compromised: deployed XGBoost, keeping DL as a V2 research project.

## Q5: How do you prioritize technical debt in an MLOps platform?
**Context:** Platform was a mess of custom bash scripts.
**Answer:** I prioritize tech debt based on "Blast Radius" and "Toil". If a script fails once a year but takes the whole system down (High Blast Radius), it's P1. If a deployment script takes 2 hours of manual engineer time every week (High Toil), it's P1.

## Q6: Tell me about a time you failed or made a major mistake.
**Context:** Deployed an untested Helm chart to the K8s cluster.
**Answer:** I accidentally overwrote the production MLflow database credentials. Tracking went down for 3 hours. I took responsibility, restored the DB from a snapshot. The real fix: I implemented a Terraform CI pipeline that runs `terraform plan` and requires a senior engineer's approval before any infrastructure changes are applied.

## Q7: How do you handle a situation where a model's business metrics drop, but technical metrics are fine?
**Context:** Model accuracy was high, but revenue dropped.
**Answer:** I brought the Data Science and Product teams together. We discovered the model was highly accurate at predicting clicks, but the users were clicking on low-margin items. The business objective had shifted from "engagement" to "profitability". We had to retrain the model with a new target variable.

## Q8: Tell me about a time you had to learn a new technology quickly.
**Context:** Company mandated moving from AWS to GCP in 3 months.
**Answer:** I had never used Vertex AI. I spent the first weekend reading the core docs. I built a tiny, end-to-end "tracer bullet" pipeline (dummy model, deployed to an endpoint). Once the end-to-end path was proven, migrating the massive production models was just an exercise in scaling.

## Q9: How do you mentor junior MLOps engineers?
**Context:** Helping a software engineer transition to MLOps.
**Answer:** I pair-program with them on real tickets. I don't just tell them to "use Kubernetes." I explain *why* ML needs K8s (resource isolation, GPU scheduling). I assign them low-risk, high-impact tasks (like building a new Datadog dashboard) to build confidence.

## Q10: How do you balance speed to market with platform stability?
**Context:** PM wanted a model live by Friday, bypassing CI/CD.
**Answer:** I use the "Shadow Mode" compromise. I offered to deploy the model by Friday, but only in shadow mode (logging predictions, not serving them to users). This satisfied the PM's need to "get it running" while protecting the user experience until we could properly validate it on Monday.

## Q11: Describe a time you optimized a costly ML process.
**Context:** AWS bill was out of control.
**Answer:** I noticed teams were leaving Jupyter notebooks running on expensive A100 GPUs over the weekend. I wrote a Lambda script that automatically hibernates idle notebook instances at 7 PM Friday. It saved the company $20,000 in the first month and won an internal engineering award.

## Q12: How do you stay updated with the fast-moving MLOps landscape?
**Context:** New tools launch every week.
**Answer:** I ignore the hype cycle (I don't adopt a tool just because it has a shiny website). I read engineering blogs from companies operating at scale (Uber, Netflix, DoorDash). I focus on fundamental principles (isolation, reproducibility) rather than specific tools.

## Q13: Tell me about a time you had to push back on a stakeholder request.
**Context:** Marketing wanted a "real-time" deep learning model.
**Answer:** I asked *why* they needed it in real-time. They said they wanted to email users based on their activity today. I explained that an hourly batch job would achieve the exact same business outcome, but cost 1/10th the price and be far more reliable. They agreed.

## Q14: How do you foster a culture of reliability in MLOps?
**Context:** Teams were throwing models over the wall.
**Answer:** I implemented "You build it, you run it." The Data Science team and the MLOps team shared the PagerDuty rotation. Once Data Scientists had to wake up at 3 AM because their pandas script crashed on a null value, they started writing much more robust data validation tests.

## Q15: Describe your approach to conducting a post-mortem.
**Context:** A model served garbage predictions for a day.
**Answer:** Blameless culture. We focus on the *system*, not the *person*. If John deployed a bad model, the question isn't "Why did John do that?", it's "Why did the CI/CD pipeline allow a bad model to reach production?" We generate actionable Jira tickets to fix the pipeline.

## Q16: How do you bridge the gap between Data Engineering and Data Science?
**Context:** DEs changed schemas, breaking DS models.
**Answer:** I introduced data contracts. The DE team cannot change a table schema without a PR that runs a test against the downstream ML pipelines. I also implemented a Feature Store, which acts as the official API contract between the two teams.

## Q17: Tell me about a complex migration project you led.
**Context:** Moving from Jenkins to ArgoCD (GitOps).
**Answer:** I didn't migrate everything at once. I chose a low-priority, internal-facing model. I built the GitOps pipeline, documented the pain points, and showed the DS team how much easier rollbacks were. I then created a migration script and moved the rest of the 50 models over a 2-month period.

## Q18: How do you evaluate whether to build vs. buy an MLOps tool?
**Context:** Deciding whether to buy Weights & Biases or host MLflow.
**Answer:** I look at the core competency of the business. Are we an infrastructure company? No. Do we have 3 engineers to dedicate to maintaining MLflow, Postgres, and S3 backups? No. Buying W&B cost $10k/year, but saved us $150k in engineering time.

## Q19: How do you handle an underperforming team member?
**Context:** An engineer kept cutting corners on testing.
**Answer:** I sat down with them privately. I didn't attack their character; I pointed out the specific pull requests where tests were skipped and explained the risk to the platform. I discovered they were overwhelmed by the test framework syntax. We paired programmed for a week to get them comfortable.

## Q20: Describe a time you advocated for a new technology that was initially rejected.
**Context:** I wanted to move to Kubernetes, management said it was too complex.
**Answer:** I didn't argue with slides. I built a proof-of-concept. I took our slowest, most expensive model, containerized it, and ran it on a local Minikube cluster. I demonstrated how it automatically recovered from crashes and could scale to 0. The demo won them over.

## Q21: How do you ensure your ML platform is inclusive and accessible?
**Context:** Platform was too hard for junior analysts to use.
**Answer:** I built abstractions. A Senior ML Engineer can write raw K8s YAML if they want. But for junior analysts, I built a simple CLI tool where they just type `ml-deploy my_model.pkl`. The platform automatically generates the Dockerfile and YAML behind the scenes.

## Q22: Tell me about a time you had to troubleshoot a problem with zero documentation.
**Context:** A legacy Scala Spark job written by someone who left 3 years ago broke.
**Answer:** I treated it like a black box. I couldn't understand the Scala code quickly, so I looked at the inputs (S3) and the outputs (Redis). I noticed the output count was exactly half of the input count. I ran it locally with a debugger and found a silent data truncation error.

## Q23: How do you handle cross-functional dependencies delaying your project?
**Context:** Waiting on the Security team to approve a new Docker base image.
**Answer:** I didn't just complain in standup. I met with the Security lead, understood their concerns (CVE scanning), and offered to write the Trivy CI pipeline integration myself to automate their job. By doing the heavy lifting for them, the approval went through the next day.

## Q24: What is your philosophy on documentation in MLOps?
**Context:** The wiki was outdated and useless.
**Answer:** Documentation that lives outside the code dies. I enforce "Docs as Code". Architecture decision records (ADRs) are markdown files stored in the same Git repo as the infrastructure code. Runbooks are stored alongside the Datadog alert definitions.

## Q25: How do you approach setting goals (OKRs) for an MLOps team?
**Context:** Leading a platform squad.
**Answer:** I align them with business value, not just tech metrics. Bad: "Deploy Kubernetes." Good: "Reduce model time-to-market from 3 weeks to 3 days." Good: "Reduce ML infrastructure cloud spend by 20% while maintaining 99.9% uptime."

## Q26: Tell me about a time you compromised on technical excellence to meet a deadline.
**Context:** Product launch in 1 week.
**Answer:** We didn't have time to build the automated retraining pipeline. I deployed the model with manual retraining, but I explicitly documented this as Technical Debt in Jira, and got PM agreement that the first sprint *after* launch would be dedicated exclusively to automating the pipeline.

## Q27: How do you measure the success of the MLOps platform itself?
**Context:** Justifying the platform team's existence to the CTO.
**Answer:** I track three core platform metrics: 1. Deployment frequency (how often are models updated). 2. Lead time for changes (time from Git commit to production). 3. Mean Time To Recovery (MTTR) when an inference node goes down.

## Q28: Tell me about a time you had to deal with ambiguity.
**Context:** PM said "We need to use AI for customer support."
**Answer:** That's not a requirement. I spent two weeks shadowing the customer support team. I found they spent 40% of their time just classifying the category of an incoming ticket. We built a simple NLP classification model for that specific task, delivering immediate value.

## Q29: How do you handle security vulnerabilities found in production models?
**Context:** Zero-day Log4j vulnerability hit our Java-based feature store.
**Answer:** Incident response mode. We immediately identified all affected containers using our Trivy SBOMs (Software Bill of Materials). We patched the base images, bypassed the standard 2-week QA cycle using our emergency "hotfix" CI pipeline, and rolled out the fix in 4 hours.

## Q30: Why do you want to be a Principal/Staff MLOps Engineer?
**Context:** Closing question.
**Answer:** At the senior level, I enjoyed solving complex technical problems (like optimizing GPU memory). But at the Staff level, I want to multiply my impact. I want to design architectures that prevent entire categories of problems, mentor junior engineers, and bridge the strategic gap between ML capabilities and business objectives.
