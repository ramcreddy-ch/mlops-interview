# Section 10: LLMOps — Operating Large Language Models in Production
## Tools: vLLM, LangChain, RAG Pipelines, Prompt Versioning

---

## 10.1 LLMOps vs Traditional MLOps — What's Different

```
Traditional MLOps:
  Model size:    10MB - 2GB
  Inference:     <10ms GPU time
  Cost/request:  <$0.0001
  Output:        Deterministic (same input → same output)
  Testing:       Accuracy, AUC, F1 (numeric metrics)

LLMOps:
  Model size:    7GB - 700GB
  Inference:     100ms - 5s GPU time
  Cost/request:  $0.001 - $0.10 (token-based billing)
  Output:        Stochastic (same input → different output)
  Testing:       LLM-as-judge, human eval, task-specific metrics
```

**The three hardest problems in LLMOps:**
1. **Cost control** — A poorly configured LLM endpoint can cost $50k/day
2. **Latency at scale** — Generating 500 tokens on a 70B model takes 10+ seconds
3. **Quality assurance** — You cannot write a unit test for "does this sound good?"

---

## 10.2 vLLM — The Production Standard for LLM Serving

vLLM solves the #1 LLM serving bottleneck: **KV Cache memory management**. Traditional serving frameworks allocate the full context window for every request, wasting 60-80% of GPU memory. vLLM uses **PagedAttention** to allocate KV cache in pages (like OS virtual memory), achieving 24x throughput improvement.

### Deploy vLLM on Kubernetes:

```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference-server
  namespace: ml-serving
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.4.0
          command:
            - python
            - -m
            - vllm.entrypoints.openai.api_server
            - --model
            - /models/mistral-7b-instruct
            - --tensor-parallel-size   # Split model across N GPUs
            - "2"
            - --gpu-memory-utilization
            - "0.90"                   # Use 90% of GPU VRAM
            - --max-model-len
            - "8192"                   # Max context length
            - --max-num-seqs
            - "256"                    # Max concurrent sequences
            - --quantization
            - "awq"                    # 4-bit quantization (AWQ)
            - --served-model-name
            - "mistral-7b-instruct"
            - --port
            - "8000"
            - --host
            - "0.0.0.0"
          ports:
            - containerPort: 8000
          resources:
            requests:
              nvidia.com/gpu: 2
              memory: "40Gi"
              cpu: "8"
            limits:
              nvidia.com/gpu: 2
              memory: "40Gi"
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: token
          volumeMounts:
            - name: model-cache
              mountPath: /models
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120   # Models take 2 min to load
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 180
            failureThreshold: 3
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-weights-pvc
      nodeSelector:
        nvidia.com/gpu: "true"
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: ml-serving
spec:
  selector:
    app: vllm-server
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
```

### Python Client for vLLM (OpenAI-compatible API):

```python
# llm_client.py — Production vLLM client with retry + cost tracking
import time
import os
import logging
from openai import OpenAI
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type
)
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# ── Metrics ────────────────────────────────────────────────────────────────────
LLM_REQUESTS = Counter("llm_requests_total", "LLM API calls", ["model", "status"])
LLM_TOKENS = Counter("llm_tokens_total", "Tokens used", ["model", "type"])  # prompt/completion
LLM_LATENCY = Histogram("llm_latency_seconds", "LLM response latency", ["model"])
LLM_COST = Counter("llm_cost_usd_total", "Estimated LLM cost in USD", ["model"])

# Cost per 1M tokens (update as pricing changes)
COST_PER_1M_TOKENS = {
    "gpt-4o": {"prompt": 5.00, "completion": 15.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "mistral-7b-instruct": {"prompt": 0.10, "completion": 0.10},  # Self-hosted estimate
    "claude-3-5-sonnet": {"prompt": 3.00, "completion": 15.00},
}


class ProductionLLMClient:
    def __init__(
        self,
        base_url: str = "http://vllm-service.ml-serving.svc.cluster.local/v1",
        model: str = "mistral-7b-instruct",
        api_key: str = "not-needed-for-vllm",
        max_retries: int = 3,
    ):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.max_retries = max_retries

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
    )
    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1,   # Low temp for deterministic outputs
        timeout: float = 30.0,
    ) -> dict:
        """Send completion request with full observability."""
        start = time.time()

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                stream=False,
            )

            latency = time.time() - start
            LLM_LATENCY.labels(model=self.model).observe(latency)
            LLM_REQUESTS.labels(model=self.model, status="success").inc()

            # Track token usage
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            LLM_TOKENS.labels(model=self.model, type="prompt").inc(prompt_tokens)
            LLM_TOKENS.labels(model=self.model, type="completion").inc(completion_tokens)

            # Track cost
            if self.model in COST_PER_1M_TOKENS:
                costs = COST_PER_1M_TOKENS[self.model]
                total_cost = (
                    prompt_tokens * costs["prompt"] / 1_000_000
                    + completion_tokens * costs["completion"] / 1_000_000
                )
                LLM_COST.labels(model=self.model).inc(total_cost)

            return {
                "text": response.choices[0].message.content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": round(latency * 1000, 2),
                "finish_reason": response.choices[0].finish_reason,
            }

        except Exception as e:
            LLM_REQUESTS.labels(model=self.model, status="error").inc()
            logger.error(f"LLM request failed: {e}")
            raise
```

---

## 10.3 RAG Pipelines — Production Architecture

RAG (Retrieval-Augmented Generation) solves LLM hallucination by grounding responses in real retrieved documents.

```
RAG Architecture:

OFFLINE (Indexing) — runs periodically:
  Documents → Chunking → Embedding Model → Vector DB (Pinecone/Milvus/pgvector)

ONLINE (Query time) — runs per request:
  User Query → Embedding → ANN Search in Vector DB → Top-K Chunks
       → Prompt Assembly (query + chunks) → LLM → Response
```

### Production RAG Implementation:

```python
# rag_pipeline.py — Production-grade RAG with LangChain + pgvector
import os
import hashlib
import logging
from typing import List, Optional
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Document:
    doc_id: str
    content: str
    metadata: dict
    chunk_id: Optional[int] = None


class ProductionRAGPipeline:
    def __init__(
        self,
        db_url: str,
        llm_base_url: str,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
    ):
        # Embedding model (runs locally — no API cost for embeddings)
        self.embedder = SentenceTransformer(
            embedding_model,
            device="cuda" if os.environ.get("USE_GPU") else "cpu"
        )

        # PostgreSQL with pgvector extension
        self.conn = psycopg2.connect(db_url)
        self._init_db()

        # vLLM for generation
        self.llm = OpenAI(base_url=llm_base_url, api_key="not-needed")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

    def _init_db(self):
        """Initialize pgvector table."""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(1024),    -- BGE-large produces 1024-dim
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(doc_id, chunk_id)
                );
            """)
            # IVFFlat index for fast ANN search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embedding_idx
                ON document_chunks
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            self.conn.commit()

    def chunk_document(self, content: str) -> List[str]:
        """Split document into overlapping chunks."""
        words = content.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def index_document(self, doc: Document) -> int:
        """Embed and store a document. Returns number of chunks indexed."""
        chunks = self.chunk_document(doc.content)
        embeddings = self.embedder.encode(
            chunks,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        with self.conn.cursor() as cur:
            for chunk_id, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                cur.execute("""
                    INSERT INTO document_chunks
                        (doc_id, chunk_id, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id, chunk_id)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata;
                """, (
                    doc.doc_id,
                    chunk_id,
                    chunk,
                    embedding.tolist(),
                    doc.metadata,
                ))
            self.conn.commit()

        logger.info(f"Indexed document {doc.doc_id}: {len(chunks)} chunks")
        return len(chunks)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """Retrieve most relevant chunks for a query."""
        top_k = top_k or self.top_k

        # Embed the query with the same model used for indexing
        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True
        )[0]

        with self.conn.cursor() as cur:
            # Cosine similarity search using pgvector
            cur.execute("""
                SELECT
                    doc_id,
                    chunk_id,
                    content,
                    metadata,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (
                query_embedding.tolist(),
                query_embedding.tolist(),
                top_k
            ))

            results = []
            for row in cur.fetchall():
                results.append({
                    "doc_id": row[0],
                    "chunk_id": row[1],
                    "content": row[2],
                    "metadata": row[3],
                    "similarity": float(row[4]),
                })

        return results

    def query(
        self,
        user_question: str,
        system_prompt: str = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> dict:
        """Full RAG query: retrieve + generate."""

        # Step 1: Retrieve relevant chunks
        chunks = self.retrieve(user_question, top_k=self.top_k)

        if not chunks:
            return {
                "answer": "I don't have relevant information to answer this question.",
                "sources": [],
                "retrieved_chunks": 0,
            }

        # Step 2: Filter low-quality retrievals (below similarity threshold)
        MIN_SIMILARITY = 0.5
        chunks = [c for c in chunks if c["similarity"] >= MIN_SIMILARITY]

        # Step 3: Assemble context-augmented prompt
        context_parts = []
        for i, chunk in enumerate(chunks):
            source_label = f"[Source {i+1}: {chunk['metadata'].get('title', chunk['doc_id'])}]"
            context_parts.append(f"{source_label}\n{chunk['content']}")

        context = "\n\n---\n\n".join(context_parts)

        if not system_prompt:
            system_prompt = """You are a helpful assistant. Answer questions based ONLY
on the provided context. If the context doesn't contain the answer, say so explicitly.
Do not make up information. Always cite your sources."""

        full_prompt = f"""Context:
{context}

---

Question: {user_question}

Answer based on the context above:"""

        # Step 4: Generate with LLM
        response = self.llm.chat.completions.create(
            model="mistral-7b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": [
                {
                    "doc_id": c["doc_id"],
                    "similarity": round(c["similarity"], 4),
                    "excerpt": c["content"][:200] + "...",
                }
                for c in chunks
            ],
            "retrieved_chunks": len(chunks),
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
```

---

## 10.4 Prompt Versioning — Git for Prompts

Prompts are production artifacts. They must be versioned, tested, and reviewed like code.

```
Problem: A data scientist edits a system prompt to "make it sound friendlier."
         The change silently causes the model to ignore safety rules.
         No one notices for 3 weeks.

Solution: Every prompt change goes through Git + CI/CD.
```

```python
# prompt_registry.py — Version-controlled prompt management
import yaml
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptVersion:
    name: str
    version: str
    template: str
    variables: list
    model: str
    temperature: float
    max_tokens: int
    description: str
    sha: str  # Content hash for integrity


class PromptRegistry:
    """Loads prompts from YAML files — prompts live in Git."""

    def __init__(self, prompts_dir: str = "prompts/"):
        self.prompts_dir = Path(prompts_dir)
        self._cache = {}

    def load(self, name: str, version: str = "latest") -> PromptVersion:
        """Load a specific prompt version."""
        cache_key = f"{name}:{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if version == "latest":
            # Find the latest version file
            files = sorted(self.prompts_dir.glob(f"{name}-v*.yaml"))
            if not files:
                raise FileNotFoundError(f"No prompt found: {name}")
            prompt_file = files[-1]
        else:
            prompt_file = self.prompts_dir / f"{name}-{version}.yaml"

        with open(prompt_file) as f:
            config = yaml.safe_load(f)

        content = yaml.dump(config)
        sha = hashlib.sha256(content.encode()).hexdigest()[:12]

        pv = PromptVersion(
            name=config["name"],
            version=config["version"],
            template=config["template"],
            variables=config.get("variables", []),
            model=config["model"],
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 512),
            description=config.get("description", ""),
            sha=sha,
        )
        self._cache[cache_key] = pv
        return pv

    def render(self, name: str, variables: dict, version: str = "latest") -> str:
        """Render a prompt template with variables."""
        prompt = self.load(name, version)
        return prompt.template.format(**variables)
```

```yaml
# prompts/fraud-explanation-v1.2.yaml
name: fraud-explanation
version: "1.2"
model: mistral-7b-instruct
temperature: 0.1
max_tokens: 256
description: >
  Explains why a transaction was flagged as potentially fraudulent.
  Used in the customer-facing dispute portal.

variables:
  - fraud_probability
  - amount
  - currency
  - merchant_name
  - risk_factors

template: |
  You are a fraud detection assistant at a bank. A transaction has been flagged.
  Explain to the customer why this transaction was flagged without revealing
  internal model details or security mechanisms.

  Transaction details:
  - Amount: {amount} {currency}
  - Merchant: {merchant_name}
  - Risk score: {fraud_probability:.0%}
  - Risk factors: {risk_factors}

  Provide a brief, non-technical, customer-friendly explanation (2-3 sentences).
  Do not use technical jargon. Be empathetic and professional.
```

---

## 10.5 LLM Cost Optimization — Production Techniques

```
Real scenario: A company deployed GPT-4 for all customer service queries.
Monthly cost: $180,000.
After optimization: $12,000/month (93% reduction).
```

### Technique 1: Model Routing — Use Small Models When Possible

```python
# intelligent_router.py — Route to cheapest model that can handle the query
import anthropic
from openai import OpenAI

class ModelRouter:
    """Route queries to the most cost-effective model."""

    ROUTING_RULES = [
        # Simple factual queries → cheapest model
        {
            "condition": lambda q: len(q.split()) < 20 and "?" in q,
            "model": "gpt-4o-mini",
            "reason": "simple_factual_query"
        },
        # Code generation → mid-tier
        {
            "condition": lambda q: any(kw in q.lower()
                for kw in ["code", "python", "function", "sql", "write a"]),
            "model": "gpt-4o",
            "reason": "code_generation"
        },
        # Complex reasoning → premium
        {
            "condition": lambda q: any(kw in q.lower()
                for kw in ["analyze", "explain", "compare", "strategy"]),
            "model": "claude-3-5-sonnet-20241022",
            "reason": "complex_reasoning"
        },
    ]
    DEFAULT_MODEL = "gpt-4o-mini"

    def route(self, query: str) -> tuple[str, str]:
        for rule in self.ROUTING_RULES:
            if rule["condition"](query):
                return rule["model"], rule["reason"]
        return self.DEFAULT_MODEL, "default"
```

### Technique 2: Response Caching for Identical Queries

```python
# llm_cache.py — Semantic caching using embeddings
import redis
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticLLMCache:
    """Cache LLM responses. Reuse for semantically similar queries."""

    def __init__(self, redis_url: str, similarity_threshold: float = 0.97):
        self.redis = redis.from_url(redis_url)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.similarity_threshold = similarity_threshold

    def _exact_key(self, query: str, system_prompt: str, model: str) -> str:
        """Exact match cache key."""
        content = f"{model}:{system_prompt}:{query}"
        return f"llm:exact:{hashlib.md5(content.encode()).hexdigest()}"

    def get_or_none(self, query: str, system_prompt: str, model: str):
        """Check exact cache first, then semantic cache."""
        # 1. Try exact match (fastest)
        exact_key = self._exact_key(query, system_prompt, model)
        cached = self.redis.get(exact_key)
        if cached:
            return cached.decode(), "exact_cache_hit"

        # 2. Try semantic similarity (for paraphrased queries)
        query_embedding = self.embedder.encode([query])[0]
        # ... scan recent embeddings and find similar ones
        # (In production, use a vector DB like Redis with vector search)

        return None, "cache_miss"

    def store(self, query: str, system_prompt: str, model: str, response: str):
        """Store response in cache with 24h TTL."""
        exact_key = self._exact_key(query, system_prompt, model)
        self.redis.setex(exact_key, 86400, response)
```

### Technique 3: Token Optimization

```python
# token_optimizer.py — Reduce prompt tokens without losing quality

def optimize_prompt(prompt: str, max_chars: int = 2000) -> str:
    """Truncate context intelligently — keep start and end."""
    if len(prompt) <= max_chars:
        return prompt

    # Keep first 60% and last 40% (beginning = instructions, end = question)
    keep_start = int(max_chars * 0.6)
    keep_end = max_chars - keep_start

    return (
        prompt[:keep_start]
        + "\n...[middle context truncated]...\n"
        + prompt[-keep_end:]
    )


def compress_context(chunks: list, budget_tokens: int = 2000) -> list:
    """Select most relevant chunks within token budget."""
    selected = []
    used_tokens = 0

    # Sort by similarity (most relevant first)
    chunks_sorted = sorted(chunks, key=lambda x: x["similarity"], reverse=True)

    for chunk in chunks_sorted:
        # Rough token estimate: 1 token ≈ 4 characters
        chunk_tokens = len(chunk["content"]) // 4
        if used_tokens + chunk_tokens <= budget_tokens:
            selected.append(chunk)
            used_tokens += chunk_tokens
        else:
            break

    return selected
```

---

## 10.6 LLM Observability — What to Monitor

```python
# llm_evaluator.py — Automated LLM quality evaluation pipeline
import json
from openai import OpenAI

class LLMEvaluator:
    """Use a strong LLM to evaluate outputs of a weaker LLM (LLM-as-judge)."""

    def __init__(self):
        self.judge = OpenAI()  # Use GPT-4o as judge (stronger model)

    def evaluate_response(
        self,
        question: str,
        context: str,
        response: str,
    ) -> dict:
        """Score an LLM response on multiple dimensions."""

        judge_prompt = f"""You are an expert evaluator. Score the following AI response.

Question: {question}

Context provided to the AI:
{context[:1000]}

AI Response:
{response}

Score on these dimensions (1-5, where 5 is best):
1. Faithfulness: Does the answer stick to the provided context?
2. Relevance: Does the answer address the question?
3. Completeness: Is the answer thorough enough?
4. Hallucination: Does it make up facts? (5=no hallucination, 1=heavy hallucination)

Return ONLY JSON: {{"faithfulness": N, "relevance": N, "completeness": N, "hallucination": N, "reasoning": "brief explanation"}}"""

        judge_response = self.judge.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        scores = json.loads(judge_response.choices[0].message.content)
        scores["overall"] = (
            scores["faithfulness"] * 0.35
            + scores["relevance"] * 0.30
            + scores["completeness"] * 0.20
            + scores["hallucination"] * 0.15
        )

        return scores
```

---

*Next → [Section 11: Security & Governance](../security/README.md)*
