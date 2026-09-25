# AWS deployment boundary

The local application is ready for AWS integration through these seams:

- `LLMProviderFactory` can select `BedrockProvider` with `LLM_PROVIDER=bedrock`.
- `S3KnowledgeSource` implements the knowledge-source contract for `.pdf`,
  `.txt`, and `.md` objects.
- `app.knowledge.interfaces.VectorStore` is the boundary for replacing local
  TF-IDF retrieval with an AWS-backed vector service.
- `load_secret_environment()` loads string key/value JSON from Secrets
  Manager without putting secrets in the image.

The deployment layer should provision ECR, ECS/Fargate, an ALB, CloudWatch
logs, Secrets Manager, IAM roles, and S3. Keep these resources outside the
application package and inject names/ARNs through environment variables.

Before applying infrastructure, choose one IaC tool and configure the AWS
account, region, VPC, subnets, domain/TLS, and secret names. Do not hard-code
those values in application code.
