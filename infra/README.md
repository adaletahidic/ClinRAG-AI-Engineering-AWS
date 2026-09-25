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

## Terraform deployment

From `infra/`:

```powershell
terraform init
Copy-Item .\terraform.tfvars.example .\terraform.tfvars
# Edit terraform.tfvars with the AWS account image URI.
terraform fmt -recursive
terraform validate
terraform plan
terraform apply
```

Run `terraform fmt -recursive` and `terraform validate` before every plan.
Terraform is not bundled with the Python environment, so install Terraform
CLI separately before running these commands.

The stack creates ECR, a public VPC, ECS/Fargate, ALB, CloudWatch Logs,
private-by-default S3 access, IAM roles, and a Secrets Manager secret
container. It does not write provider secret values. Populate the secret
through AWS Secrets Manager after provisioning, then push the image to ECR and
update `container_image`.

For a first image push:

```powershell
aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.eu-central-1.amazonaws.com
docker tag clinrag-ai-engineering-aws-clinrag-api:latest ACCOUNT_ID.dkr.ecr.eu-central-1.amazonaws.com/clinrag-dev:latest
docker push ACCOUNT_ID.dkr.ecr.eu-central-1.amazonaws.com/clinrag-dev:latest
```
