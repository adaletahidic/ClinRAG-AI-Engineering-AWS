variable "aws_region" {
  type    = string
  default = "eu-central-1"
}

variable "project_name" {
  type    = string
  default = "clinrag"
}

variable "environment" {
  type    = string
  default = "dev"
}

variable "container_image" {
  type        = string
  description = "Full ECR image URI, including tag."
}

variable "llm_provider" {
  type    = string
  default = "groq"
}

variable "secret_arn" {
  type        = string
  description = "Secrets Manager ARN containing provider keys as JSON fields."
  default     = ""
}

variable "vpc_cidr" {
  type    = string
  default = "10.42.0.0/16"
}
