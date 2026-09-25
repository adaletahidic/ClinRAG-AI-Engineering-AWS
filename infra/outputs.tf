output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "alb_url" {
  value = "http://${aws_lb.api.dns_name}"
}

output "knowledge_bucket_name" {
  value = aws_s3_bucket.knowledge.bucket
}

output "cloudwatch_log_group" {
  value = aws_cloudwatch_log_group.api.name
}

output "provider_secret_arn" {
  value = var.secret_arn != "" ? var.secret_arn : aws_secretsmanager_secret.provider[0].arn
}
