locals {
  name = "${var.project_name}-${var.environment}"
  secret_name = var.secret_arn != "" ? var.secret_arn : aws_secretsmanager_secret.provider[0].name
  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags                 = merge(local.tags, { Name = local.name })
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = local.tags
}

resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags                    = merge(local.tags, { Name = "${local.name}-public-${count.index + 1}" })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  tags   = local.tags

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "alb" {
  name        = "${local.name}-alb"
  description = "Allow public HTTP traffic to the ClinRAG ALB."
  vpc_id      = aws_vpc.main.id
  tags        = local.tags

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "service" {
  name        = "${local.name}-service"
  description = "Allow ALB traffic to the ClinRAG ECS service."
  vpc_id      = aws_vpc.main.id
  tags        = local.tags

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecr_repository" "api" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"
  force_delete         = false
  tags                 = local.tags

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}"
  retention_in_days = 30
  tags              = local.tags
}

resource "aws_s3_bucket" "knowledge" {
  bucket_prefix = "${local.name}-knowledge-"
  force_destroy = false
  tags          = local.tags
}

resource "aws_s3_bucket_public_access_block" "knowledge" {
  bucket = aws_s3_bucket.knowledge.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_iam_role" "ecs_execution" {
  name = "${local.name}-ecs-execution"
  tags = local.tags

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  name = "${local.name}-ecs-task"
  tags = local.tags

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "ecs_task" {
  name = "${local.name}-ecs-task-policy"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.knowledge.arn, "${aws_s3_bucket.knowledge.arn}/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = var.secret_arn != "" ? var.secret_arn : aws_secretsmanager_secret.provider[0].arn
      }
    ]
  })
}

resource "aws_secretsmanager_secret" "provider" {
  count                   = var.secret_arn == "" ? 1 : 0
  name                    = "${local.name}/provider"
  recovery_window_in_days = 7
  tags                    = local.tags
}

resource "aws_ecs_cluster" "main" {
  name = local.name
  tags = local.tags
}

resource "aws_ecs_task_definition" "api" {
  family                   = local.name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "4096"
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn
  tags                     = local.tags

  container_definitions = jsonencode([{
    name      = "clinrag-api"
    image     = var.container_image
    essential = true
    portMappings = [{
      containerPort = 8000
      hostPort      = 8000
      protocol      = "tcp"
    }]
    environment = [
      { name = "LLM_PROVIDER", value = var.llm_provider },
      { name = "AWS_SECRET_NAME", value = local.secret_name },
      { name = "ALLOWED_ORIGINS", value = "*" },
      { name = "LOG_LEVEL", value = "INFO" }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.api.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "ecs"
      }
    }
    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')\""]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])
}

resource "aws_lb" "api" {
  name               = substr("${local.name}-alb", 0, 32)
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
  tags               = local.tags
}

resource "aws_lb_target_group" "api" {
  name        = substr("${local.name}-tg", 0, 32)
  port        = 8000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = aws_vpc.main.id
  tags        = local.tags

  health_check {
    path                = "/health"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    forward {
      target_group {
        arn = aws_lb_target_group.api.arn
      }
    }
  }
}

resource "aws_ecs_service" "api" {
  name            = local.name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  tags            = local.tags

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.service.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "clinrag-api"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.http]
}
