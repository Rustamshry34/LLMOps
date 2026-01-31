provider "aws" {
  region = var.region
}

# Ubuntu 22.04 LTS (Jammy) - x86_64
data "aws_ami" "ubuntu_x86_64" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# Default VPC subnets (one per AZ)
data "aws_subnets" "default" {
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

# SSH key pair — Terraform tarafından yönetilir
resource "aws_key_pair" "deployer" {
  key_name   = var.key_pair_name
  public_key = var.public_key
}

# Security group: allow SSH (22) and inference API (8000)
resource "aws_security_group" "ec2_sg" {
  name_prefix = "llmops-sg-"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "llmops-inference-sg"
  }
}

# EC2 Instance
resource "aws_instance" "llm_app" {
  ami                    = data.aws_ami.ubuntu_x86_64.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.deployer.key_name
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]
  subnet_id              = data.aws_subnets.default.ids[0]

  user_data = <<-EOF
              #!/bin/bash
              apt-get update
              apt-get install -y docker.io
              systemctl start docker
              systemctl enable docker
              usermod -aG docker ubuntu

              # Pull and run the inference server from GHCR (public image)
              docker run -d \
                --name llm-inference \
                -p 8000:8000 \
                -e MODEL_ID=${var.hf_model_id} \
                ghcr.io/${var.github_repo}:${var.docker_image_tag}
              EOF

  lifecycle {
    replace_triggered_by = [aws_key_pair.deployer]
  }

  tags = {
    Name = "llmops-inference-server"
  }
}
