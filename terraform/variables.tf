variable "region" {
  description = "AWS region"
  type        = string
  default     = "eu-north-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

variable "key_pair_name" {
  description = "Name of the SSH key pair"
  type        = string
  default     = "llmops-deployer"
}

variable "github_repo" {
  description = "GitHub repository name for GHCR image (e.g., owner/repo)"
  type        = string
  default     = "rustamshry34/llmops"
}

variable "docker_image_tag" {
  description = "Docker image tag to pull from GHCR"
  type        = string
  default     = "latest"
}

variable "hf_model_id" {
  description = "Hugging Face model ID used by the inference server"
  type        = string
  default     = "Rustamshry/Qwen3-CoT"
}
