# terraform/backend.tf
terraform {
  backend "s3" {
    bucket         = "terraform-remote-state4"
    key            = "llm/terraform.tfstate"
    region         = "eu-north-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}
