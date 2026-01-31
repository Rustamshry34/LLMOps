output "instance_public_ip" {
  description = "Public IP address of the deployed inference server"
  value       = aws_instance.ats_app.public_ip
}

output "inference_endpoint" {
  description = "FastAPI inference endpoint URL"
  value       = "http://${aws_instance.ats_app.public_ip}:8000/generate"
}

output "health_check_url" {
  description = "Health check URL for the inference server"
  value       = "http://${aws_instance.ats_app.public_ip}:8000/health"
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/id_rsa ubuntu@${aws_instance.ats_app.public_ip}"
}
