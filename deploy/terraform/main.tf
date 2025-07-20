provider "aws" {
  region = "us-east-1"
}

resource "aws_vpc" "coloran_vpc" {
  cidr_block = "10.0.0.0/16"
  tags = {
    Name = "coloran-vpc"
  }
}

# Add more resources for EKS, EC2, etc.
