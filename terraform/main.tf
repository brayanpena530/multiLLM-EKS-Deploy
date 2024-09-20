################################################################################
# Common locals
################################################################################
locals {
  name   = "self-hosted-llm"
  region = "us-west-2"

  vpc_cidr   = "10.0.0.0/16"
  num_of_azs = 2

  tags = {
    GithubRepo       = "https://github.com/eliran89c/self-hosted-llm-on-eks"
    BaseEKSBlueprint = "github.com/aws-ia/terraform-aws-eks-blueprints/patterns/karpenter"
  }
}

################################################################################
# Providers
################################################################################
provider "aws" {
  region = local.region
}

# Required for public ECR where Karpenter artifacts are hosted
provider "aws" {
  region = "us-east-1"
  alias  = "us-east-1"
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    # This requires the awscli to be installed locally where Terraform is executed
    args = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--region", local.region]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      # This requires the awscli to be installed locally where Terraform is executed
      args = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--region", local.region]
    }
  }
}

provider "kubectl" {
  apply_retry_count      = 5
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  load_config_file       = false

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    # This requires the awscli to be installed locally where Terraform is executed
    args = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--region", local.region]
  }
}

################################################################################
# Data
################################################################################
data "aws_availability_zones" "available" {}

data "aws_ecrpublic_authorization_token" "token" {
  provider = aws.us-east-1
}

################################################################################
# Network
################################################################################
locals {
  azs = slice(data.aws_availability_zones.available.names, 0, 2)
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = local.name
  cidr = local.vpc_cidr

  azs             = local.azs
  private_subnets = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 4, k)]
  public_subnets  = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 48)]

  enable_nat_gateway = true
  single_nat_gateway = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
    # Tags subnets for Karpenter auto-discovery
    "karpenter.sh/discovery" = local.name
  }

  tags = local.tags
}

################################################################################
# Node Group Role
################################################################################

resource "aws_iam_role" "gpu_node_group" {
  name = "eks-node-gpu-group"

  assume_role_policy = jsonencode({
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
    Version = "2012-10-17"
  })
}

resource "aws_iam_role_policy_attachment" "gpu-node-group-AmazonEKSWorkerNodePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.gpu_node_group.name
}

resource "aws_iam_role_policy_attachment" "gpu-node-group-AmazonEKS_CNI_Policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.gpu_node_group.name
}

resource "aws_iam_role_policy_attachment" "gpu-node-group-AmazonEC2ContainerRegistryReadOnly" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.gpu_node_group.name
}

################################################################################
# Cluster
################################################################################
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.21"

  cluster_name    = local.name
  cluster_version = "1.29"

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  # Fargate profiles use the cluster primary security group so these are not utilized
  create_cluster_security_group = false
  create_node_security_group    = false

  fargate_profiles = {
    karpenter = {
      selectors = [
        { namespace = "karpenter" }
      ]
    }
    kube_system = {
      name = "coredns"
      selectors = [
        { namespace = "kube-system", labels = { "k8s-app" = "kube-dns" } }
      ]
    }
  }

  manage_aws_auth_configmap = true
  aws_auth_roles = [
    # We need to add in the Karpenter node IAM role for nodes launched by Karpenter
    {
      rolearn  = module.core_addons.karpenter.node_iam_role_arn
      username = "system:node:{{EC2PrivateDNSName}}"
      groups = [
        "system:bootstrappers",
        "system:nodes",
      ]
    },
  ]

  tags = merge(local.tags, {
    # NOTE - if creating multiple security groups with this module, only tag the
    # security group that Karpenter should utilize with the following tag
    # (i.e. - at most, only one security group should have this tag in your account)
    "karpenter.sh/discovery" = local.name
  })
}

resource "aws_eks_node_group" "example" {
  cluster_name    = module.eks.cluster_name
  node_group_name = "gpu-nodes"
  node_role_arn   = aws_iam_role.gpu_node_group.arn
  subnet_ids      = module.vpc.private_subnets

  scaling_config {
    desired_size = 1
    max_size     = 2
    min_size     = 1
  }

  instance_types = ["g5.xlarge"]

  update_config {
    max_unavailable = 1
  }

  ami_type = "BOTTLEROCKET_x86_64_NVIDIA"

  # Ensure that IAM Role permissions are created before and deleted after EKS Node Group handling.
  # Otherwise, EKS will not be able to properly delete EC2 Instances and Elastic Network Interfaces.
  depends_on = [
    aws_iam_role_policy_attachment.gpu-node-group-AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.gpu-node-group-AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.gpu-node-group-AmazonEC2ContainerRegistryReadOnly,
  ]
}

################################################################################
# Core addons
################################################################################
module "core_addons" {
  source  = "aws-ia/eks-blueprints-addons/aws"
  version = "~> 1.14"

  cluster_name      = module.eks.cluster_name
  cluster_endpoint  = module.eks.cluster_endpoint
  cluster_version   = module.eks.cluster_version
  oidc_provider_arn = module.eks.oidc_provider_arn

  # We want to wait for the Fargate profiles to be deployed first
  create_delay_dependencies = [for prof in module.eks.fargate_profiles : prof.fargate_profile_arn]

  eks_addons = {
    coredns = {
      configuration_values = jsonencode({
        computeType = "Fargate"
        # Ensure that the we fully utilize the minimum amount of resources that are supplied by
        # Fargate https://docs.aws.amazon.com/eks/latest/userguide/fargate-pod-configuration.html
        # Fargate adds 256 MB to each pod's memory reservation for the required Kubernetes
        # components (kubelet, kube-proxy, and containerd). Fargate rounds up to the following
        # compute configuration that most closely matches the sum of vCPU and memory requests in
        # order to ensure pods always have the resources that they need to run.
        resources = {
          limits = {
            cpu = "0.25"
            # We are targeting the smallest Task size of 512Mb, so we subtract 256Mb from the
            # request/limit to ensure we can fit within that task
            memory = "256M"
          }
          requests = {
            cpu = "0.25"
            # We are targeting the smallest Task size of 512Mb, so we subtract 256Mb from the
            # request/limit to ensure we can fit within that task
            memory = "256M"
          }
        }
      })
    }
    vpc-cni    = {}
    kube-proxy = {}
  }

  # Enable Karpenter
  enable_karpenter = true

  karpenter = {
    repository_username = data.aws_ecrpublic_authorization_token.token.user_name
    repository_password = data.aws_ecrpublic_authorization_token.token.password
  }

  karpenter_node = {
    iam_role_use_name_prefix = false
  }

  tags = local.tags
}

################################################################################
# Karpenter manifest
################################################################################
resource "kubectl_manifest" "karpenter" {
  for_each = fileset("${path.module}/karpenter", "*.yaml")

  yaml_body = templatefile("${path.module}/karpenter/${each.key}", {
    cluster_name = module.eks.cluster_name
  })

  depends_on = [module.core_addons]
}

################################################################################
# Additional addons
################################################################################
# We need to wait for the Karpenter manifest to be deployed first
module "additional_addons" {
  source  = "aws-ia/eks-blueprints-addons/aws"
  version = "~> 1.16"

  cluster_name      = module.eks.cluster_name
  cluster_endpoint  = module.eks.cluster_endpoint
  cluster_version   = module.eks.cluster_version
  oidc_provider_arn = module.eks.oidc_provider_arn

  # Install Prometheus and Grafana
  enable_metrics_server        = true
  enable_kube_prometheus_stack = false

  # Disable Prometheus node exporter
  kube_prometheus_stack = {
    values = [
      jsonencode({
        nodeExporter = {
          enabled = false
        },
        alertmanager = {
          enabled = false
        }
      })
    ]
  }

  # Install the nvidia-device-plugin
  helm_releases = {
    nvidia-plugin = {
      repository       = "https://nvidia.github.io/k8s-device-plugin"
      chart            = "nvidia-device-plugin"
      chart_version    = "0.15.0"
      namespace        = "nvidia-device-plugin"
      create_namespace = true
    }

    # This Helm chart configures the KubeRay Operator, which can be used for advanced setups.
    # For instance, serving a model across multiple nodes.
    # For more details: https://github.com/eliran89c/self-hosted-llm-on-eks/multi-node-serving.md 
    # kuberay = {
    #   repository       = "https://ray-project.github.io/kuberay-helm/"
    #   chart            = "kuberay-operator"
    #   version          = "1.1.0"
    #   namespace        = "kuberay-operator"
    #   create_namespace = true
    # }
  }

  tags = local.tags

  depends_on = [kubectl_manifest.karpenter]
}