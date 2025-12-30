# vLLM ARC Testing

This file is created to trigger the ARC-based vLLM testing workflow.

Testing with:

- EKS cluster with g5.12xlarge nodes
- Actions Runner Controller (ARC)
- Public ECR image: `public.ecr.aws/deep-learning-containers/vllm:0.13.0-gpu-py312-ec2`

Test suites included:

- EC2 regression, CUDA, and example tests
- RayServe regression, CUDA, and example tests
- SageMaker regression, CUDA, example, and endpoint tests
  
