# Mistral 7B BentoML Deployment

This directory contains the complete deployment automation for deploying a 4-bit quantized Mistral 7B model using BentoML on Google Cloud Platform.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Client/User   │    │   GCP VM     │    │  Docker         │
│                 │    │              │    │  Container      │
│ HTTP Requests   ├───►│ Nginx Proxy  ├───►│                 │
│                 │    │ (Port 80)    │    │ BentoML Service │
│                 │    │              │    │ (Port 3000)     │
└─────────────────┘    └──────────────┘    └─────────────────┘
                             │
                             ▼
                       ┌──────────────┐
                       │ Quantized    │
                       │ Mistral 7B   │
                       │ (4-bit)      │
                       └──────────────┘
```

## Features

- **4-bit Quantized Model**: Reduces memory usage by ~75% (7GB → ~2GB)
- **CPU Optimized**: Designed for GCP free tier without GPU
- **Docker Containerized**: Clean isolation and easy management
- **Ansible Automated**: Infrastructure as Code deployment
- **UV Package Management**: Ultra-fast dependency resolution and installation
- **Production Ready**: Includes monitoring, logging, and health checks
- **Nginx Reverse Proxy**: Rate limiting and SSL-ready

## UV Package Management Benefits

This deployment uses **UV** (ultra-fast Python package manager) instead of pip:

- **10-100x faster** dependency resolution and installation
- **Deterministic builds** with lockfile support
- **Better error messages** and conflict resolution
- **Cross-platform compatibility** and caching
- **Drop-in pip replacement** with better UX

## Prerequisites

1. **GCP VM with static IP** (already set up)
2. **Local machine with**:
   - Python 3.8+
   - Ansible installed (`pip install ansible`)
   - SSH access to your GCP VM
3. **Project Dependencies**: Managed via `pyproject.toml` and `uv.lock`

## Quick Start

1. **Configure your deployment**:
   ```bash
   cd src/deploy
   cp inventory.yml inventory.yml.backup
   # Edit inventory.yml with your VM details
   ```

2. **Update inventory.yml**:
   ```yaml
   all:
     hosts:
       gcp-vm:
         ansible_host: YOUR_ACTUAL_STATIC_IP  # Replace this
         ansible_user: YOUR_VM_USERNAME       # Replace this
         ansible_ssh_private_key_file: ~/.ssh/your_key  # Update path
   ```

3. **Deploy everything**:
   ```bash
   ./deploy.sh
   ```

4. **Test your deployment**:
   ```bash
   curl http://YOUR_STATIC_IP:3000/health
   ```

## Manual Deployment Steps

If you prefer to run steps individually:

### Step 1: Docker Setup
```bash
ansible-playbook -i inventory.yml playbooks/setup-docker.yml
```

### Step 2: BentoML Deployment (with UV)
```bash
ansible-playbook -i inventory.yml playbooks/deploy-bentoml.yml
```

This step will:
- Install UV (ultra-fast Python package manager)
- Copy your entire project to the VM
- Run `uv sync` to install all dependencies from pyproject.toml
- Build the BentoML service using `uv run bentoml build`

### Step 3: Network Configuration
```bash
ansible-playbook -i inventory.yml playbooks/configure-network.yml
```

## API Usage

### Health Check
```bash
curl http://YOUR_STATIC_IP:3000/health
```

### Simple Text Generation
```bash
curl -X POST http://YOUR_STATIC_IP:3000/quick_generate \
  -d "What is artificial intelligence?"
```

### Advanced Generation with Parameters
```bash
curl -X POST http://YOUR_STATIC_IP:3000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain machine learning in simple terms:",
    "max_length": 512,
    "temperature": 0.7
  }'
```

### API Documentation
Visit `http://YOUR_STATIC_IP:3000/docs` for interactive API documentation.

## Configuration

### Resource Limits
Edit `vars/main.yml` to adjust:
- `memory_limit`: Container memory limit (default: 4g)
- `cpu_limit`: CPU cores (default: 2)
- `host_port`: External port (default: 3000)

### Model Configuration
- **Model**: TheBloke/Mistral-7B-Instruct-v0.1-GPTQ (4-bit quantized)
- **Memory Usage**: ~2-3GB total
- **Inference Time**: 10-30 seconds per request
- **Concurrent Requests**: 1-2 maximum (limited by CPU)

## Monitoring

### Container Logs
```bash
# On your VM
docker logs mistral-7b-service -f
```

### System Resources
```bash
# Monitor memory and CPU usage
docker stats mistral-7b-service
```

### Nginx Logs
```bash
# On your VM
tail -f /var/log/nginx/mistral-7b-service_access.log
tail -f /var/log/nginx/mistral-7b-service_error.log
```

## Troubleshooting

### Service Won't Start
1. Check Docker logs: `docker logs mistral-7b-service`
2. Verify memory limits: `free -h`
3. Check port conflicts: `netstat -tlnp | grep 3000`

### Out of Memory Errors
1. Reduce `max_length` in API calls
2. Limit concurrent requests
3. Increase VM memory or enable swap

### Slow Response Times
1. Normal for CPU inference (10-30s expected)
2. Reduce `max_length` for faster responses
3. Consider using smaller model variants

### Connection Issues
1. Check firewall: `sudo ufw status`
2. Verify nginx: `sudo nginx -t && sudo systemctl status nginx`
3. Test locally: `curl localhost:3000/health`

## File Structure

```
src/deploy/
├── README.md                    # This documentation
├── deploy.sh                    # Main deployment script
├── ansible.cfg                  # Ansible configuration
├── inventory.yml                # VM connection details
├── docker-compose.yml           # Container orchestration
├── playbooks/
│   ├── setup-docker.yml         # Docker installation
│   ├── deploy-bentoml.yml       # BentoML deployment
│   └── configure-network.yml    # Firewall and nginx
├── vars/
│   └── main.yml                 # Deployment variables
└── templates/
    ├── .env.j2                  # Environment variables
    └── nginx.conf.j2            # Nginx configuration
```

## Security Considerations

- Firewall configured to allow only necessary ports
- Nginx rate limiting (10 requests/second)
- No sensitive data in containers
- Regular security updates recommended

## Cost Optimization

- Model automatically cached to avoid re-downloading
- Container uses resource limits to prevent overuse
- Nginx compression reduces bandwidth
- Logs rotated to save disk space

## Support

For issues or questions:
1. Check logs first (see Monitoring section)
2. Review troubleshooting guide above
3. Verify all prerequisites are met
4. Test connectivity with `ansible all -m ping`