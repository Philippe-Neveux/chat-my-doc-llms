# Chat My Doc LLMs

A FastAPI-based application that provides a chat interface with Google's Gemini LLM model for document interaction and AI-powered conversations.

## Features

- FastAPI web API with health check endpoints
- Integration with Google Gemini 2.0 Flash model via Langchain
- Dockerized deployment with Cloud Run support
- Automated CI/CD pipeline with GitHub Actions

## Prerequisites

- Python 3.12+
- Docker (for containerized deployment)
- Google Cloud Platform account (for deployment)
- Google API key for Gemini access

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd chat-my-doc-llms
```

2. Install dependencies using uv:
```bash
uv sync --locked
```

3. Set up environment variables:
```bash
export GOOGLE_API_KEY=your_google_api_key
```

## Usage

### Local Development

Start the development server:
```bash
make run_api_dev
# or
uv run fastapi dev src/api/main.py
```

Start the production server:
```bash
make run_api
# or
uv run fastapi run src/api/main.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /` - Root endpoint returning a welcome message
- `GET /health` - Health check endpoint
- `GET /gemini?prompt=<your_prompt>` - Chat with Gemini model

### Example Usage

```bash
curl "http://localhost:8000/gemini?prompt=Hello, how are you?"
```

## Docker Deployment

### Build and Run Locally

```bash
make build_docker_image
docker run -p 8000:8000 -e GOOGLE_API_KEY=your_key llm-gateway:latest
```

### Google Cloud Run Deployment

The project includes automated deployment to Google Cloud Run via GitHub Actions.

#### Prerequisites

1. Set up Google Cloud Project
2. Enable Cloud Run API
3. Create a service account with appropriate permissions
4. Set up GitHub secrets and variables:

**Secrets:**
- `GCP_SA_KEY` - Service account JSON key
- `GCP_PROJECT_ID` - Your Google Cloud project ID
- `GOOGLE_API_KEY` - Your Google API key

**Variables:**
- `GCP_REGION` - Deployment region (default: australia-southeast1)
- `SERVICE_NAME` - Cloud Run service name

#### Manual Deployment

```bash
# Build and tag image
make build_docker_image
make tag_docker_image_for_gcp

# Push to Google Artifact Registry
make push_docker_image_to_gcp
```

## Project Structure

```
chat-my-doc-llms/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI application
│   └── chat_my_doc_llms/
│       ├── __init__.py
│       └── main.py              # Gemini chat functionality
├── .github/
│   └── workflows/
│       └── deploy_cloud_run.yaml # CI/CD pipeline
├── Dockerfile                   # Container configuration
├── Makefile                     # Build and deployment commands
├── pyproject.toml              # Python project configuration
└── uv.lock                     # Dependency lock file
```

## Dependencies

- **FastAPI** - Modern web framework for building APIs
- **Langchain** - Framework for developing applications with LLMs
- **Google GenAI** - Google's generative AI models integration

## Environment Variables

- `PORT` - Server port (default: 8000)
- `GOOGLE_API_KEY` - Required for Gemini model access

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure tests pass
5. Submit a pull request

## License

[Add your license information here]