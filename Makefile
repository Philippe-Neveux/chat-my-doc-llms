run_api:
	uv run fastapi run src/api/main.py

run_api_dev:
	uv run fastapi dev src/api/main.py

GCP_REGION ?= australia-southeast1
GCP_PROJECT_ID ?= gen-ai-466406
GCP_ARTIFACT_REPOSITORY ?= api-images
DOCKER_IMAGE_NAME ?= llm-gateway

build_docker_image:
	docker image build --no-cache . --tag $(DOCKER_IMAGE_NAME):latest

tag_docker_image_for_gcp:
	docker tag $(DOCKER_IMAGE_NAME):latest $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(GCP_ARTIFACT_REPOSITORY)/$(DOCKER_IMAGE_NAME):latest

push_docker_image_to_gcp:
	docker push $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT_ID)/$(GCP_ARTIFACT_REPOSITORY)/$(DOCKER_IMAGE_NAME):latest