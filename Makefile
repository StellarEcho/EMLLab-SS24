.PHONY: image

image:
	docker build -f docker/Dockerfile --network host -t python3-gpu-$(USER)  docker
