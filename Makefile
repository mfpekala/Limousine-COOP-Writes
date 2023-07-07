DOCKER_CMD = docker

BASE_DIR := $(shell pwd)
WORK_DIR := buffered-pgm

build:
	${DOCKER_CMD} build -t buffered-pgm:1.0 .

startcontainer:
	$(eval DOCKER_CONT_ID := $(shell docker container run \
		-v $(BASE_DIR)/src:/$(WORK_DIR)/src \
		-d --rm -t --privileged -i buffered-pgm:1.0 bash))
	echo $(DOCKER_CONT_ID) > status.current_container_id

stopcontainer:
	$(eval DOCKER_CONT_ID := $(shell cat status.current_container_id | awk '{print $1}'))
	$(DOCKER_CMD) stop $(DOCKER_CONT_ID)
	rm status.current_container_id

shell:
	@$(eval DOCKER_CONT_ID := $(shell cat status.current_container_id | awk '{print $1}'))
	$(DOCKER_CMD) exec -it $(DOCKER_CONT_ID) bash
