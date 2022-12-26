
.PHONY: default fc-clean fc-setup fc-test fc-build fc-build-run fc-run-container fc-test-container fc-deploy fc-build-deploy website-deploy make website-deploy-test


default:
	$(info No default, run one of these:)
	$(info - make fc-clean    : clean forecaster folder)
	$(info - make fc-setup    : populate forecaster folder with models/data so it can be run locally or built for prod)
	$(info - make fc-test     : Test forecaster.py locally, not in container image)
	$(info - make fc-build    : build the container image for AWS lambda that runs the forecaster)
	$(info - make fc-build-run: fc-build and fc-run-container)
	$(info - make fc-run-container: run the container image locally for testing)
	$(info - make fc-test-container: test the generated forecaster container image - runs forecaster.py in container)
	$(info - make fc-deploy   : deploy the container image to AWS)
	$(info - make fc-build-deploy: fc-build and fc-deploy)
	$(info - make website-deploy: deploy the static website files to the poduction S3 bucket)
	$(info - make website-deploy-test: deploy the static website files to the test S3 bucket)

##########################################
### Forecaster

fc-clean:
	-rm ./forecaster/latest_forecasts.json
	-mkdir ./forecaster/data
	-rm -rf ./forecaster/models

# populate forecaster folder with models & data so it can be run locally or built for prod
fc-setup:
	cp ./dataset_generator/aemo.py ./forecaster/aemo.py
	cp ./data/australian_public_holidays_scraped_2010-2024.csv ./forecaster/data/australian_public_holidays_scraped_2010-2024.csv
	-rm -rf ./forecaster/models
	cp -r ./models/latest ./forecaster/models

# Test forecaster.py locally (not in container image)
fc-test:
	cd forecaster; python forecaster.py

# make the container image
# if getting error put a sudo in front of docker build 
fc-build:
	$(info )
	$(info Building forecaster container image)
	$(info Note: the docker daemon needs to be running. In WSL: `sudo dockerd` in a separate WSL console *started as admin*.)
	docker build -t greenforecast_forecaster forecaster

# run the forecaster container image locally for testing
# --rm means the container is cleaned up after it stops running... remove this if need to look over the corpse.
# --read-only --tmpfs /tmp  is to mimic lambda - read only filesystem, excpet /tmp
# if getting error put a sudo in front of docker run 
AWS_KEY := $(shell aws --profile default configure get aws_access_key_id)
AWS_SECRET := $(shell aws --profile default configure get aws_secret_access_key)
fc-run-container:
	$(info Starting container image. Unless there's an error, this won't return - open new tab to run make fc-test)
	docker run \
			--rm \
			--read-only --tmpfs /tmp \
			-e RUNNING_LOCALLY=1 \
			-e AWS_ACCESS_KEY_ID=$(AWS_KEY) -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET) \
			-e AWS_LAMBDA_FUNCTION_TIMEOUT=900 \
			-e GREENFORECAST_TEST=1 \
			-p 9000:8080 \
			greenforecast_forecaster:latest

fc-build-run: fc-build fc-run-container

# test the generated forecaster container image - runs forecaster.py in container)
fc-test-container:
	$(info Running test of the forecaster.)
	$(info Note: container needs to be running. Do this with make fc-run-container)
	curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{}'

# deploys the container image to ECR. First, tag the docker image with the right name, then actually push it with docker push. 
fc-deploy:
	$(info Uploading greenforecast_forecaster to AWS ECR)
	aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 856549827017.dkr.ecr.ap-southeast-2.amazonaws.com
	docker tag  greenforecast_forecaster:latest 856549827017.dkr.ecr.ap-southeast-2.amazonaws.com/greenforecast_ecr:latest
	docker push 856549827017.dkr.ecr.ap-southeast-2.amazonaws.com/greenforecast_ecr:latest
	$(info Now, update the lambda function so it uses the new version of the container image.)
	$(info The lifecycle policy in ECR will delete the old image)
	aws lambda update-function-code --region ap-southeast-2 --function-name greenforecast_forecaster --image-uri 856549827017.dkr.ecr.ap-southeast-2.amazonaws.com/greenforecast_ecr:latest

fc-build-deploy: fc-build fc-deploy

##########################################
### Website

# Upload the website folder to the website bucket:
website-deploy:
	aws s3 sync website s3://greenforecast.au --exclude "*.DS_Store" --exclude "latest_forecasts*"

# Upload the website folder to the website bucket:
website-deploy-test:
	aws s3 sync website s3://greenforecast.test --exclude "*.DS_Store" --exclude "latest_forecasts*"



