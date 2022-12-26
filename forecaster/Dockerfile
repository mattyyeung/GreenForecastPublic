# This is old version - based on AWS image, didn't work with WSL2 because of some certificate issue.
FROM public.ecr.aws/lambda/python:3.9

# Install the function's dependencies
COPY requirements_forecaster.txt  .
RUN  pip3 install -r requirements_forecaster.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code, input data and models
RUN mkdir ${LAMBDA_TASK_ROOT}/data
COPY data ${LAMBDA_TASK_ROOT}/data
RUN mkdir ${LAMBDA_TASK_ROOT}/models
COPY models ${LAMBDA_TASK_ROOT}/models
RUN mkdir ${LAMBDA_TASK_ROOT}/download_cache
COPY *.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "forecaster.main" ] 