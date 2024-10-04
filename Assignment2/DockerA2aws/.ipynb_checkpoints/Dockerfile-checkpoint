FROM public.ecr.aws/lambda/python:3.8

RUN pip install tensorflow matplotlib

COPY image_classification.py ${LAMBDA_TASK_ROOT}

CMD ["image_classification.lambda_handler"]
