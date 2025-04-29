FROM jupyter/pyspark-notebook

ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3


RUN pip install flask flask-cors
RUN pip install numpy
RUN pip install pyspark

WORKDIR /app

COPY infer.py /app
COPY trainingweights /app/trainingweights

EXPOSE 5000

CMD ["python", "infer.py"]