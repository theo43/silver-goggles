# silver-goggles
Deploy locally an end-to-end machine learning application using Docker. It deploys an image classification pretrained model, ResNet50 and allows classifying images on a large number of classes (model trained on Image50 dataset).

## How to run
From application root folder run: 
```
docker-compose up --build -d
```

Fetching an authentication token is mandatory: please refer to http://localhost:8080/docs for documentation.

You can then classify images using http://localhost:8082/predict. Please refer to documentation on http://localhost:8082/docs.
