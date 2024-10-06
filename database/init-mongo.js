// init-mongo.js
db = db.getSiblingDB('db_sg');

// Create collections
db.createCollection('cats_images');
