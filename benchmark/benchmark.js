const axios = require('axios');
const fs = require('fs');
const path = require('path');
const FormData = require('form-data');

// Directory containing images
const imageDir = path.join(__dirname, 'images_input');
// Directory to store results
const resultsDir = path.join(__dirname, 'results');

// Ensure the results directory exists
if (!fs.existsSync(resultsDir)) {
    fs.mkdirSync(resultsDir);
}

// FastAPI predict endpoint URL and authentication URL
const API_URL = 'http://localhost:8082/predict';
const TOKEN_URL = 'http://localhost:8080/token';

// User credentials for login
const USERNAME = 'theo'; 
const PASSWORD = 'pwd_theo';

// Get the wait time from command-line arguments
const waitTime = process.argv[2] ? parseFloat(process.argv[2]) : 0;  // Default to 0 if not provided

// Function to get the authentication token
async function getAuthToken() {
    try {
        const response = await axios.post(TOKEN_URL, new URLSearchParams({
            username: USERNAME,
            password: PASSWORD
        }), {
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        });

        if (response.data && response.data.access_token) {
            console.log("Authentication successful. Token received.");
            return response.data.access_token;
        } else {
            console.error("Failed to get token. Response:", response.data);
            return null;
        }
    } catch (error) {
        console.error("Error during authentication:", error.message);
        return null;
    }
}

// Function to get the next results file name
function getNextResultFileName() {
    const existingFiles = fs.readdirSync(resultsDir);
    let iterationNumber = 1;

    while (existingFiles.includes(`results_iteration_${iterationNumber}.json`)) {
        iterationNumber++;
    }

    return path.join(resultsDir, `results_iteration_${iterationNumber}.json`);
}

// Function to send image to the /predict endpoint with wait_time as a query parameter
async function sendImage(imagePath, token, waitTime) {
    try {
        const imageStream = fs.createReadStream(imagePath);
        const formData = new FormData();
        formData.append('image', imageStream);

        const response = await axios.post(`${API_URL}?wait_time=${waitTime}`, formData, {
            headers: {
                'Authorization': `Bearer ${token}`,
                ...formData.getHeaders()  // Use formData's getHeaders for proper multipart headers
            },
            maxContentLength: Infinity,
            maxBodyLength: Infinity,
        });

        // Return the predicted data and the success status
        return {
            success: response.status === 200,
            data: response.data  // Capture the predicted class index, label, and model inference time
        };
    } catch (error) {
        console.error(`Error uploading ${imagePath}:`, error.message);
        return { success: false };
    }
}

// Function to store results in the results_iteration_N.json file
function storeResults(resultFilePath, imageName, resultData) {
    let results = {};

    // If the file exists, read the existing results
    if (fs.existsSync(resultFilePath)) {
        const fileContents = fs.readFileSync(resultFilePath);
        results = JSON.parse(fileContents);
    }

    // Add the new result
    results[imageName] = {
        predicted_class_idx: resultData.predicted_class_idx,
        predicted_class_label: resultData.predicted_class_label,
        model_inference_time_ms: (resultData.model_inference_time * 1000).toFixed(2)  // Store model inference time in milliseconds
    };

    // Write the updated results back to the file
    fs.writeFileSync(resultFilePath, JSON.stringify(results, null, 2));
    console.log(`Result for ${imageName} saved to ${resultFilePath}`);
}

// Benchmark function that sends images concurrently
async function benchmark(waitTime) {
    // Get the authentication token
    const token = await getAuthToken();
    if (!token) {
        console.error("Unable to proceed without authentication.");
        return;
    }

    const files = fs.readdirSync(imageDir);
    const totalFiles = files.length;

    let totalTime = 0;
    let successCount = 0;

    // Get the next results file name
    const resultFilePath = getNextResultFileName();

    console.log(`Starting benchmark for ${totalFiles} images with wait_time=${waitTime} seconds...`);

    // Create a list of promises to send all images concurrently
    const promises = files.map(async (file) => {
        const imagePath = path.join(imageDir, file);
        const startTime = Date.now();  // Start measuring total request time

        const result = await sendImage(imagePath, token, waitTime);
        const timeTaken = Date.now() - startTime;  // Total request time

        totalTime += timeTaken;

        if (result.success) {
            successCount++;

            // Store the prediction result in the results file
            storeResults(resultFilePath, file, result.data);
        }

        console.log(`Processed ${file}: ${result.success ? 'Success' : 'Failed'} (Total Time: ${timeTaken} ms, Model Inference Time: ${result.data ? (result.data.model_inference_time * 1000).toFixed(2) : 'N/A'} ms)`);
    });

    // Wait for all promises (image uploads) to complete
    await Promise.all(promises);

    // Calculate success rate, total time, and average time per image
    const successRate = (successCount / totalFiles) * 100;
    const averageTimePerImage = totalFiles > 0 ? (totalTime / totalFiles) : 0;

    console.log(`\nBenchmark Complete!`);
    console.log(`Total Files: ${totalFiles}`);
    console.log(`Success Rate: ${successRate.toFixed(1)}%`);
    console.log(`Total Time: ${totalTime / 1000} seconds`);
    console.log(`Average Total Time per Image: ${averageTimePerImage.toFixed(0) / 1000} seconds`);
}

// Run the benchmark with the passed wait time
benchmark(waitTime);
