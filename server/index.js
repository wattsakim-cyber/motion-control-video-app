require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const axios = require('axios');
const Replicate = require('replicate');

const app = express();
const PORT = process.env.PORT || 5000;

// Initialize Replicate client
const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

// Store for tracking jobs
const jobs = new Map();

// Middleware
app.use(cors());
app.use(express.json());
app.use('/uploads', express.static('uploads'));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({ storage });

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Motion Control Video App API is running' });
});

// Upload character image endpoint
app.post('/api/upload/image', upload.single('image'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image file uploaded' });
  }
  const fileUrl = `${req.protocol}://${req.get('host')}/uploads/${req.file.filename}`;
  res.json({
    success: true,
    filename: req.file.filename,
    path: `/uploads/${req.file.filename}`,
    url: fileUrl
  });
});

// Upload motion video endpoint
app.post('/api/upload/video', upload.single('video'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No video file uploaded' });
  }
  const fileUrl = `${req.protocol}://${req.get('host')}/uploads/${req.file.filename}`;
  res.json({
    success: true,
    filename: req.file.filename,
    path: `/uploads/${req.file.filename}`,
    url: fileUrl
  });
});

// Generate motion control video endpoint
app.post('/api/generate', async (req, res) => {
  try {
    const { characterImageUrl, motionVideoUrl, prompt } = req.body;
    
    if (!characterImageUrl || !motionVideoUrl) {
      return res.status(400).json({ error: 'Character image and motion video are required' });
    }

    const jobId = Date.now().toString();
    
    // Start the Replicate prediction
    // Using AnimateDiff-based motion transfer model
    const prediction = await replicate.predictions.create({
      version: "lucataco/animate-diff:1531004ee4c98894ab11f62a7e6b40edd9ccc75c97974f1fd2f3a98ecc8c85f9",
      input: {
        path: motionVideoUrl,
        seed: 255224557,
        steps: 25,
        prompt: prompt || "high quality, best quality",
        n_prompt: "badhandv4, easynegative, ng_deepnegative_v1_75t, verybadimagenegative_v1.3, bad-artist, bad_prompt_version2-neg, teeth",
        motion_module: "mm_sd_v14",
        guidance_scale: 7.5
      }
    });

    // Store job info
    jobs.set(jobId, {
      predictionId: prediction.id,
      status: 'processing',
      characterImageUrl,
      motionVideoUrl,
      prompt,
      createdAt: new Date()
    });

    res.json({
      success: true,
      message: 'Video generation started',
      jobId,
      predictionId: prediction.id,
      status: 'processing'
    });
  } catch (error) {
    console.error('Generation error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Check generation status endpoint
app.get('/api/status/:jobId', async (req, res) => {
  const { jobId } = req.params;
  
  const job = jobs.get(jobId);
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }

  try {
    // Get prediction status from Replicate
    const prediction = await replicate.predictions.get(job.predictionId);
    
    // Update job status
    job.status = prediction.status;
    if (prediction.output) {
      job.videoUrl = prediction.output;
    }
    jobs.set(jobId, job);

    res.json({
      jobId,
      status: prediction.status,
      videoUrl: prediction.output || null,
      error: prediction.error || null
    });
  } catch (error) {
    console.error('Status check error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log('Make sure REPLICATE_API_TOKEN is set in your .env file');
});
