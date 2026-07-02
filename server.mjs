import express from 'express';
import cors from 'cors';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import { MilvusClient, DataType } from '@zilliz/milvus2-sdk-node';
import * as faceapi from 'face-api.js';
import { Canvas, Image, ImageData, loadImage } from 'canvas';

// 1. Give Node.js the ability to read images for face-api
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
app.use(cors());
app.use(express.json());

// Serve the frontend HTML file securely on the root URL
app.get('/', (req, res) => {
  res.sendFile(path.resolve('./index.html'));
});

// 2. Configure Multer to save Postman uploads to an 'uploads' folder
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const dir = './uploads';
    if (!fs.existsSync(dir)) fs.mkdirSync(dir);
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});
const upload = multer({ storage });

const client = new MilvusClient({ address: 'localhost:19530' });
const COLLECTION_NAME = 'cctv_identities_v2';

// 3. Helper to auto-download AI weights from GitHub (Includes Shard Fix)
async function downloadModels() {
  const dir = path.resolve('./models');
  if (!fs.existsSync(dir)) fs.mkdirSync(dir);

  const MODEL_URL = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/';
  const files = [
    'ssd_mobilenetv1_model-weights_manifest.json', 'ssd_mobilenetv1_model-shard1', 'ssd_mobilenetv1_model-shard2',
    'face_landmark_68_model-weights_manifest.json', 'face_landmark_68_model-shard1',
    'face_recognition_model-weights_manifest.json', 'face_recognition_model-shard1', 'face_recognition_model-shard2'
  ];

  for (const file of files) {
    const filePath = path.join(dir, file);
    if (!fs.existsSync(filePath)) {
      console.log(`⬇️ Downloading AI Weight: ${file}...`);
      const response = await fetch(MODEL_URL + file);
      const buffer = await response.arrayBuffer();
      fs.writeFileSync(filePath, Buffer.from(buffer));
    }
  }
}

// 4. Initialize the Complete Backend
async function setupBackend() {
  await downloadModels();

  console.log('🧠 Loading AI Models into memory...');
  const modelsDir = path.resolve('./models');
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelsDir);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelsDir);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelsDir);
  console.log('✅ Models Loaded.');

  const hasCollection = await client.hasCollection({ collection_name: COLLECTION_NAME });
  if (!hasCollection.value) {
    console.log('📦 Creating Milvus Collection...');
    await client.createCollection({
      collection_name: COLLECTION_NAME,
      fields: [
        { name: 'id', data_type: DataType.Int64, is_primary_key: true, autoID: true },
        { name: 'person_name', data_type: DataType.VarChar, max_length: 100 },
        { name: 'photo_path', data_type: DataType.VarChar, max_length: 500 },
        { name: 'face_vector', data_type: DataType.FloatVector, dim: 128 }
      ]
    });
    await client.createIndex({
      collection_name: COLLECTION_NAME,
      field_name: 'face_vector',
      index_type: 'HNSW', metric_type: 'L2', params: { M: 16, efConstruction: 200 }
    });
  }
  await client.loadCollectionSync({ collection_name: COLLECTION_NAME });
  console.log('✅ Milvus Database Ready.');
}
setupBackend();

// ---------------------------------------------------------
// ROUTE 1: ENROLL VIA POSTMAN (IMAGE UPLOAD)
// ---------------------------------------------------------
app.post('/api/enroll', upload.single('image'), async (req, res) => {
  try {
    const name = req.body.name;
    const filePath = req.file.path;

    console.log(`📸 Received upload for: ${name}`);

    const img = await loadImage(filePath);
    const detection = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();

    if (!detection) {
      fs.unlinkSync(filePath);
      return res.status(400).json({ error: "No face detected in the photo." });
    }

    const vector = Array.from(detection.descriptor);

    await client.insert({
      collection_name: COLLECTION_NAME,
      data: [{ person_name: name, photo_path: filePath, face_vector: vector }]
    });
    console.log(`📸 Saved upload for: ${name}  \n ********`);
    res.json({ success: true, message: `${name} enrolled! Photo saved at ${filePath}` });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ---------------------------------------------------------
// ROUTE 2: RECOGNIZE VIA WEBCAM
// ---------------------------------------------------------
app.post('/api/recognize', async (req, res) => {
  const { liveVector } = req.body;
  try {
    const searchResult = await client.search({
      collection_name: COLLECTION_NAME,
      data: [liveVector],
      limit: 1,
      output_fields: ['person_name'],
      metric_type: 'L2'
    });

    const topMatch = searchResult.results[0];
    const captureTime = new Date().toLocaleTimeString();

    if (topMatch && topMatch.score < 0.3) {
      res.json({ recognized: true, name: topMatch.person_name, timestamp: captureTime });
    } else {
      res.json({ recognized: false, name: "Unknown", timestamp: captureTime });
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(3000, () => console.log('🚀 Server running on http://localhost:3000'));