import { 
  AutoTokenizer, 
  AutoProcessor, 
  CLIPTextModelWithProjection, 
  CLIPVisionModelWithProjection, 
  RawImage 
} from '@huggingface/transformers';
import { MilvusClient, DataType } from '@zilliz/milvus2-sdk-node';
import ffmpeg from 'fluent-ffmpeg';
import fs from 'fs';
import path from 'path';
import ffmpegInstaller from '@ffmpeg-installer/ffmpeg';
ffmpeg.setFfmpegPath(ffmpegInstaller.path);

// 1. Core System Configuration
const MILVUS_ADDRESS = 'localhost:19530';
const COLLECTION_NAME = 'dynamic_video_analytics';
const MODEL_ID = 'Xenova/clip-vit-base-patch32';
const VECTOR_DIMENSION = 512;
const FRAMES_DIR = path.resolve('./temp_frames');

// INPUT: Place your downloaded 30-second accident MP4 video file here
const INPUT_VIDEO_PATH = path.resolve('./accident_video.mp4'); 

const client = new MilvusClient({ address: MILVUS_ADDRESS });
let tokenizer, processor, textModel, visionModel;

async function initModels() {
  console.log('🧠 Loading CLIP Multimodal Models...');
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  processor = await AutoProcessor.from_pretrained(MODEL_ID);
  textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL_ID);
  visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID);
  console.log('✅ Machine learning models ready.');
}

// 2. Step 1 of the Pipeline: Dynamic Frame Extraction
function extractFramesAsync(videoPath, outputDir) {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    } else {
      // Clear out any old frames from a previous run
      fs.readdirSync(outputDir).forEach(file => fs.unlinkSync(path.join(outputDir, file)));
    }

    console.log('🎞️ FFmpeg cutting video into 1-second keyframes...');
    ffmpeg(videoPath)
      .output(path.join(outputDir, 'frame_%03d.jpg'))
      .outputOptions('-vf', 'fps=1') // Extract exactly 1 frame per second
      .on('end', () => {
        console.log('✅ Frame extraction complete.');
        resolve();
      })
      .on('error', (err) => {
        reject(err);
      })
      .run();
  });
}

async function main() {
  try {
    // Ensure the local video asset actually exists before running
    if (!fs.existsSync(INPUT_VIDEO_PATH)) {
      throw new Error(`Missing video asset file at: ${INPUT_VIDEO_PATH}. Please download the video and save it there first.`);
    }

    await initModels();
    await extractFramesAsync(INPUT_VIDEO_PATH, FRAMES_DIR);

    // 3. Database Ingestion Layer
    console.log('📦 Dropping and recreating Milvus collection...');
    const hasCollection = await client.hasCollection({ collection_name: COLLECTION_NAME });
    if (hasCollection.value) {
      await client.dropCollection({ collection_name: COLLECTION_NAME });
    }

    await client.createCollection({
      collection_name: COLLECTION_NAME,
      fields: [
        { name: 'id', data_type: DataType.Int64, is_primary_key: true, autoID: true },
        { name: 'timestamp_seconds', data_type: DataType.Int32 },
        { name: 'file_path', data_type: DataType.VarChar, max_length: 500 },
        { name: 'vector', data_type: DataType.FloatVector, dim: VECTOR_DIMENSION }
      ]
    });

    // 4. Dynamic Directory Scanning and Vector Encoding
    const files = fs.readdirSync(FRAMES_DIR).filter(f => f.endsWith('.jpg'));
    const insertData = [];

    console.log(`🧠 Loop processing ${files.length} frames through CLIP Vision Encoder...`);
    for (const file of files) {
      const filePath = path.join(FRAMES_DIR, file);
      
      // Parse out the timestamp from the filename (frame_001.jpg -> 1 second, frame_015.jpg -> 15 seconds)
      const frameIndex = parseInt(file.replace('frame_', '').replace('.jpg', ''), 10);
      const timestampSeconds = frameIndex - 1; 

      // Read local image directly into memory using Transformers.js RawImage
      const image = await RawImage.read(filePath);
      const imageInputs = await processor(image);
      const { image_embeds } = await visionModel(imageInputs);
      const vector = Array.from(image_embeds.data);

      insertData.push({
        timestamp_seconds: timestampSeconds,
        file_path: filePath,
        vector: vector
      });
      
      console.log(`   Encoded timestamp: ${timestampSeconds}s -> Vector mapped.`);
    }

    console.log('📥 Batch uploading local vectors to Milvus database...');
    await client.insert({ collection_name: COLLECTION_NAME, data: insertData });

    // Index & Load
    await client.createIndex({
      collection_name: COLLECTION_NAME,
      field_name: 'vector',
      index_type: 'HNSW',
      metric_type: 'COSINE',
      params: { M: 16, efConstruction: 200 }
    });
    await client.loadCollectionSync({ collection_name: COLLECTION_NAME });

    // 5. Query the Real Dynamic Data State
    const userQuery = "when did bike and car collide";
    console.log(`\n🔍 Searching Video Timeline for: "${userQuery}"`);

    const textInputs = tokenizer([userQuery], { padding: true, truncation: true });
    const { text_embeds } = await textModel(textInputs);
    const queryVector = Array.from(text_embeds.data);

    const searchResult = await client.search({
      collection_name: COLLECTION_NAME,
      data: [queryVector],
      limit: 3, // Look at the top 3 closest matches in time
      output_fields: ['timestamp_seconds', 'file_path'],
      metric_type: 'COSINE'
    });

    console.log('\n🎯 Identified High-Probability Video Moments:');
    searchResult.results.forEach((match, index) => {
      console.log(`[Rank ${index + 1}] Similarity Score: ${(match.score * 100).toFixed(2)}%`);
      console.log(`     Exact Time in Video: ${match.timestamp_seconds} seconds`);
      console.log(`     Local Reference Frame Path: ${match.file_path}\n`);
    });

  } catch (error) {
    console.error('❌ Dynamic Pipeline Error:', error.message);
  } finally {
    await client.closeConnection();
  }
}

main();