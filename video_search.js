import { 
  AutoTokenizer, 
  AutoProcessor, 
  CLIPTextModelWithProjection, 
  CLIPVisionModelWithProjection, 
  RawImage 
} from '@huggingface/transformers';
import { MilvusClient, DataType } from '@zilliz/milvus2-sdk-node';

const MILVUS_ADDRESS = 'localhost:19530';
const COLLECTION_NAME = 'video_moment_search';
const MODEL_ID = 'Xenova/clip-vit-base-patch32';
const VECTOR_DIMENSION = 512;

const client = new MilvusClient({ address: MILVUS_ADDRESS });
let tokenizer, processor, textModel, visionModel;

async function initModels() {
  console.log('🧠 Loading CLIP Encoders...');
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  processor = await AutoProcessor.from_pretrained(MODEL_ID);
  textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL_ID);
  visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID);
  console.log('✅ Models loaded.');
}

async function getTextEmbedding(text) {
  const textInputs = tokenizer([text], { padding: true, truncation: true });
  const { text_embeds } = await textModel(textInputs);
  return Array.from(text_embeds.data);
}

async function getImageEmbedding(url) {
  const image = await RawImage.fromURL(url);
  const imageInputs = await processor(image);
  const { image_embeds } = await visionModel(imageInputs);
  return Array.from(image_embeds.data);
}

async function main() {
  try {
    await initModels();

    // 2. Rebuild Collection with Timestamp and Video Meta Metadata
    console.log('📦 Rebuilding Milvus collection...');
    const hasCollection = await client.hasCollection({ collection_name: COLLECTION_NAME });
    if (hasCollection.value) {
      await client.dropCollection({ collection_name: COLLECTION_NAME });
    }

    await client.createCollection({
      collection_name: COLLECTION_NAME,
      fields: [
        { name: 'id', data_type: DataType.Int64, is_primary_key: true, autoID: true },
        { name: 'video_url', data_type: DataType.VarChar, max_length: 500 },
        { name: 'timestamp_seconds', data_type: DataType.Int32 }, // Tracks where in the video this frame lives
        { name: 'frame_description', data_type: DataType.VarChar, max_length: 200 },
        { name: 'vector', data_type: DataType.FloatVector, dim: VECTOR_DIMENSION }
      ]
    });

    // 3. Simulating a video breakdown
    // Imagine we uploaded a 30-second drone video of a city coastline. 
    // We extracted frames at 0s, 10s, 20s, and 30s.
    const videoFrames = [
      { timestamp: 0, url: "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=500", desc: "City skyscrapers skyline at dawn" },
      { timestamp: 10, url: "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=500", desc: "Ocean waves crashing on sandy beach" },
      { timestamp: 20, url: "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=500", desc: "Sunlight beaming through deep green forest trees" },
      { timestamp: 30, url: "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=500", desc: "Yosemite valley mountains and river" }
    ];

    console.log('🎞️ Simulating video keyframe processing pipeline...');
    const insertData = [];

    for (const frame of videoFrames) {
      console.log(`   Processing frame at timestamp: ${frame.timestamp}s...`);
      const vector = await getImageEmbedding(frame.url);
      insertData.push({
        video_url: "https://youtu.be/YF_DzoTDO-0?si=eGXd8pChrOYOPpCy",
        timestamp_seconds: frame.timestamp,
        frame_description: frame.desc,
        vector: vector
      });
    }

    console.log('📥 Uploading frame vectors to Milvus...');
    await client.insert({ collection_name: COLLECTION_NAME, data: insertData });

    // Index and load
    await client.createIndex({
      collection_name: COLLECTION_NAME,
      field_name: 'vector',
      index_type: 'HNSW',
      metric_type: 'COSINE',
      params: { M: 16, efConstruction: 200 }
    });
    await client.loadCollectionSync({ collection_name: COLLECTION_NAME });

    // 4. Perform Search for a Moment in Time
    const userQuery = "motorcycle accident on highway";
    console.log(`\n🔍 Text Query: "${userQuery}"`);

    const queryVector = await getTextEmbedding(userQuery);

    const searchResult = await client.search({
      collection_name: COLLECTION_NAME,
      data: [queryVector],
      limit: 1,
      output_fields: ['video_url', 'timestamp_seconds', 'frame_description'],
      metric_type: 'COSINE'
    });

    const match = searchResult.results[0];
    console.log(`\n🎯 Match Found!`);
    console.log(`📹 Video Source: ${match.video_url}`);
    console.log(`⏱️ Action happens at: ${match.timestamp_seconds} seconds!`);
    console.log(`📝 Visual description context: "${match.frame_description}"`);
    console.log(`📈 Score: ${(match.score * 100).toFixed(2)}%`);

  } catch (error) {
    console.error('❌ Pipeline failure:', error);
  } finally {
    await client.closeConnection();
  }
}

main();