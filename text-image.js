import { 
  AutoTokenizer, 
  AutoProcessor, 
  CLIPTextModelWithProjection, 
  CLIPVisionModelWithProjection, 
  RawImage 
} from '@huggingface/transformers';
import { MilvusClient, DataType } from '@zilliz/milvus2-sdk-node';

// 1. Configuration Constants
const MILVUS_ADDRESS = 'localhost:19530';
const COLLECTION_NAME = 'cross_modal_gallery';
const MODEL_ID = 'Xenova/clip-vit-base-patch32';
const VECTOR_DIMENSION = 512; // CLIP-base models output 512-dimensional spaces

const client = new MilvusClient({ address: MILVUS_ADDRESS });

// Global holders for our model instances
let tokenizer, processor, textModel, visionModel;

// Helper to initialize the machine learning components
async function initModels() {
  console.log('🧠 Loading CLIP Multimodal Encoders (this may take a minute on first run)...');
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  processor = await AutoProcessor.from_pretrained(MODEL_ID);
  textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL_ID);
  visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID);
  console.log('✅ Models fully loaded into memory.');
}

// Extract vector from a text string
async function getTextEmbedding(text) {
  const textInputs = tokenizer([text], { padding: true, truncation: true });
  const { text_embeds } = await textModel(textInputs);
  // Convert ONNX tensor output to a regular JS array and normalize it
  return Array.from(text_embeds.data);
}

// Extract vector from an image URL
async function getImageEmbedding(url) {
  const image = await RawImage.fromURL(url);
  const imageInputs = await processor(image);
  const { image_embeds } = await visionModel(imageInputs);
  return Array.from(image_embeds.data);
}

async function main() {
  try {
    await initModels();

    // 2. Sample Image Dataset (Public URLs representing completely different subjects)
    const mediaCatalog = [
      { id: 1, url: "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=500", label: "Sunny tropical beach scene" },
      { id: 2, url: "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=500", label: "A bright red running shoe sitting on a dark surface" },
      { id: 3, url: "https://images.unsplash.com/photo-1472396961693-142e6e269027?w=500", label: "A wild deer standing in a misty morning forest" },
      { id: 4, url: "https://images.unsplash.com/photo-1511512578047-dfb367046420?w=500", label: "A modern black gaming console controller" }
    ];

    // 3. Rebuild Milvus Collection with 512 Dimensions
    console.log('📦 Rebuilding Milvus collection...');
    const hasCollection = await client.hasCollection({ collection_name: COLLECTION_NAME });
    if (hasCollection.value) {
      await client.dropCollection({ collection_name: COLLECTION_NAME });
    }

    await client.createCollection({
      collection_name: COLLECTION_NAME,
      fields: [
        { name: 'id', data_type: DataType.Int64, is_primary_key: true },
        { name: 'file_url', data_type: DataType.VarChar, max_length: 500 },
        { name: 'description_hint', data_type: DataType.VarChar, max_length: 200 },
        { name: 'vector', data_type: DataType.FloatVector, dim: VECTOR_DIMENSION }
      ]
    });

    // 4. Process and Insert Images
    console.log('🖼️ Running image catalog through CLIP Vision Encoder...');
    const insertData = [];
    
    for (const item of mediaCatalog) {
      console.log(`   Processing asset #${item.id}...`);
      const vector = await getImageEmbedding(item.url);
      insertData.push({
        id: item.id,
        file_url: item.url,
        description_hint: item.label,
        vector: vector
      });
    }

    console.log('📥 Uploading vision vectors to Milvus...');
    await client.insert({ collection_name: COLLECTION_NAME, data: insertData });

    // 5. Index & Load
    await client.createIndex({
      collection_name: COLLECTION_NAME,
      field_name: 'vector',
      index_type: 'HNSW',
      metric_type: 'COSINE',
      params: { M: 16, efConstruction: 200 }
    });
    await client.loadCollectionSync({ collection_name: COLLECTION_NAME });

    // 6. Test Cross-Modal Searches
    // Notice how these text strings don't exactly match the image descriptions!
    const testQueries = [
      "i Want to swim"
    ];

    for (const query of testQueries) {
      console.log(`\n🔍 Input Text Query: "${query}"`);
      
      // Convert TEXT to a vector using the text model
      const queryVector = await getTextEmbedding(query);

      // Search IMAGE vectors inside Milvus
      const searchResult = await client.search({
        collection_name: COLLECTION_NAME,
        data: [queryVector],
        limit: 1,
        output_fields: ['description_hint', 'file_url'],
        metric_type: 'COSINE'
      });

      const topMatch = searchResult.results[0];
      console.log(`🎯 Best Matching Image: "${topMatch.description_hint}"`);
      console.log(`🔗 Asset Link: ${topMatch.file_url}`);
      console.log(`📈 Match Score: ${(topMatch.score * 100).toFixed(2)}%`);
    }

  } catch (error) {
    console.error('❌ Pipeline failure:', error);
  } finally {
    await client.closeConnection();
  }
}

main();