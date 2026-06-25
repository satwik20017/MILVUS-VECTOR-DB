import { pipeline } from '@huggingface/transformers';
import { MilvusClient, DataType } from '@zilliz/milvus2-sdk-node';

// 1. Initialize the Milvus client pointing to your local Docker instance
const MILVUS_ADDRESS = 'localhost:19530';
const COLLECTION_NAME = 'local_text_search';
const client = new MilvusClient({ address: MILVUS_ADDRESS });

// We use an ultra-lightweight text embedding model that generates 384-dimensional vectors
const MODEL_NAME = 'Xenova/all-MiniLM-L6-v2';
const VECTOR_DIMENSION = 384; 

let embeddingPipeline = null;

// Helper function to generate vector embeddings from raw text
async function getEmbedding(text) {
  if (!embeddingPipeline) {
    // Downloads and caches the model locally on the first run
    embeddingPipeline = await pipeline('feature-extraction', MODEL_NAME);
  }
  
  const output = await embeddingPipeline(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

async function main() {
  try {
    console.log('🔄 Initializing local embedding pipeline...');
    
    // 2. Define data to index
    const sampleDocuments = [
      { id: 1, text: "The swift development of artificial intelligence systems is changing software development forever." },
      { id: 2, text: "A beautiful red sports car was seen zooming down the highway during a heavy rainstorm." },
      { id: 3, text: "Classic Italian recipes rely heavily on fresh tomatoes, garlic, olive oil, and fresh basil." },
      { id: 4, text: "Node.js uses an event-driven, non-blocking I/O model that makes it lightweight and efficient." }
    ];

    // 3. Prepare the Milvus Collection Schema
    console.log('📦 Checking Milvus collection status...');
    const hasCollection = await client.hasCollection({ collection_name: COLLECTION_NAME });
    
    if (hasCollection.value) {
      await client.dropCollection({ collection_name: COLLECTION_NAME });
    }

    await client.createCollection({
      collection_name: COLLECTION_NAME,
      fields: [
        { name: 'id', data_type: DataType.Int64, is_primary_key: true },
        { name: 'text', data_type: DataType.VarChar, max_length: 500 },
        { name: 'vector', data_type: DataType.FloatVector, dim: VECTOR_DIMENSION }
      ]
    });

    // 4. Transform raw text into mathematical vectors and insert them
    console.log('🧠 Generating embeddings for sample documents...');
    const insertData = [];
    
    for (const doc of sampleDocuments) {
      const vector = await getEmbedding(doc.text);
      insertData.push({
        id: doc.id,
        text: doc.text,
        vector: vector
      });
    }

    console.log('📥 Inserting vectors into Milvus...');
    await client.insert({
      collection_name: COLLECTION_NAME,
      data: insertData
    });

    // 5. Create an Index for high-speed Approximate Nearest Neighbor (ANN) search
    console.log('⚡ Building HNSW Vector Index...');
    await client.createIndex({
      collection_name: COLLECTION_NAME,
      field_name: 'vector',
      index_type: 'HNSW',
      metric_type: 'COSINE', // Best metric for semantic text matching
      params: { M: 16, efConstruction: 200 }
    });

    // Load the collection into memory to make it searchable
    await client.loadCollectionSync({ collection_name: COLLECTION_NAME });

    // 6. Execute a semantic search with a brand-new query sentence
    const userQuery = "Will artificial intelligence replace employees from IT companies?";
    console.log(`\n🔍 Searching for: "${userQuery}"`);

    const queryVector = await getEmbedding(userQuery);

    const searchResult = await client.search({
      collection_name: COLLECTION_NAME,
      data: [queryVector],
      limit: 2,
      output_fields: ['text'],
      metric_type: 'COSINE'
    });

    // 7. Parse and display the results
    console.log('\n🎯 Top Search Results:');
    searchResult.results.forEach((match, index) => {
      console.log(`[Rank ${index + 1}] Similarity Score: ${(match.score * 100).toFixed(2)}%`);
      console.log(`     Text: "${match.text}"\n`);
    });

  } catch (error) {
    console.error('❌ Error executing vector pipeline:', error);
  } finally {
    // Close the connection gracefully when finished
    await client.closeConnection();
  }
}

main();