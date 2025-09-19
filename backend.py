import os
import chromadb
from llama_cpp import Llama
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
import re
import logging
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5
    max_tokens: Optional[int] = 500

class QueryResponse(BaseModel):
    response: str
    retrieved_data: List[Dict[str, Any]]
    coordinates: List[Dict[str, Any]]
    processing_time: float
    query: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool

class OceanographicRAGSystem:
    def __init__(self, model_path: str, chroma_db_path: str = "./chroma_db"):
        """
        Initialize the RAG system with Mistral-7B and ChromaDB
        
        Args:
            model_path: Path to your Mistral-7B model file
            chroma_db_path: Path to your ChromaDB collection
        """
        self.model_path = model_path
        self.chroma_db_path = chroma_db_path
        self.llm = None
        self.chroma_client = None
        self.collection = None
        
    def initialize_model(self):
        """Initialize Mistral-7B model"""
        try:
            logger.info("Loading Mistral-7B model...")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,  # Context window
                n_threads=4,  # Adjust based on your CPU
                verbose=False,
                temperature=0.1,  # Low temperature for consistent responses
                n_gpu_layers=0,  # Set to > 0 if you have GPU support
            )
            logger.info("Mistral-7B model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    # def initialize_database(self):
    #     """Initialize ChromaDB connection"""
    #     try:
    #         logger.info("Connecting to ChromaDB...")
    #         self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
    #         self.collection = self.chroma_client.get_collection("argo_floats")
    #         logger.info("ChromaDB connected successfully!")
    #         return True
    #     except Exception as e:
    #         logger.error(f"Error connecting to ChromaDB: {str(e)}")
    #         return False

    # In OceanographicRAGSystem class
    def initialize_database(self):
        """Initialize ChromaDB connection by pointing to a running server"""
        try:
            logger.info("Connecting to remote ChromaDB server...")
            self.chroma_client = chromadb.HttpClient(host='localhost', port=8000) # Use the correct host and port
            self.collection = self.chroma_client.get_or_create_collection("argo_floats")
            logger.info("ChromaDB connected successfully!")
            return True
        except Exception as e:
            logger.error(f"Error connecting to ChromaDB: {str(e)}")
            return False
    
    def query_database(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query ChromaDB for relevant ARGO float data"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results for LLM context
            context_data = []
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                context_data.append({
                    'content': doc,
                    'metadata': metadata,
                    'score': results['distances'][0][i] if 'distances' in results else 0
                })
            
            return context_data
        except Exception as e:
            logger.error(f"Error querying database: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    
    def create_prompt(self, user_query: str, context_data: List[Dict]) -> str:
        """Create a structured prompt for Mistral-7B"""
        
        # Format context from retrieved data
        context_str = ""
        for i, data in enumerate(context_data[:3]):  # Use top 3 results
            context_str += f"Data Point {i+1}:\n{data['content']}\n\n"
        
        prompt = f"""<s>[INST] You are an expert oceanographer assistant helping users analyze ARGO float data. Use the provided oceanographic data to answer the user's question accurately and concisely.

Context - ARGO Float Data:
{context_str}

User Question: {user_query}

Instructions:
- Base your answer on the provided ARGO float data
- If asking about locations, mention specific coordinates when available
- For temperature/salinity ranges, provide specific values
- If data is insufficient, clearly state what information is missing
- Keep responses focused on oceanographic insights
- Use scientific terminology appropriately
- Provide quantitative data when possible

Answer: [/INST]"""
        
        return prompt
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response using Mistral-7B"""
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                stop=["</s>", "[INST]"],
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")
    
    def extract_coordinates_from_results(self, context_data: List[Dict]) -> List[Dict]:
        """Extract coordinates for mapping from retrieved data"""
        coordinates = []
        
        for data in context_data:
            try:
                content = data['content']
                # Extract coordinates using regex
                lat_match = re.search(r'Location: (-?\d+\.?\d*)°N', content)
                lon_match = re.search(r'(-?\d+\.?\d*)°E', content)
                
                if lat_match and lon_match:
                    # Extract other info
                    floater_match = re.search(r'Floater: (\w+)', content)
                    date_match = re.search(r'Date: ([\d-]+)', content)
                    temp_match = re.search(r'Temperature: ([\d.]+) to ([\d.]+)°C', content)
                    salinity_match = re.search(r'Salinity: ([\d.]+) to ([\d.]+) PSU', content)
                    depth_match = re.search(r'Depth range: ([\d.]+) to ([\d.]+) dbar', content)
                    region_match = re.search(r'Region: ([^\n]+)', content)
                    
                    coordinates.append({
                        'lat': float(lat_match.group(1)),
                        'lon': float(lon_match.group(1)),
                        'floater_id': floater_match.group(1) if floater_match else 'Unknown',
                        'date': date_match.group(1) if date_match else 'Unknown',
                        'temp_range': f"{temp_match.group(1)}-{temp_match.group(2)}°C" if temp_match else 'N/A',
                        'salinity_range': f"{salinity_match.group(1)}-{salinity_match.group(2)} PSU" if salinity_match else 'N/A',
                        'depth_range': f"{depth_match.group(1)}-{depth_match.group(2)} dbar" if depth_match else 'N/A',
                        'region': region_match.group(1) if region_match else 'Unknown',
                        'content': content
                    })
            except Exception as e:
                logger.warning(f"Error extracting coordinates from data point: {str(e)}")
                continue
        
        return coordinates
    
    def process_query(self, user_query: str, n_results: int = 5, max_tokens: int = 500) -> Dict[str, Any]:
        """Main pipeline: query -> retrieve -> generate"""
        start_time = datetime.now()
        
        try:
            # Step 1: Retrieve relevant data from ChromaDB
            context_data = self.query_database(user_query, n_results)
            
            # Step 2: Create prompt with context
            prompt = self.create_prompt(user_query, context_data)
            
            # Step 3: Generate response
            response = self.generate_response(prompt, max_tokens)
            
            # Step 4: Extract coordinates for visualization
            coordinates = self.extract_coordinates_from_results(context_data)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'response': response,
                'retrieved_data': context_data,
                'coordinates': coordinates,
                'processing_time': processing_time,
                'query': user_query
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# Global RAG system instance
rag_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_system
    
    # Configuration - UPDATE THESE PATHS
    MODEL_PATH = os.getenv("MODEL_PATH", r"C:\Users\dronn\OneDrive\Desktop\sih\mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    logger.info("Starting up Oceanographic RAG System...")
    
    rag_system = OceanographicRAGSystem(MODEL_PATH, CHROMA_DB_PATH)
    
    # Initialize model and database
    model_loaded = rag_system.initialize_model()
    db_connected = rag_system.initialize_database()
    
    if not model_loaded:
        logger.error("Failed to load model!")
    if not db_connected:
        logger.error("Failed to connect to database!")
    
    if model_loaded and db_connected:
        logger.info("System initialization completed successfully!")
    else:
        logger.error("System initialization failed!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Oceanographic AI Assistant API",
    description="AI-powered API for querying ARGO float oceanographic data using natural language",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if rag_system and rag_system.llm and rag_system.collection else "unhealthy",
        model_loaded=rag_system.llm is not None if rag_system else False,
        database_connected=rag_system.collection is not None if rag_system else False
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Main endpoint for processing natural language queries about oceanographic data"""
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not rag_system.llm:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not rag_system.collection:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    logger.info(f"Processing query: {request.query}")
    
    result = rag_system.process_query(
        request.query, 
        request.n_results, 
        request.max_tokens
    )
    
    return QueryResponse(**result)

@app.get("/examples")
async def get_example_queries():
    """Get example queries for the frontend"""
    examples = [
        {
            "query": "Show me salinity profiles near the equator in March 2023",
            "description": "Find salinity measurements in equatorial regions"
        },
        {
            "query": "What are the temperature ranges in the Arabian Sea?",
            "description": "Get temperature data for Arabian Sea region"
        },
        {
            "query": "Find ARGO floats with deep water measurements",
            "description": "Locate floats that measure deep ocean data"
        },
        {
            "query": "Compare temperature data in tropical regions",
            "description": "Compare temperature across tropical ocean areas"
        },
        {
            "query": "Show me recent float data from the Indian Ocean",
            "description": "Get latest measurements from Indian Ocean floats"
        },
        {
            "query": "What floats are operating near Madagascar?",
            "description": "Find active floats in Madagascar vicinity"
        }
    ]
    
    return {"examples": examples}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Oceanographic AI Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/query - Post natural language queries",
            "health": "/health - Check system status",
            "examples": "/examples - Get example queries",
            "docs": "/docs - API documentation"
        }
    }

# Additional utility endpoints
@app.get("/stats")
async def get_database_stats():
    """Get basic statistics about the database"""
    if not rag_system or not rag_system.collection:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        # Get collection count
        collection_info = rag_system.collection.count()
        return {
            "total_records": collection_info,
            "collection_name": "argo_floats"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # For development
    uvicorn.run(
        "main:app",  # Replace "main" with your filename
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )

# Production deployment example:
"""
# Install uvicorn: pip install uvicorn
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000

# Or with gunicorn for production:
# pip install gunicorn
# gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
"""