import os
import psycopg2
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm
import time

# -------------------
# Config
# -------------------
def configure():
    load_dotenv()
    return {
        "POSTGRES_CONNECTION_STRING": f"dbname={os.getenv('DB_NAME', 'argo')} user={os.getenv('DB_USER', 'postgres')} password={os.getenv('DB_PASS', 'root')}",
        "CHROMA_DIR": os.getenv("CHROMA_DIR", "./chroma_store"),
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",  # Fast, good quality, free
        "BATCH_SIZE": 800,  # Process 800 profiles at once
    }

# -------------------
# Function to summarize a profile
# -------------------
def summarize_profile(cur, profile_id):
    cur.execute("""
        SELECT floater_id, juld, latitude, longitude
        FROM profiles WHERE profile_id = %s;
    """, (profile_id,))
    row = cur.fetchone()
    if not row:
        return None
    floater_id, juld, lat, lon = row

    cur.execute("""
        SELECT MIN(pressure), MAX(pressure),
               MIN(temperature), MAX(temperature),
               MIN(salinity), MAX(salinity),
               COUNT(*)
        FROM measurements WHERE profile_id = %s;
    """, (profile_id,))
    measurement_data = cur.fetchone()
    if not measurement_data or measurement_data[6] == 0:
        return None
        
    p_min, p_max, t_min, t_max, s_min, s_max, count = measurement_data

    # Enhanced summary for better RAG performance - Indian Ocean specific
    # Indian Ocean regional classification
    if lat > 10:
        region = "Northern Indian Ocean"
    elif lat > 0:
        region = "Equatorial Indian Ocean"
    elif lat > -20:
        region = "Southern Tropical Indian Ocean" 
    elif lat > -40:
        region = "Southern Subtropical Indian Ocean"
    else:
        region = "Southern Indian Ocean"
    
    # Longitude-based sub-regions for Indian Ocean
    if lon < 60:
        subregion = "Western Indian Ocean"
    elif lon < 90:
        subregion = "Central Indian Ocean" 
    else:
        subregion = "Eastern Indian Ocean"
    
    return f"""Profile ID: {profile_id}
Floater: {floater_id}
Date: {juld}
Location: {lat:.3f}°N, {lon:.3f}°E
Ocean measurements: {count} data points
Depth range: {p_min:.1f} to {p_max:.1f} dbar
Temperature: {t_min:.2f} to {t_max:.2f}°C
Salinity: {s_min:.2f} to {s_max:.2f} PSU
Region: {region}
Sub-region: {subregion}
Water mass: {"Surface water" if p_max < 200 else "Intermediate water" if p_max < 1000 else "Deep water"}
Temperature range: {"Warm" if t_max > 25 else "Moderate" if t_max > 15 else "Cold"}
Profile type: {"Shallow" if p_max < 500 else "Deep" if p_max < 2000 else "Very deep"}"""

# -------------------
# Main pipeline - Fast local embeddings
# -------------------
def build_embeddings(config, max_profiles=None, resume_from=0):
    """
    Build embeddings using local sentence-transformers model
    - Completely free
    - Fast batch processing
    - Good quality for semantic search
    """
    print("🚀 Loading embedding model...")
    # Load the embedding model (downloads ~80MB on first use)
    model = SentenceTransformer(config["EMBEDDING_MODEL"])
    print(f"✅ Loaded {config['EMBEDDING_MODEL']}")

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=config["CHROMA_DIR"])
    collection = chroma_client.get_or_create_collection("argo_profiles")
    
    print(f"📊 ChromaDB collection has {collection.count()} existing profiles")

    # with psycopg2.connect(config["POSTGRES_CONNECTION_STRING"]) as conn:
    #     with conn.cursor() as cur:
    #         cur.execute("SELECT profile_id FROM profiles ORDER BY profile_id;")
    #         all_profile_ids = [row[0] for row in cur.fetchall()]
            
    #         # Apply limits and resume logic
    #         if max_profiles:
    #             profile_ids = all_profile_ids[resume_from:resume_from + max_profiles]
    #         else:
    #             profile_ids = all_profile_ids[resume_from:]

    #         print(f"🌊 Processing {len(profile_ids)} ARGO profiles...")

    #         batch_size = config["BATCH_SIZE"]
    #         total_processed = 0
    #         start_time = time.time()
            
    #         # Process in batches with progress bar
    #         for i in tqdm(range(0, len(profile_ids), batch_size), desc="Processing batches"):
    #             batch_pids = profile_ids[i:i + batch_size]
    #             batch_summaries = []
    #             batch_ids = []

    #             # Step 1: Generate summaries for the batch
    #             for pid in batch_pids:
    #                 summary = summarize_profile(cur, pid)
    #                 if summary:
    #                     batch_summaries.append(summary)
    #                     batch_ids.append(str(pid))

    #             if not batch_summaries:
    #                 continue

    #             # Step 2: Generate embeddings in one batch call (FAST!)
    #             try:
    #                 embeddings = model.encode(
    #                     batch_summaries, 
    #                     batch_size=32,  # Internal batch size for model
    #                     show_progress_bar=False,
    #                     convert_to_numpy=True
    #                 )
                    
    #                 # Step 3: Store in ChromaDB
    #                 collection.add(
    #                     ids=batch_ids,
    #                     documents=batch_summaries,
    #                     embeddings=embeddings.tolist()
    #                 )
                    
    #                 total_processed += len(batch_ids)
                    
    #                 # Progress update every 10 batches
    #                 if (i // batch_size + 1) % 10 == 0:
    #                     elapsed = time.time() - start_time
    #                     rate = total_processed / elapsed
    #                     eta = (len(profile_ids) - total_processed) / rate if rate > 0 else 0
    #                     print(f"⚡ {total_processed}/{len(profile_ids)} profiles | "
    #                           f"{rate:.1f} profiles/sec | ETA: {eta/60:.1f}min")

    #             except Exception as e:
    #                 print(f"❌ Error processing batch {i//batch_size + 1}: {e}")
    #                 continue

    # elapsed = time.time() - start_time
    # print(f"🎉 Completed! Processed {total_processed} profiles in {elapsed/60:.1f} minutes")
    # print(f"📈 Average rate: {total_processed/elapsed:.1f} profiles/second")

# -------------------
# Query function for RAG chatbot
# -------------------
def query_profiles(config, query_text, n_results=5):
    """
    Query the vector database for relevant profiles
    This is what your RAG chatbot will use
    """
    model = SentenceTransformer(config["EMBEDDING_MODEL"])
    chroma_client = chromadb.PersistentClient(path=config["CHROMA_DIR"])
    collection = chroma_client.get_collection("argo_profiles")
    
    # Generate query embedding
    query_embedding = model.encode([query_text])[0]
    
    # Search similar profiles
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    return {
        "profiles": results["documents"][0],
        "ids": results["ids"][0],   
        "distances": results["distances"][0]
    }

# -------------------
# Utility functions
# -------------------
def test_embeddings(config, num_profiles=1000):
    """Test with a smaller dataset first"""
    print(f"🧪 Testing with {num_profiles} profiles...")
    build_embeddings(config, max_profiles=num_profiles)

def resume_embeddings(config, resume_from_index):
    """Resume from a specific index"""
    print(f"🔄 Resuming from profile index {resume_from_index}...")
    build_embeddings(config, resume_from=resume_from_index)

def test_query(config):
    """Test the query functionality"""
    print("🔍 Testing query functionality...")
    
    test_queries = [
        "Arctic ocean temperature profiles",
        "Deep water salinity measurements",
        "Tropical ocean data from 2023",
        "High salinity profiles in Atlantic",
        "Temperature anomalies in Pacific"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = query_profiles(config, query, n_results=3)
        
        for i, (profile, distance) in enumerate(zip(results["profiles"], results["distances"])):
            print(f"  {i+1}. Distance: {distance:.3f}")
            print(f"     {profile[:100]}...")

# if __name__ == "__main__":
#     config = configure()
    
#     # Choose your approach:
    
#     # # 1. Start with a test run (RECOMMENDED)
#     # print("Starting test run...")
#     # test_embeddings(config, num_profiles=1000)
    
#     # # 2. Test the query functionality
#     # print("\nTesting queries...")
#     # test_query(config)
    
#     3. Uncomment to process all profiles (after testing)
#     print("Processing all profiles...")
#     build_embeddings(config)
    
#     4. Resume from interruption (if needed)
#     resume_embeddings(config, resume_from_index=50000)

if __name__ == "__main__":
    config = configure()
    
    # # Delete existing ChromaDB collection first
    # chroma_client = chromadb.PersistentClient(path=config["CHROMA_DIR"])
    # try:
    #     chroma_client.delete_collection("argo_profiles")
    #     print("🗑️  Deleted existing collection")
    # except:
    #     pass
    
    # Process all from the beginning
    build_embeddings(config, max_profiles=None, resume_from=26918)