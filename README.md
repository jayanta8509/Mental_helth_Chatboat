# Mental Health RAG Chatbot

A sophisticated AI-powered mental health support system using Retrieval Augmented Generation (RAG) with Pinecone vector database. This system provides empathetic, evidence-based mental health guidance while maintaining professional boundaries and safety protocols.

## 🧠 Features

- **Specialized Mental Health Support**: Purpose-built for mental health counseling assistance
- **Dual Data Sources**: 
  - CSV data (interview sessions, synthetic counseling data)
  - PDF data (research papers, psychology textbooks, diagnostic manuals)
- **Crisis Detection**: Automatic safety monitoring with immediate emergency resources
- **User-Specific Memory**: Persistent conversation history per user
- **Multiple Interaction Modes**: Counselor mode (empathetic) and Agent mode (research-focused)
- **Data Source Tracking**: Transparency about which data informed each response
- **FastAPI REST API**: Production-ready endpoints with comprehensive documentation

## 📁 Project Structure

```
Mental_Health_Chatbot/
├── app.py                 # FastAPI application with REST endpoints
├── rag.py                 # Core RAG functionality (function-based)
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
├── Data/                  # Mental health data files
│   ├── Interview_Data_6K.csv
│   ├── Synthetic_Data_10K.csv
│   ├── DSM-5-TR-2022.pdf
│   ├── ICD 11.pdf
│   └── ... (8 PDF files total)
└── vector_store/          # Local FAISS storage (if used)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Mental_Health_Chatbot

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the project root:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone API Key
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 3. Data Setup

Upload your mental health data to Pinecone vector stores:

```bash
# Upload CSV data (interview and synthetic data)
python csv_pinecone.py

# Upload PDF data (research papers and textbooks)
python pdf_pinecone.py
```

### 4. Start the API

```bash
python app.py
```

The API will be available at: `http://localhost:8000`

- **Interactive Documentation**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

## 📡 API Endpoints

### 1. Health Check
```http
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "message": "Mental Health RAG API is running",
  "timestamp": 1640995200.0,
  "pinecone_connected": true,
  "models_loaded": true,
  "status_code": 200
}
```

### 2. Main Chat Endpoint
```http
POST /chat
```

**Request Body:**
```json
{
  "user_id": "user123",
  "query": "I'm feeling anxious about my job interview tomorrow",
  "use_agent": false
}
```

**Response (200 OK):**
```json
{
  "user_id": "user123",
  "query": "I'm feeling anxious about my job interview tomorrow",
  "response": "I understand that you're feeling anxious about your upcoming job interview. This is a very common experience, and it's completely normal to feel this way before an important event...",
  "mode": "counselor",
  "crisis_detected": false,
  "data_source": "csv",
  "timestamp": 1640995200.0,
  "error": null,
  "status_code": 200
}
```

### 3. Agent Mode Chat
```http
POST /chat/agent
```

**Request Body:**
```json
{
  "user_id": "user123",
  "query": "What does research say about cognitive behavioral therapy effectiveness?"
}
```

**Response (200 OK):**
```json
{
  "user_id": "user123",
  "query": "What does research say about cognitive behavioral therapy effectiveness?",
  "response": "According to research findings from clinical studies, Cognitive Behavioral Therapy (CBT) has demonstrated significant effectiveness in treating various mental health conditions...",
  "mode": "agent",
  "crisis_detected": false,
  "data_source": "pdf",
  "timestamp": 1640995200.0,
  "error": null,
  "status_code": 200
}
```

### 4. User Conversation Info
```http
GET /chat/{user_id}
```

**Response (200 OK):**
```json
{
  "user_id": "user123",
  "summary": "Conversation thread: user_user123 - Mental health support session",
  "timestamp": 1640995200.0,
  "status_code": 200
}
```

## 🛡️ Safety Features

### Crisis Detection
Automatically detects crisis indicators and provides immediate resources:

**Crisis Keywords Monitored:**
- "suicide", "kill myself", "end my life"
- "self-harm", "hurt myself"
- "want to die", "better off dead"
- "no point living", "ending it all"

**Crisis Response (200 OK):**
```json
{
  "user_id": "user123",
  "query": "I'm having thoughts of ending my life",
  "response": "🚨 **I'm concerned about your safety right now.** 🚨\n\nYour life has value, and there are people who want to help...\n\n**IMMEDIATE RESOURCES:**\n• **National Suicide Prevention Lifeline: 988** (US)\n• **Crisis Text Line: Text HOME to 741741**\n• **Emergency Services: 911**",
  "mode": "counselor",
  "crisis_detected": true,
  "data_source": "none",
  "timestamp": 1640995200.0,
  "status_code": 200
}
```

## 📊 Data Source Tracking

The `data_source` field indicates which vector store was used:

| Value | Description | Use Case |
|-------|-------------|----------|
| `"csv"` | Interview/synthetic data | Personal experiences, counseling examples |
| `"pdf"` | Research/textbook data | Academic research, clinical information |
| `"both"` | Multiple sources | Complex queries requiring comprehensive data |
| `"none"` | No retrieval | Crisis responses, general conversation, errors |

## 🔧 Error Handling

### Service Unavailable (503)
```json
{
  "detail": "Mental health system is not initialized. Please check server logs.",
  "status_code": 503
}
```

### Internal Server Error (500)
```json
{
  "user_id": "user123",
  "query": "test query",
  "response": "I apologize, but I encountered an error while processing your message. Please try again.",
  "mode": "error",
  "crisis_detected": false,
  "data_source": "none",
  "timestamp": 1640995200.0,
  "error": "Error details...",
  "status_code": 500
}
```

## 🧪 Testing

### Manual Testing with cURL

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test basic chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "query": "I need help with anxiety"
  }'

# Test agent mode
curl -X POST http://localhost:8000/chat/agent \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "query": "What are CBT techniques for depression?"
  }'

# Test crisis detection
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "query": "I am having thoughts of suicide"
  }'
```

### Verify RAG is Working

**Indicators that RAG is functioning:**
1. **`data_source`** field shows "csv", "pdf", or "both" (not "none")
2. **Response quality** is contextual and specific
3. **Different queries** use appropriate data sources
4. **Server logs** show embedding and completion API calls

**Expected response patterns:**
- Anxiety questions → `data_source: "csv"` (counseling data)
- Research questions → `data_source: "pdf"` (academic data)
- Crisis messages → `data_source: "none"` (safety protocol)

## 🏗️ Architecture

### Function-Based RAG System (`rag.py`)

```python
# Core functions
initialize_rag_system()          # Setup complete system
get_response()                   # Main API function
detect_crisis()                  # Safety monitoring
create_retrieval_tools()         # Vector search tools
setup_conversational_chain()    # Memory management
```

### FastAPI Application (`app.py`)

- **Lifespan management**: Automatic system initialization
- **User-specific memory**: Each user gets isolated conversation history
- **Error handling**: Graceful degradation with helpful error messages
- **CORS support**: Cross-origin requests enabled

### Vector Stores

- **CSV Index**: `csv-mental-health` (interview + synthetic data)
- **PDF Index**: `pdf-mental-health` (research + textbooks)
- **Embedding Model**: `text-embedding-3-large`
- **Storage**: Pinecone serverless (AWS us-east-1)

## 📋 Dependencies

```txt
langchain
langchain-openai
langchain-community
langchain-pinecone
pinecone-client
langgraph
beautifulsoup4
python-dotenv
pypdf
pandas
fastapi
uvicorn
pydantic
```

## 🔐 Security & Privacy

- **API Key Protection**: Environment variables for sensitive data
- **Crisis Intervention**: Immediate safety resources for at-risk users
- **Professional Boundaries**: Clear limitations and referral guidance
- **Data Privacy**: User conversations isolated by user_id

## 🛠️ Troubleshooting

### Common Issues

**1. 503 Service Unavailable**
- Check Pinecone API key in `.env`
- Verify vector indexes exist and contain data
- Check server logs for initialization errors

**2. `data_source: "none"` for all queries**
- Verify Pinecone indexes are populated
- Check OpenAI API key for embeddings
- Ensure vector stores are connected

**3. Generic responses**
- Upload data to Pinecone vector stores
- Test with specific, targeted queries
- Check if retrieval tools are functioning

### Monitoring

```bash
# Check system health
curl http://localhost:8000/health

# Monitor server logs
tail -f server.log

# Verify data sources in responses
curl -X POST http://localhost:8000/chat -d '{"user_id":"test","query":"specific mental health question"}'
```

## 📈 Production Deployment

### Environment Variables
```env
PORT=8000
OPENAI_API_KEY=your_production_key
PINECONE_API_KEY=your_production_key
LOG_LEVEL=INFO
```

### Docker Deployment
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📄 License

[Your License Here]

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For support and questions:
- Check the [Issues](issues-url) page
- Review the API documentation at `/docs`
- Verify system health at `/health`

---

**⚠️ Important Note**: This system is designed to support, not replace, professional mental health care. Always encourage users to seek professional help when appropriate.
