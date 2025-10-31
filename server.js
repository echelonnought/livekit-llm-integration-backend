require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { RoomServiceClient, AccessToken } = require('livekit-server-sdk');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { VertexAI } = require('@google-cloud/vertexai');
const { createClient } = require('@deepgram/sdk');
// Note: ElevenLabsClient not needed - using direct API calls with axios
const socketio = require('socket.io');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');
const { HttpsProxyAgent } = require('https-proxy-agent');

// Centralized configuration constants
const AUDIO_DIR = path.join(__dirname, 'audio');
const ELEVEN_TTS_MODEL = process.env.ELEVEN_TTS_MODEL || 'eleven_monolingual_v1';

// OpenAI config (fallback LLM)
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-3.5-turbo';

function getOpenAIAdapter() {
  if (!OPENAI_API_KEY) throw new Error('OPENAI_API_KEY not set');

  return {
    // Keep the same calling shape as Google/Vertex adapters used elsewhere in this file
    generateContent: (prompt) => {
      return {
        // `response` is a promise resolving to an object with text() method
        response: (async () => {
          try {
            const payload = {
              model: OPENAI_MODEL,
              messages: [{ role: 'user', content: prompt }],
              max_tokens: 1200
            };

            const headers = {
              Authorization: `Bearer ${OPENAI_API_KEY}`,
              'Content-Type': 'application/json'
            };

            const resp = await axios.post('https://api.openai.com/v1/chat/completions', payload, {
              headers,
              timeout: 20000,
              validateStatus: () => true
            });

            if (!(resp.status >= 200 && resp.status < 300)) {
              throw new Error(`OpenAI API error ${resp.status}: ${JSON.stringify(resp.data)}`);
            }

            const text = (resp.data?.choices || [])
              .map(c => (c?.message?.content || ''))
              .join('\n')
              .trim();

            return {
              text: () => text
            };
          } catch (err) {
            // Bubble up error to caller
            throw err;
          }
        })()
      };
    }
  };
}

// Initialize services
const app = express();
const upload = multer({ storage: multer.memoryStorage() });

// Initialize AI services (supports both Google Cloud and direct API)
let genAI;
let vertexAI;

// Initialize Google AI Studio if API key is provided
if (process.env.GEMINI_API_KEY) {
  console.log('🔄 Initializing Google AI Studio with API key');
  genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
}

// Initialize Vertex AI if credentials are provided
if (process.env.GOOGLE_APPLICATION_CREDENTIALS && process.env.GOOGLE_CLOUD_PROJECT_ID) {
  try {
    console.log('🔄 Initializing Google Cloud Vertex AI with service account credentials');
    vertexAI = new VertexAI({
      project: process.env.GOOGLE_CLOUD_PROJECT_ID,
      location: 'us-central1' // You can change this to your preferred region
    });
  } catch (error) {
    console.warn('⚠️ Vertex AI initialization failed:', error.message);
    vertexAI = null;
  }
}

// Check if at least one AI service is available
if (!genAI && !vertexAI) {
  console.warn('⚠️ No Google AI credentials found. AI features will not work.');
  console.warn('Please set either:');
  console.warn('1. GEMINI_API_KEY for Google AI Studio (recommended for development)');
  console.warn('2. GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT_ID for Vertex AI');
  console.warn('Server will start but AI endpoints will return errors.');
} else {
  if (genAI) console.log('✅ Google AI Studio initialized');
  if (vertexAI) console.log('✅ Vertex AI initialized');
}

const deepgram = createClient(process.env.DEEPGRAM_API_KEY);
// Note: ElevenLabs client not needed - using direct API calls with axios
// Note: Google TTS fallback uses Google Translate API (no authentication needed)
const livekit = new RoomServiceClient(
  process.env.LIVEKIT_URL,
  process.env.LIVEKIT_API_KEY,
  process.env.LIVEKIT_API_SECRET
);

// Configuration
const AZURE_SPEECH_KEY = process.env.AZURE_SPEECH_KEY;
const AZURE_SPEECH_REGION = process.env.AZURE_SPEECH_REGION || 'eastus';
const activeRooms = new Map();
const roomParticipants = new Map();
// track active recorders: key = trackId or generated id -> { participantName, socketId, trackId, lastActiveAt }
const activeRecorders = new Map();
// track which participants in a room have opted in to receive STT forwards
const roomSttOptIns = new Map(); // roomName -> Set(participantSocketId)
// map requestId -> requesterSocketId for request-transcript flow
const transcriptRequestMap = new Map();

// Middleware
app.use(cors({
  origin: [
    'http://localhost:5173',
    'http://localhost:5174',
    'https://your-production-domain.com'
  ],
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true
}));
app.options('*', cors());  
app.use(express.json());
app.use('/audio', express.static(AUDIO_DIR));

// Ensure directories exist
if (!fs.existsSync(AUDIO_DIR)) fs.mkdirSync(AUDIO_DIR, { recursive: true });

// Network connectivity test function
async function testNetworkConnectivity() {
  try {
    console.log('🌐 Testing network connectivity...');
    
    // Test basic internet connectivity
    const testResponse = await axios.get('https://httpbin.org/ip', {
      timeout: 5000,
      validateStatus: () => true
    });
    
    if (testResponse.status === 200) {
      console.log('✅ Basic internet connectivity: OK');
      return true;
    } else {
      console.log('❌ Basic internet connectivity: Failed');
      return false;
    }
  } catch (error) {
    console.log('❌ Basic internet connectivity: Failed -', error.message);
    return false;
  }
}

// Helper function to get an available Google AI Studio model
async function getAvailableGoogleAIModel() {
  if (!genAI) {
    throw new Error('GoogleGenerativeAI not initialized');
  }
  
  // List of models to try in order of preference
  const modelNames = [
    'gemini-1.5-flash',
    'gemini-1.5-flash-001',
    'gemini-1.5-pro',
    'gemini-1.5-pro-001',
    'gemini-pro',
    'gemini-1.0-pro',
    'gemini-1.0-pro-001'
  ];
  
  // Try each model until one works
  for (const modelName of modelNames) {
    try {
      console.log(`🧪 Testing model: ${modelName}`);
      const model = genAI.getGenerativeModel({ model: modelName });
      
      // Test the model with a simple request
      const result = await model.generateContent('Respond with "OK" only');
      const response = await result.response;
      
      if (response.text()) {
        console.log(`✅ Successfully using model: ${modelName}`);
        return model;
      }
    } catch (error) {
      console.log(`⚠️ Model ${modelName} not available: ${error.message.substring(0, 100)}...`);
      // Continue to next model
    }
  }
  
  throw new Error('No available Google AI Studio models found');
}

// Helper Functions
async function getAIModel() {
  // Try Google AI Studio first if API key is available (simpler for development)
  if (genAI && process.env.GEMINI_API_KEY) {
    try {
      console.log('🔄 Using Google AI Studio with API key (development mode)');
      const model = await getAvailableGoogleAIModel();
      return model;
    } catch (error) {
      console.warn('⚠️ Google AI Studio model selection failed:', error.message);
    }
  }
  
  // Fallback to Vertex AI if Google AI Studio isn't configured or fails
  if (vertexAI && process.env.GOOGLE_APPLICATION_CREDENTIALS && process.env.GOOGLE_CLOUD_PROJECT_ID) {
    try {
      console.log('🔄 Attempting to use Google Cloud Vertex AI...');
      // Try different Vertex AI models
      const vertexModels = [
        'gemini-1.0-pro',
        'gemini-pro'
      ];
      
      for (const modelName of vertexModels) {
        try {
          const model = vertexAI.getGenerativeModel({ model: modelName });
          // Test the model
          console.log(`✅ Vertex AI model created successfully: ${modelName}`);
          return model;
        } catch (modelError) {
          console.warn(`⚠️ Vertex AI model ${modelName} not available:`, modelError.message);
        }
      }
      
      // If no specific model works, try with default
      const model = vertexAI.getGenerativeModel({ model: 'gemini-1.0-pro' });
      console.log('✅ Vertex AI model created successfully with default model');
      return model;
    } catch (error) {
      console.warn('⚠️ Vertex AI model creation failed:', error.message);
    }
  }
  
  // If neither is available, provide helpful error message
  // Try OpenAI as a final fallback if configured
  if (OPENAI_API_KEY) {
    try {
      console.log('🔄 Falling back to OpenAI');
      const openaiModel = getOpenAIAdapter();
      return openaiModel;
    } catch (err) {
      console.warn('⚠️ OpenAI adapter failed to initialize:', err.message);
    }
  }

  const errorMessage = [];
  if (!process.env.GEMINI_API_KEY) {
    errorMessage.push('- GEMINI_API_KEY not set (get from: https://makersuite.google.com/app/apikey)');
  }
  if (!process.env.GOOGLE_APPLICATION_CREDENTIALS || !process.env.GOOGLE_CLOUD_PROJECT_ID) {
    errorMessage.push('- Vertex AI credentials not properly configured');
  }
  if (!OPENAI_API_KEY) {
    errorMessage.push('- OPENAI_API_KEY not set (get from: https://platform.openai.com/account/api-keys)');
  }

  throw new Error(`No AI model available. Please configure:\n${errorMessage.join('\n')}`);
}

async function getAvailableVoices() {
  const httpsProxy = process.env.HTTPS_PROXY || process.env.https_proxy;
  const agent = httpsProxy ? new HttpsProxyAgent(httpsProxy) : undefined;

  const maxRetries = 3;
  let lastError;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`Attempting to fetch voices (attempt ${attempt}/${maxRetries})...`);
      
      const response = await axios.get('https://api.elevenlabs.io/v1/voices', {
        headers: { 'xi-api-key': process.env.ELEVENLABS_API_KEY },
        httpsAgent: agent,
        timeout: 20000,
        validateStatus: () => true,
      });

      if (response.status !== 200) {
        throw new Error(`API returned status ${response.status}: ${response.data?.detail?.message || 'Failed to fetch voices'}`);
      }

      return (response.data?.voices || []).map(v => ({
        voice_id: v.voice_id,
        name: v.name,
        category: v.category,
        labels: v.labels,
      }));
    } catch (error) {
      lastError = error;
      console.warn(`Attempt ${attempt} failed:`, error.message);
      
      if (attempt < maxRetries) {
        const delayMs = Math.pow(2, attempt) * 1000; // Exponential backoff
        console.log(`Retrying in ${delayMs}ms...`);
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }
  }

  throw new Error(`Failed to fetch voices after ${maxRetries} attempts. Last error: ${lastError.message}`);
}

async function synthesizeAndStore(text, voiceId) {
  const audioPath = path.join('audio', `msg-${Date.now()}.mp3`);
  
  // Use axios for consistency instead of ElevenLabs SDK
  const httpsProxy = process.env.HTTPS_PROXY || process.env.https_proxy;
  const agent = httpsProxy ? new HttpsProxyAgent(httpsProxy) : undefined;

  const audioResponse = await axios.post(
    `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
    {
      text: text,
      model_id: "eleven_multilingual_v2",
      voice_settings: {
        stability: 0.5,
        similarity_boost: 0.5
      }
    },
    {
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': process.env.ELEVENLABS_API_KEY
      },
      responseType: 'arraybuffer',
      timeout: 30000,
      httpsAgent: agent,
      validateStatus: () => true
    }
  );

  if (audioResponse.status !== 200) {
    throw new Error(`Failed to synthesize: ${audioResponse.status}`);
  }

  // Save audio file
  fs.writeFileSync(audioPath, Buffer.from(audioResponse.data));
  
  return `/audio/${path.basename(audioPath)}`;
}

function generateToken(apiKey, apiSecret, roomName, participantName) {
  const token = new AccessToken(apiKey, apiSecret, {
    identity: participantName,
    ttl: '1h',
  });

  token.addGrant({
    roomJoin: true,
    room: roomName,
    roomAdmin: false,
    roomCreate: false,
    canPublish: true,
    canSubscribe: true,
    canPublishData: true,
  });

  return token.toJwt();
}

// --- API Endpoints --- //

// Voice Cloning
app.post('/clone-voice', upload.single('audio'), async (req, res) => {
  try {
    const { voiceName, userId } = req.body;
    
    if (!req.file?.buffer || req.file.size < 500_000) {
      return res.status(400).json({
        error: 'Invalid audio',
        details: 'Please provide at least 30-60 seconds of clear speech'
      });
    }

    const form = new FormData();
    form.append('name', voiceName || `voice-${Date.now()}`);
    if (userId) form.append('description', `Voice for user ${userId}`);
    form.append('files', req.file.buffer, {
      filename: req.file.originalname || 'sample.wav',
      contentType: req.file.mimetype || 'audio/wav'
    });

    const httpsProxy = process.env.HTTPS_PROXY || process.env.https_proxy;
    const agent = httpsProxy ? new HttpsProxyAgent(httpsProxy) : undefined;

    const response = await axios.post('https://api.elevenlabs.io/v1/voices/add', form, {
      headers: {
        'xi-api-key': process.env.ELEVENLABS_API_KEY,
        ...form.getHeaders(),
      },
      maxBodyLength: Infinity,
      timeout: 60000,
      httpsAgent: agent,
      validateStatus: () => true,
    });

    if (response.status !== 200) {
      if (response.data?.detail?.status === 'can_not_use_instant_voice_cloning') {
        return res.status(402).json({
          error: 'Instant cloning not allowed',
          details: 'Your plan does not include instant voice cloning'
        });
      }
      throw new Error(response.data?.detail?.message || 'Voice cloning failed');
    }

    res.json({
      success: true,
      voiceId: response.data.voice_id,
      voiceName: response.data.name
    });
    
  } catch (err) {
    console.error('Voice cloning error:', err);
    res.status(500).json({ 
      error: 'Voice cloning failed', 
      details: err.message 
    });
  }
});

// voice preview


// Cleanup old preview files (optional)
setInterval(() => {
  const audioDir = path.join(__dirname, 'public', 'audio');
  fs.readdir(audioDir, (err, files) => {
    if (err) return;
    
    files.forEach(file => {
      if (file.startsWith('preview_')) {
        const filePath = path.join(audioDir, file);
        fs.stat(filePath, (err, stats) => {
          if (err) return;
          
          // Delete files older than 1 hour
          const ageMs = Date.now() - stats.mtime.getTime();
          if (ageMs > 60 * 60 * 1000) {
            fs.unlink(filePath, () => {});
          }
        });
      }
    });
  });
}, 30 * 60 * 1000);

// Text-to-Speech
app.get('/synthesize', async (req, res) => {
  try {
    let { text, voiceId } = req.query;
    
    if (!voiceId) {
      const voices = await getAvailableVoices();
      const stockVoice = voices.find(v => v.category === 'premade') || voices[0];
      voiceId = stockVoice?.voice_id;
      if (!voiceId) throw new Error('No voices available');
    }

    console.log(`🎵 Synthesizing audio for voice: ${voiceId}`);
    
    // Use axios for consistency instead of ElevenLabs SDK
    const httpsProxy = process.env.HTTPS_PROXY || process.env.https_proxy;
    const agent = httpsProxy ? new HttpsProxyAgent(httpsProxy) : undefined;

    const audioResponse = await axios.post(
      `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
      {
        text: text,
        model_id: "eleven_multilingual_v2",
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.5
        }
      },
      {
        headers: {
          'Accept': 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': process.env.ELEVENLABS_API_KEY
        },
        responseType: 'arraybuffer',
        timeout: 30000,
        httpsAgent: agent,
        validateStatus: () => true
      }
    );

    if (audioResponse.status !== 200) {
      throw new Error(`Failed to synthesize: ${audioResponse.status}`);
    }

    // Save audio file
    const audioPath = path.join('audio', `${voiceId}-${Date.now()}.mp3`);
    fs.writeFileSync(audioPath, Buffer.from(audioResponse.data));
    
    console.log(`✅ Audio synthesized successfully: ${path.basename(audioPath)}`);
    
    res.json({ url: `/audio/${path.basename(audioPath)}` });
    
  } catch (err) {
    console.error('❌ Synthesis error:', err);
    res.status(500).json({ 
      error: 'Speech synthesis failed',
      details: err.message
    });
  }
});

// AI Processing with Gemini
app.post('/ask-ai', async (req, res) => {
  try {
    const { messages, voiceId } = req.body;
    const voices = await getAvailableVoices();
    
    if (!voices.some(v => v.voice_id === voiceId)) {
      return res.status(400).json({ error: 'Invalid voiceId' });
    }

    // Format messages for Gemini
    const prompt = messages.map(m => {
      if (typeof m === 'string') return m;
      return m.content || '';
    }).join('\n');

    // Get Gemini model (works with both Google Cloud and AI Studio)
    const model = await getAIModel();
    
    // Generate content with Gemini
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const responseText = response.text() || "No response from AI";

    // Generate audio using the synthesize endpoint
    const synthesizeUrl = `${req.protocol}://${req.get('host')}/synthesize?text=${encodeURIComponent(responseText)}&voiceId=${voiceId}`;
    const audioResponse = await axios.get(synthesizeUrl, {
      timeout: 30000,
      validateStatus: () => true
    });
    
    if (audioResponse.status !== 200) {
      throw new Error(`Failed to synthesize audio: ${audioResponse.status}`);
    }
    
    const audioData = audioResponse.data;

    res.json({
      text: responseText,
      audioUrl: audioData.url,
      voiceId: voiceId
    });
  } catch (err) {
    console.error('AI processing error:', err);
    res.status(500).json({ error: 'AI processing failed', details: err.message });
  }
});

// Speech-to-Text with Deepgram
app.post('/transcribe', upload.single('audio'), async (req, res) => {
  try {
    if (!req.file?.buffer) {
      return res.status(400).json({ error: 'No audio file provided' });
    }

    // Determine the mimetype for Deepgram
    const mimeType = req.file.mimetype || 'audio/wav';
    
    // Transcribe with Deepgram
    const { result, error } = await deepgram.listen.prerecorded.transcribeFile(
      req.file.buffer,
      {
        model: 'nova-2',
        language: 'en-US',
        smart_format: true,
        punctuate: true,
        diarize: false,
        mimetype: mimeType
      }
    );

    if (error) {
      throw error;
    }

    const transcript = result?.results?.channels?.[0]?.alternatives?.[0]?.transcript || '';
    const confidence = result?.results?.channels?.[0]?.alternatives?.[0]?.confidence || 0;

    res.json({
      success: true,
      transcript: transcript,
      confidence: confidence,
      words: result?.results?.channels?.[0]?.alternatives?.[0]?.words || []
    });
    
  } catch (err) {
    console.error('Transcription error:', err);
    res.status(500).json({ 
      error: 'Speech transcription failed', 
      details: err.message 
    });
  }
});

// Health endpoint - reports availability of configured AI / STT services
app.get('/health', async (req, res) => {
  try {
    const services = {
      googleAIStudio: !!(genAI && process.env.GEMINI_API_KEY),
      vertexAI: !!(vertexAI && process.env.GOOGLE_APPLICATION_CREDENTIALS && process.env.GOOGLE_CLOUD_PROJECT_ID),
      openai: !!OPENAI_API_KEY,
      deepgram: !!process.env.DEEPGRAM_API_KEY,
      elevenlabs: !!process.env.ELEVENLABS_API_KEY
    };

    res.json({ ok: true, services });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// AI Chat Response with Voice Output
app.post('/ai-chat-response', async (req, res) => {
  console.log('🚀 AI CHAT RESPONSE ENDPOINT CALLED!');
  console.log('Request body:', JSON.stringify(req.body, null, 2));
  
  try {
    const { message, voiceId, roomName, participantName } = req.body;
    
    if (!message || !voiceId) {
      console.log('❌ Missing required parameters:', { message: !!message, voiceId: !!voiceId });
      return res.status(400).json({ 
        error: 'message and voiceId are required' 
      });
    }

    console.log(`🤖 Generating AI response for message: "${message}"`);
    console.log(`🎤 Using voice ID: ${voiceId}`);
    
    // Step 1: Generate AI response using Vertex AI/Gemini with fallback
    console.log('🧠 Step 1: Calling AI model...');
    let responseText;
    
    // Try Vertex AI first if available
    if (vertexAI && process.env.GOOGLE_APPLICATION_CREDENTIALS && process.env.GOOGLE_CLOUD_PROJECT_ID) {
      try {
        console.log('🔄 Attempting to use Google Cloud Vertex AI...');
        const model = vertexAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
        const prompt = `You are a helpful AI assistant in a live chat. Respond naturally and conversationally to this message: "${message}". Keep your response concise and engaging, suitable for voice output.`;
        
        const aiResult = await model.generateContent(prompt);
        const aiResponse = await aiResult.response;
        responseText = aiResponse.text() || "I'm sorry, I couldn't generate a response right now.";
        
        console.log('✅ Vertex AI response generated successfully');
      } catch (error) {
        console.warn('⚠️ Vertex AI failed during generation:', error.message);
        console.log('🔄 Falling back to Google AI Studio...');
        responseText = null; // Reset to trigger fallback
      }
    }
    
    // Fallback to Google AI Studio if Vertex AI failed or isn't available
    if (!responseText && genAI && process.env.GEMINI_API_KEY) {
      try {
        console.log('🔄 Using Google AI Studio with API key');
        const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
        const prompt = `You are a helpful AI assistant in a live chat. Respond naturally and conversationally to this message: "${message}". Keep your response concise and engaging, suitable for voice output.`;
        
        const aiResult = await model.generateContent(prompt);
        const aiResponse = await aiResult.response;
        responseText = aiResponse.text() || "I'm sorry, I couldn't generate a response right now.";
        
        console.log('✅ Google AI Studio response generated successfully');
      } catch (error) {
        console.error('❌ Google AI Studio also failed:', error.message);
        // Leave responseText as null so downstream fallbacks (OpenAI) can be attempted
        responseText = null;
      }
    }

    // Fallback: if Gemini/Vertex failed to produce a response, try OpenAI (if configured)
    // if (!responseText && OPENAI_API_KEY) {
    //   try {
    //     console.log('🔄 Falling back to OpenAI API');
    //     const prompt = `You are a helpful AI assistant in a live chat. Respond naturally and conversationally to this message: "${message}". Keep your response concise and engaging, suitable for voice output.`;
    //     const openaiAdapter = getOpenAIAdapter();
    //     const aiResult = openaiAdapter.generateContent(prompt);
    //     const aiResponse = await aiResult.response;
    //     responseText = aiResponse.text() || "I'm sorry, I couldn't generate a response right now.";
    //     console.log('✅ OpenAI response generated successfully');
    //   } catch (err) {
    //     console.warn('⚠️ OpenAI fallback failed:', err.message);
    //     // keep responseText null so later logic can handle unavailability
    //   }
    // }
    
    // If no AI service is available or both failed
    if (!responseText) {
      responseText = "I'm sorry, AI services are currently unavailable. Please check your configuration.";
    }
    
    console.log(`✅ AI generated response: "${responseText}"`);
    
    // Step 2: Convert AI response to speech using ElevenLabs
    console.log('🎵 Step 2: Converting to speech...');
    console.log('Voice ID:', voiceId);
    console.log('Response text length:', responseText.length, 'characters');
    console.log('API Key present:', !!process.env.ELEVENLABS_API_KEY);
    console.log('API Key format check:', process.env.ELEVENLABS_API_KEY ? process.env.ELEVENLABS_API_KEY.substring(0, 10) + '...' : 'NOT SET');
    
    const httpsProxy = process.env.HTTPS_PROXY || process.env.https_proxy;
    const agent = httpsProxy ? new HttpsProxyAgent(httpsProxy) : undefined;

    const maxRetries = 3;
    let audioResponse;
    let lastError;
    
    // Add delay between attempts to avoid triggering rate limits
    const baseDelay = 5000; // 5 seconds base delay

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`🔄 Audio generation attempt ${attempt}/${maxRetries}...`);
        
        // Add delay before each attempt (except first)
        if (attempt > 1) {
          console.log(`⏳ Waiting ${baseDelay}ms to avoid rate limits...`);
          await new Promise(resolve => setTimeout(resolve, baseDelay));
        }
        console.log('Request URL:', `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`);
        console.log('Request headers:', {
          'Accept': 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': process.env.ELEVENLABS_API_KEY ? process.env.ELEVENLABS_API_KEY.substring(0, 10) + '...' : 'NOT SET'
        });
        
        audioResponse = await axios.post(
          `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
          {
            text: responseText,
            model_id: 'eleven_monolingual_v1',
            voice_settings: {
              stability: 0.5,
              similarity_boost: 0.5
            }
          },
          {
            headers: {
              'Accept': 'audio/mpeg',
              'Content-Type': 'application/json',
              'xi-api-key': process.env.ELEVENLABS_API_KEY
            },
            responseType: 'arraybuffer',
            timeout: 30000,
            httpsAgent: agent,
            validateStatus: () => true
          }
        );

        console.log(`🎤 Audio API response status: ${audioResponse.status}`);
        console.log('Response headers:', audioResponse.headers);
        
        if (audioResponse.status === 401) {
          // Handle permission errors separately - don't retry
          console.log('Raw 401 response:', audioResponse.data);
          let errorData;
          try {
            // Try to parse the error response
            if (audioResponse.data instanceof ArrayBuffer) {
              const decoder = new TextDecoder();
              const errorText = decoder.decode(audioResponse.data);
              console.log('Decoded error text:', errorText);
              errorData = JSON.parse(errorText);
            } else if (Buffer.isBuffer(audioResponse.data)) {
              const errorText = audioResponse.data.toString('utf8');
              console.log('Decoded error text:', errorText);
              errorData = JSON.parse(errorText);
            } else {
              errorData = audioResponse.data;
            }
          } catch (parseError) {
            console.log('Could not parse error response:', parseError.message);
            errorData = { detail: { message: 'Authentication failed - unable to parse error details' } };
          }
          
          console.log('Parsed error data:', errorData);
          
          if (errorData?.detail?.status === 'missing_permissions') {
            throw new Error(`ElevenLabs API key missing permissions: ${errorData.detail.message}. Please check your API key has text-to-speech permissions.`);
          } else if (errorData?.detail?.status === 'detected_unusual_activity') {
            // Don't retry for permanent free tier restrictions
            if (errorData.detail.message.includes('Free Tier usage disabled')) {
              throw new Error(`ElevenLabs free tier permanently disabled: ${errorData.detail.message}`);
            } else {
              throw new Error(`ElevenLabs account temporarily restricted: ${errorData.detail.message}. Please wait 15-30 minutes or contact support.`);
            }
          } else {
            throw new Error(`ElevenLabs authentication failed: ${errorData?.detail?.message || errorData?.message || 'Invalid API key'}`);
          }
        } else if (audioResponse.status === 200) {
          console.log('✅ Audio generation successful!');
          console.log('Audio data size:', audioResponse.data.byteLength, 'bytes');
          break; // Success, exit retry loop
        } else {
          console.log('Unexpected status code:', audioResponse.status);
          console.log('Response data:', audioResponse.data);
          throw new Error(`API returned status ${audioResponse.status}: ${audioResponse.data || 'Unknown error'}`);
        }
      } catch (error) {
        lastError = error;
        console.warn(`⚠️ Audio generation attempt ${attempt} failed:`, error.message);
        
        if (attempt < maxRetries) {
          const delayMs = Math.pow(2, attempt) * 1000;
          console.log(`🔄 Retrying in ${delayMs}ms...`);
          await new Promise(resolve => setTimeout(resolve, delayMs));
        }
      }
    }

    if (!audioResponse || audioResponse.status !== 200) {
      const errorMessage = `Failed to generate audio after ${maxRetries} attempts. Last error: ${lastError?.message || 'Unknown error'}`;
      console.error('❌ ElevenLabs audio generation failed:', errorMessage);
      
      // Try a free TTS API as fallback (no authentication required)
      if (process.env.GEMINI_API_KEY) {
        console.log('🔄 Trying free TTS API as fallback...');
        try {
          // Use a simple free TTS service (ResponsiveVoice or similar)
          const ttsText = encodeURIComponent(responseText);
          const ttsUrl = `https://translate.google.com/translate_tts?ie=UTF-8&tl=en&client=tw-ob&q=${ttsText}`;
          
          console.log('🔄 Downloading TTS audio from Google Translate...');
          const ttsResponse = await axios.get(ttsUrl, {
            responseType: 'arraybuffer',
            timeout: 15000,
            headers: {
              'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            },
            validateStatus: () => true
          });
          
          if (ttsResponse.status === 200 && ttsResponse.data.byteLength > 1000) {
            // Save Google Translate TTS audio file
            const fileName = `google_translate_tts_${Date.now()}.mp3`;
            const filePath = path.join(__dirname, 'audio', fileName);
            
            if (!fs.existsSync(path.dirname(filePath))) {
              fs.mkdirSync(path.dirname(filePath), { recursive: true });
            }
            
            fs.writeFileSync(filePath, Buffer.from(ttsResponse.data));
            
            console.log('✅ Free TTS audio generated successfully:', fileName);
            console.log('🔗 Audio URL: /audio/' + fileName);
            
            const response = {
              aiResponse: responseText,
              audioUrl: `/audio/${fileName}`,
              voiceId: 'google-translate-tts',
              originalMessage: message,
              ttsProvider: 'Google Translate TTS (Free)'
            };
            
            // Still broadcast to room if requested
            if (roomName && io) {
              console.log(`📡 Broadcasting free TTS response to room: ${roomName}`);
              io.to(roomName).emit('ai-chat-response', {
                ...response,
                participantName: participantName || 'AI Assistant',
                timestamp: new Date().toISOString()
              });
            }
            
            console.log('🎉 AI CHAT RESPONSE COMPLETED WITH FREE TTS!');
            return res.json(response);
          } else {
            console.log('⚠️ Free TTS response too small or failed:', ttsResponse.status);
          }
          
        } catch (freeTtsError) {
          console.error('❌ Free TTS fallback also failed:', freeTtsError.message);
        }
      }
      
      // Return AI response without audio if all TTS options fail
      console.log('📋 Returning text-only response due to audio generation failure');
      const response = {
        aiResponse: responseText,
        audioUrl: null,
        voiceId: voiceId,
        originalMessage: message,
        audioError: 'Audio generation failed - text response only'
      };
      
      // Still broadcast to room if requested
      if (roomName && io) {
        console.log(`📡 Broadcasting text-only response to room: ${roomName}`);
        io.to(roomName).emit('ai-chat-response', {
          ...response,
          participantName: participantName || 'AI Assistant',
          timestamp: new Date().toISOString()
        });
      }
      
      return res.json(response);
    }

    // Step 3: Save audio file
    console.log('💾 Step 3: Saving audio file...');
    const audioBuffer = audioResponse.data;
    const fileName = `ai_response_${Date.now()}.mp3`;
    const filePath = path.join(__dirname, 'audio', fileName);
    
    // Ensure directory exists
    if (!fs.existsSync(path.dirname(filePath))) {
      fs.mkdirSync(path.dirname(filePath), { recursive: true });
    }
    
    // Write file
    fs.writeFileSync(filePath, Buffer.from(audioBuffer));
    
    console.log(`✅ Audio file saved: ${fileName}`);
    console.log(`🔗 Audio URL: /audio/${fileName}`);
    
    const response = {
      aiResponse: responseText,
      audioUrl: `/audio/${fileName}`,
      voiceId: voiceId,
      originalMessage: message
    };
    
    // Step 4: Optionally broadcast to room via WebSocket if roomName provided
    if (roomName && io) {
      console.log(`📡 Broadcasting to room: ${roomName}`);
      io.to(roomName).emit('ai-chat-response', {
        ...response,
        participantName: participantName || 'AI Assistant',
        timestamp: new Date().toISOString()
      });
      console.log(`✅ Broadcasted AI response to room: ${roomName}`);
    }
    
    console.log('🎉 AI CHAT RESPONSE COMPLETED SUCCESSFULLY!');
    console.log('Response:', JSON.stringify(response, null, 2));
    
    res.json(response);
    
  } catch (error) {
    console.error('❌ AI CHAT RESPONSE ERROR:', error.message);
    console.error('Error stack:', error.stack);
    res.status(500).json({ 
      error: 'Failed to generate AI chat response',
      details: error.message
    });
  }
});

// Combined Speech-to-Text + AI + Text-to-Speech
app.post('/voice-to-voice', upload.single('audio'), async (req, res) => {
  try {
    const { voiceId } = req.body;
    
    if (!req.file?.buffer) {
      return res.status(400).json({ error: 'No audio file provided' });
    }

    // Validate voice ID
    const voices = await getAvailableVoices();
    if (!voices.some(v => v.voice_id === voiceId)) {
      return res.status(400).json({ error: 'Invalid voiceId' });
    }

    // Step 1: Transcribe audio
    const { result, error } = await deepgram.listen.prerecorded.transcribeFile(
      req.file.buffer,
      {
        model: 'nova-2',
        language: 'en-US',
        smart_format: true,
        punctuate: true,
        mimetype: req.file.mimetype || 'audio/wav'
      }
    );

    if (error) {
      throw error;
    }

    const transcript = result?.results?.channels?.[0]?.alternatives?.[0]?.transcript || '';
    
    if (!transcript.trim()) {
      return res.status(400).json({ error: 'No speech detected in audio' });
    }

    // Step 2: Process with Gemini AI
    const model = await getAIModel();
    const aiResult = await model.generateContent(transcript);
    const aiResponse = await aiResult.response;
    const responseText = aiResponse.text() || "No response from AI";

    // Step 3: Generate audio response
    const synthesizeUrl = `${req.protocol}://${req.get('host')}/synthesize?text=${encodeURIComponent(responseText)}&voiceId=${voiceId}`;
    const audioResponse = await axios.get(synthesizeUrl, {
      timeout: 30000,
      validateStatus: () => true
    });
    
    if (audioResponse.status !== 200) {
      throw new Error(`Failed to synthesize audio: ${audioResponse.status}`);
    }
    
    const audioData = audioResponse.data;

    res.json({
      transcript: transcript,
      aiResponse: responseText,
      audioUrl: audioData.url,
      voiceId: voiceId
    });
    
  } catch (err) {
    console.error('Voice-to-voice processing error:', err);
    res.status(500).json({ 
      error: 'Voice-to-voice processing failed', 
      details: err.message 
    });
  }
});

// Voice Management
// app.get('/voices', async (req, res) => {
//   try {
//     const voices = await getAvailableVoices();
//     res.json({ voices });
//   } catch (error) {
//     console.error('Fetch voices error:', error);
//     res.status(500).json({ error: 'Failed to fetch voices' });
//   }
// });
app.get('/voices', async (req, res) => {
  try {
    if (!process.env.ELEVENLABS_API_KEY) {
      return res.status(500).json({
        error: 'ElevenLabs API key not configured',
        message: 'Please set ELEVENLABS_API_KEY environment variable'
      });
    }

    console.log('🔍 Fetching voices from ElevenLabs API...');
    
    // Use the existing getAvailableVoices helper function that properly handles axios and proxy
    const voices = await getAvailableVoices();
    
    console.log(`✅ Successfully fetched ${voices.length} voices from ElevenLabs`);

    // Transform the data to include additional metadata
    const transformedVoices = voices.map(voice => ({
      voice_id: voice.voice_id,
      name: voice.name,
      category: voice.category || 'premade',
      labels: voice.labels || {
        gender: 'unknown',
        accent: 'unknown',
        description: '',
        age: 'unknown',
        use_case: ''
      }
    }));

    // Filter for premade/stock voices (exclude cloned voices)
    const stockVoices = transformedVoices.filter(voice => 
      voice.category === 'premade' || voice.category === 'professional'
    );

    // Sort voices by gender and then by name
    stockVoices.sort((a, b) => {
      const genderA = a.labels?.gender || 'unknown';
      const genderB = b.labels?.gender || 'unknown';
      
      if (genderA !== genderB) {
        return genderA.localeCompare(genderB);
      }
      return a.name.localeCompare(b.name);
    });

    console.log(`📋 Returning ${stockVoices.length} stock voices`);
    
    res.json({
      voices: stockVoices,
      total_count: stockVoices.length,
      categories: [...new Set(stockVoices.map(v => v.category))],
      genders: [...new Set(stockVoices.map(v => v.labels?.gender || 'unknown'))]
    });

  } catch (error) {
    console.error('❌ Error fetching voices:', error);
    
    // Provide more detailed error information
    let errorDetails = error.message;
    if (error.code === 'ECONNRESET') {
      errorDetails = 'Connection was reset by the server. This might be a network connectivity issue.';
    } else if (error.code === 'ETIMEDOUT' || error.code === 'UND_ERR_CONNECT_TIMEOUT') {
      errorDetails = 'Connection timed out. Please check your internet connection and try again.';
    } else if (error.code === 'ENOTFOUND') {
      errorDetails = 'Could not resolve ElevenLabs API domain. Please check your internet connection.';
    }
    
    res.status(500).json({
      error: 'Failed to fetch voices from ElevenLabs API',
      message: errorDetails,
      code: error.code,
      suggestion: 'Try checking your internet connection, API key, or network firewall settings'
    });
  }
});

// New endpoint to preview stock voices
app.post('/preview-voice', async (req, res) => {
  try {
    const { voiceId } = req.body;
    
    if (!voiceId) {
      return res.status(400).json({
        error: 'voiceId is required in the request body'
      });
    }
    
    console.log(`🔍 Fetching detailed voice information for voice ID: ${voiceId}`);
    
    // Make a direct API call to get detailed voice information including samples
    const httpsProxy = process.env.HTTPS_PROXY || process.env.https_proxy;
    const agent = httpsProxy ? new HttpsProxyAgent(httpsProxy) : undefined;

    const response = await axios.get(`https://api.elevenlabs.io/v1/voices/${voiceId}`, {
      headers: { 'xi-api-key': process.env.ELEVENLABS_API_KEY },
      httpsAgent: agent,
      timeout: 20000,
      validateStatus: () => true,
    });

    if (response.status === 404) {
      return res.status(404).json({
        error: 'Voice not found',
        message: `No voice found with ID: ${voiceId}`
      });
    }

    if (response.status !== 200) {
      throw new Error(`API returned status ${response.status}: ${response.data?.detail?.message || 'Failed to fetch voice details'}`);
    }

    const voice = response.data;
    
    // Check if this is a stock voice
    if (!['premade', 'professional'].includes(voice.category)) {
      return res.status(400).json({
        error: 'Invalid voice category',
        message: 'Selected voice is not a stock voice'
      });
    }
    
    // Get the sample audio URL
    let audioUrl;
    
    // Use preview_url if available (this is the direct sample URL)
    if (voice.preview_url) {
      audioUrl = voice.preview_url;
    }
    // Use samples array if available
    else if (voice.samples && voice.samples.length > 0) {
      // Get the first sample's download URL
      audioUrl = voice.samples[0].sample_url || voice.samples[0].audio_url;
    }
    
    if (!audioUrl) {
      return res.status(404).json({
        error: 'No audio sample available',
        message: 'No audio samples found for this voice'
      });
    }
    
    console.log(`✅ Found sample audio for voice ${voice.name}: ${audioUrl}`);
    
    // Return voice information with audio sample
    res.json({
      voice: {
        voice_id: voice.voice_id,
        name: voice.name,
        category: voice.category,
        labels: voice.labels || {
          gender: 'unknown',
          accent: 'unknown',
          description: '',
          age: 'unknown',
          use_case: ''
        },
        audioUrl: audioUrl
      }
    });

  } catch (error) {
    console.error('❌ Error fetching voice sample:', error);
    
    // Provide more detailed error information
    let errorDetails = error.message;
    if (error.code === 'ECONNRESET') {
      errorDetails = 'Connection was reset by the server. This might be a network connectivity issue.';
    } else if (error.code === 'ETIMEDOUT' || error.code === 'UND_ERR_CONNECT_TIMEOUT') {
      errorDetails = 'Connection timed out. Please check your internet connection and try again.';
    } else if (error.code === 'ENOTFOUND') {
      errorDetails = 'Could not resolve ElevenLabs API domain. Please check your internet connection.';
    }
    
    res.status(500).json({
      error: 'Failed to fetch voice sample from ElevenLabs API',
      message: errorDetails,
      code: error.code,
      suggestion: 'Try checking your internet connection, API key, or network firewall settings'
    });
  }
});

// New endpoint to generate audio from text
app.post('/generate-audio-with-text', async (req, res) => {
  try {
    const { voiceId, text } = req.body;
    
    if (!voiceId || !text) {
      return res.status(400).json({ 
        error: 'voiceId and text are required' 
      });
    }

    console.log(`🎙️ Generating audio for voice: ${voiceId}`);
    
    // Generate audio using axios with retry logic
    const httpsProxy = process.env.HTTPS_PROXY || process.env.https_proxy;
    const agent = httpsProxy ? new HttpsProxyAgent(httpsProxy) : undefined;

    const maxRetries = 3;
    let audioResponse;
    let lastError;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`Attempting audio generation (attempt ${attempt}/${maxRetries})...`);
        
        audioResponse = await axios.post(
          `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
          {
            text: text,
            model_id: 'eleven_monolingual_v1',
            voice_settings: {
              stability: 0.5,
              similarity_boost: 0.5
            }
          },
          {
            headers: {
              'Accept': 'audio/mpeg',
              'Content-Type': 'application/json',
              'xi-api-key': process.env.ELEVENLABS_API_KEY
            },
            responseType: 'arraybuffer',
            timeout: 30000,
            httpsAgent: agent,
            validateStatus: () => true
          }
        );

        if (audioResponse.status === 200) {
          break; // Success, exit retry loop
        } else {
          throw new Error(`API returned status ${audioResponse.status}: ${audioResponse.data || 'Unknown error'}`);
        }
      } catch (error) {
        lastError = error;
        console.warn(`Attempt ${attempt} failed:`, error.message);
        
        if (attempt < maxRetries) {
          const delayMs = Math.pow(2, attempt) * 1000;
          console.log(`Retrying in ${delayMs}ms...`);
          await new Promise(resolve => setTimeout(resolve, delayMs));
        }
      }
    }

    if (!audioResponse || audioResponse.status !== 200) {
      throw new Error(`Failed to generate audio after ${maxRetries} attempts. Last error: ${lastError?.message || 'Unknown error'}`);
    }

    // Save audio file
    const audioBuffer = audioResponse.data;
    const fileName = `preview_${Date.now()}.mp3`;
    const filePath = path.join(__dirname, 'audio', fileName);
    
    // Ensure directory exists
    if (!fs.existsSync(path.dirname(filePath))) {
      fs.mkdirSync(path.dirname(filePath), { recursive: true });
    }
    
    // Write file
    fs.writeFileSync(filePath, Buffer.from(audioBuffer));
    
    console.log(`✅ Audio generated successfully: ${fileName}`);
    
    // Return public URL
    res.json({
      audioUrl: `/audio/${fileName}`
    });

  } catch (error) {
    console.error('❌ Audio generation error:', error);
    res.status(500).json({ 
      error: 'Failed to generate audio',
      details: error.message
    });
  }
});

// Cleanup old preview files (optional)
setInterval(() => {
  const audioDir = path.join(__dirname, 'audio');
  fs.readdir(audioDir, (err, files) => {
    if (err) return;
    
    files.forEach(file => {
      if (file.startsWith('preview_')) {
        const filePath = path.join(audioDir, file);
        fs.stat(filePath, (err, stats) => {
          if (err) return;
          
          // Delete files older than 1 hour
          const ageMs = Date.now() - stats.mtime.getTime();
          if (ageMs > 60 * 60 * 1000) {
            fs.unlink(filePath, () => {});
          }
        });
      }
    });
  });
}, 30 * 60 * 1000);

app.post('/check-voice', async (req, res) => {
  try {
    const { voiceName } = req.body;
    const voices = await getAvailableVoices();
    const existingVoice = voices.find(
      v => v.name.toLowerCase() === voiceName.toLowerCase()
    );
    res.json({ exists: !!existingVoice, voice: existingVoice });
  } catch (error) {
    console.error('Voice check error:', error);
    res.status(500).json({ error: 'Failed to check voices' });
  }
});

// LiveKit Endpoints
app.post('/token', async (req, res) => {
  try {
    const { roomName, participantName } = req.body;
    
    if (!roomName || !participantName) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const token = await generateToken(
      process.env.LIVEKIT_API_KEY,
      process.env.LIVEKIT_API_SECRET,
      roomName,
      participantName
    );

    // Track room and participant
    if (!activeRooms.has(roomName)) {
      activeRooms.set(roomName, {
        name: roomName,
        createdAt: new Date().toISOString(),
        participantCount: 0
      });
      roomParticipants.set(roomName, new Set());
    }
    roomParticipants.get(roomName).add(participantName);
    activeRooms.get(roomName).participantCount = roomParticipants.get(roomName).size;

    res.json({ token });
  } catch (error) {
    console.error('Token error:', error);
    res.status(500).json({ error: 'Failed to generate token' });
  }
});

// WebSocket Setup
const server = app.listen(3001, () => {
  console.log('Server running on port 3001');
});

const io = socketio(server, { cors: { origin: "*" } });

io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  socket.on('join-room', (payload) => {
    // Accept either a string roomName or an object { roomName, participantName }
    let roomName = payload;
    let participantName;
    if (payload && typeof payload === 'object') {
      roomName = payload.roomName;
      participantName = payload.participantName;
    }

    socket.join(roomName);
    // Store participant name on the socket for later use (e.g. transcriptions)
    if (participantName) socket.data.participantName = participantName;

    console.log(`Client ${socket.id} joined room: ${roomName}`, { participantName });
    // Ensure room opt-in set exists
    if (!roomSttOptIns.has(roomName)) roomSttOptIns.set(roomName, new Set());
  });

  // Allow clients to set their STT preference (opt-in to receiving transcriptions)
  socket.on('set-stt-preference', ({ roomName, enabled }) => {
    try {
      if (!roomName) return;
      if (!roomSttOptIns.has(roomName)) roomSttOptIns.set(roomName, new Set());
      const set = roomSttOptIns.get(roomName);
      if (enabled) set.add(socket.id);
      else set.delete(socket.id);
      console.log(`STT preference updated for ${socket.id} in ${roomName}: ${enabled}`);
      // Optionally acknowledge
      socket.emit('stt-preference-updated', { roomName, enabled });
    } catch (e) {
      console.warn('Failed to set STT preference', e.message);
    }
  });

  // Request a one-off transcript for a particular speaker/trackId
  socket.on('request-transcript', ({ roomName, trackId, requestId }) => {
    try {
      if (!roomName || !trackId) return socket.emit('request-error', { message: 'roomName and trackId required' });
      // store requester so when transcript is ready we can send directly
      if (requestId) transcriptRequestMap.set(requestId, socket.id);
      // notify the room (or specific recorder) to start sending audio for transcription
      // We'll emit a `start-transcription-for` event which clients that own the track can respond to
      console.log(`Transcript requested by ${socket.id} for track ${trackId} in ${roomName} (requestId=${requestId})`);
      io.to(roomName).emit('start-transcription-for', { trackId, requestId });
    } catch (e) {
      console.warn('Failed to request transcript', e.message);
    }
  });

  // Enhanced chat message handling with AI response capability
  socket.on('chat-message', async ({ message, roomName, participantName, voiceId, enableAI }) => {
    try {
      console.log(`📨 Received chat message:`, {
        message,
        roomName,
        participantName,
        voiceId,
        enableAI
      });
      
      // Broadcast the original message to the room
      socket.to(roomName).emit('chat-message', { 
        message, 
        participantName, 
        timestamp: new Date().toISOString() 
      });
      
      console.log(`📤 Broadcasted original message to room: ${roomName}`);
      
      // If AI is enabled and voiceId is provided, generate AI response
      if (enableAI && voiceId && message.trim()) {
        console.log(`🤖 AI enabled for message: "${message}" with voice: ${voiceId}`);
        
        // Add a small delay to let the original message appear first
        setTimeout(async () => {
          try {
            console.log(`🧠 Starting AI response generation...`);
            
            // Generate AI response using the same logic as /ai-chat-response
            const model = await getAIModel();
            const prompt = `You are a helpful AI assistant in a live chat. Respond naturally and conversationally to this message: "${message}". Keep your response concise and engaging, suitable for voice output.`;
            
            console.log(`🎯 Sending prompt to AI model...`);
            const aiResult = await model.generateContent(prompt);
            const aiResponse = await aiResult.response;
            const responseText = aiResponse.text() || "I'm sorry, I couldn't generate a response right now.";
            
            console.log(`✅ AI generated response: "${responseText}"`);
            
            // Generate audio for AI response
            console.log(`🎵 Starting audio generation with voiceId: ${voiceId}`);
            const httpsProxy = process.env.HTTPS_PROXY || process.env.https_proxy;
            const agent = httpsProxy ? new HttpsProxyAgent(httpsProxy) : undefined;

            const audioResponse = await axios.post(
              `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
              {
                text: responseText,
                model_id: 'eleven_monolingual_v1',
                voice_settings: {
                  stability: 0.5,
                  similarity_boost: 0.5
                }
              },
              {
                headers: {
                  'Accept': 'audio/mpeg',
                  'Content-Type': 'application/json',
                  'xi-api-key': process.env.ELEVENLABS_API_KEY
                },
                responseType: 'arraybuffer',
                timeout: 30000,
                httpsAgent: agent,
                validateStatus: () => true
              }
            );

            console.log(`🎤 Audio generation response status: ${audioResponse.status}`);

            if (audioResponse.status === 200) {
              // Save audio file
              const audioBuffer = audioResponse.data;
              const fileName = `ai_chat_${Date.now()}.mp3`;
              const filePath = path.join(__dirname, 'audio', fileName);
              
              console.log(`💾 Saving audio file: ${fileName}`);
              
              if (!fs.existsSync(path.dirname(filePath))) {
                fs.mkdirSync(path.dirname(filePath), { recursive: true });
              }
              
              fs.writeFileSync(filePath, Buffer.from(audioBuffer));
              
              const responseData = {
                aiResponse: responseText,
                audioUrl: `/audio/${fileName}`,
                voiceId: voiceId,
                originalMessage: message,
                participantName: 'AI Assistant',
                timestamp: new Date().toISOString()
              };
              
              console.log(`📡 Broadcasting AI response to room ${roomName}:`, responseData);
              
              // Broadcast AI response to room
              io.to(roomName).emit('ai-chat-response', responseData);
              
              console.log(`✅ AI response broadcasted successfully to room: ${roomName}`);
            } else {
              console.error(`❌ Audio generation failed with status: ${audioResponse.status}`);
              console.error(`Response data:`, audioResponse.data);
            }
          } catch (aiError) {
            console.error('❌ AI response generation failed:', aiError);
            console.error('Error stack:', aiError.stack);
            // Optionally send error message to room
            socket.to(roomName).emit('ai-error', {
              error: 'AI response failed',
              originalMessage: message,
              details: aiError.message
            });
          }
        }, 1000); // 1 second delay
      } else {
        console.log(`⏭️ Skipping AI response - enableAI: ${enableAI}, voiceId: ${voiceId}, message: "${message}"`);
      }
    } catch (err) {
      console.error('❌ Chat message handling error:', err);
      console.error('Error stack:', err.stack);
    }
  });

  socket.on('voice-message', async ({ text, voiceId, roomName }) => {
    try {
      const voices = await getAvailableVoices();
      if (!voices.some(v => v.voice_id === voiceId)) {
        throw new Error('Invalid voiceId');
      }
      const audioUrl = await synthesizeAndStore(text, voiceId);
      socket.to(roomName).emit('voice-message', { text, audioUrl, voiceId });
    } catch (err) {
      console.error('Voice message error:', err);
    }
  });

  // Handle real-time audio transcription
  socket.on('audio-transcribe', async ({ audioBuffer, roomName, mimetype, trackId, requestId }) => {
    try {
      // Register/refresh active recorder
      const participantName = socket.data && socket.data.participantName ? socket.data.participantName : socket.id;
      const recorderKey = trackId || `${socket.id}-${Date.now()}`;
      activeRecorders.set(recorderKey, {
        participantName,
        socketId: socket.id,
        trackId: trackId || null,
        lastActiveAt: Date.now()
      });

      const dgMime = mimetype || 'audio/wav';
      const { result, error } = await deepgram.listen.prerecorded.transcribeFile(
        Buffer.from(audioBuffer),
        {
          model: 'nova-2',
          language: 'en-US',
          smart_format: true,
          punctuate: true,
          mimetype: dgMime
        }
      );

      if (error) {
        throw error;
      }

      const transcript = result?.results?.channels?.[0]?.alternatives?.[0]?.transcript || '';

      if (transcript.trim()) {
        const messagePayload = {
          message: transcript,
          participantName,
          timestamp: new Date().toISOString(),
          source: 'speech',
          trackId: trackId || null
        };

        // If this transcription was requested specifically (requestId), send only to requester
        if (requestId && transcriptRequestMap.has(requestId)) {
          const requesterSocketId = transcriptRequestMap.get(requestId);
          const requesterSocket = io.sockets.sockets.get(requesterSocketId);
          if (requesterSocket) {
            requesterSocket.emit('transcript-result', { requestId, ...messagePayload });
          }
          // cleanup the request mapping to avoid leaks
          transcriptRequestMap.delete(requestId);
        }

        // Forward to opted-in participants in the room
        if (roomName && roomSttOptIns.has(roomName)) {
          const optInSet = roomSttOptIns.get(roomName);
          optInSet.forEach(sid => {
            const s = io.sockets.sockets.get(sid);
            if (s) s.emit('chat-message', messagePayload);
          });
        }

        // Also emit a server-side event for visibility in the room (optional)
        socket.to(roomName).emit('audio-transcribed', {
          transcript,
          socketId: socket.id,
          participantName,
          trackId: trackId || null
        });
      }
    } catch (err) {
      console.error('Real-time transcription error:', err);
      socket.emit('transcription-error', { error: err.message });
    }
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Health Check
app.get('/health', async (req, res) => {
  const health = {
    status: 'healthy', 
    activeRooms: activeRooms.size,
    totalParticipants: Array.from(roomParticipants.values()).reduce((sum, p) => sum + p.size, 0),
    services: {
      elevenlabs: 'unknown',
      deepgram: process.env.DEEPGRAM_API_KEY ? 'configured' : 'not configured',
      livekit: process.env.LIVEKIT_API_KEY ? 'configured' : 'not configured',
  gemini: (process.env.GOOGLE_APPLICATION_CREDENTIALS || process.env.GEMINI_API_KEY) ? 'configured' : 'not configured',
  openai: OPENAI_API_KEY ? 'configured' : 'not configured'
    }
  };

  // Test ElevenLabs connection
  if (process.env.ELEVENLABS_API_KEY) {
    try {
      const httpsProxy = process.env.HTTPS_PROXY || process.env.https_proxy;
      const agent = httpsProxy ? new HttpsProxyAgent(httpsProxy) : undefined;
      
      const response = await axios.get('https://api.elevenlabs.io/v1/voices', {
        headers: { 'xi-api-key': process.env.ELEVENLABS_API_KEY },
        httpsAgent: agent,
        timeout: 10000,
        validateStatus: () => true,
      });
      
      if (response.status === 200) {
        health.services.elevenlabs = 'healthy';
        health.voiceCount = response.data?.voices?.length || 0;
      } else {
        health.services.elevenlabs = `error_${response.status}`;
      }
    } catch (error) {
      health.services.elevenlabs = `error: ${error.message}`;
    }
  } else {
    health.services.elevenlabs = 'not configured';
  }

  res.json(health);
});

// Graceful Shutdown
process.on('SIGTERM', () => {
  console.log('Shutting down gracefully');
  server.close(() => process.exit(0));
});

process.on('SIGINT', () => {
  console.log('Shutting down gracefully');
  server.close(() => process.exit(0));
});

// Debug endpoint: list active recorders
app.get('/debug-recorders', (req, res) => {
  try {
    const list = [];
    for (const [key, info] of activeRecorders.entries()) {
      list.push({ id: key, ...info, lastActiveAgoMs: Date.now() - info.lastActiveAt });
    }
    res.json({ count: list.length, recorders: list });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Cleanup inactive recorders every 30s (remove entries older than 60s)
setInterval(() => {
  const now = Date.now();
  for (const [key, info] of activeRecorders.entries()) {
    if (now - info.lastActiveAt > 60 * 1000) {
      activeRecorders.delete(key);
    }
  }
}, 30 * 1000);