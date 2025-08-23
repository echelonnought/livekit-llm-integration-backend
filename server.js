require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { RoomServiceClient, AccessToken } = require('livekit-server-sdk');
const { OpenAI } = require('openai');
const { ElevenLabsClient } = require('@elevenlabs/elevenlabs-js');
const socketio = require('socket.io');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');
const { HttpsProxyAgent } = require('https-proxy-agent');

// Initialize services
const app = express();
const upload = multer({ storage: multer.memoryStorage() });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const elevenlabs = new ElevenLabsClient({ apiKey: process.env.ELEVENLABS_API_KEY });
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
app.use('/audio', express.static('audio'));

// Ensure directories exist
if (!fs.existsSync('audio')) fs.mkdirSync('audio');

// Helper Functions
async function getAvailableVoices() {
  const httpsProxy = process.env.HTTPS_PROXY || process.env.https_proxy;
  const agent = httpsProxy ? new HttpsProxyAgent(httpsProxy) : undefined;

  const response = await axios.get('https://api.elevenlabs.io/v1/voices', {
    headers: { 'xi-api-key': process.env.ELEVENLABS_API_KEY },
    httpsAgent: agent,
    timeout: 15000,
    validateStatus: () => true,
  });

  if (response.status !== 200) {
    throw new Error('Failed to fetch voices');
  }

  return (response.data?.voices || []).map(v => ({
    voice_id: v.voice_id,
    name: v.name,
    category: v.category,
    labels: v.labels,
  }));
}

async function synthesizeAndStore(text, voiceId) {
  const audioPath = path.join('audio', `msg-${Date.now()}.mp3`);
  const audioStream = await elevenlabs.generate({
    voice: voiceId,
    text: text,
    model_id: "eleven_multilingual_v2",
    stream: true
  });

  return new Promise((resolve) => {
    const writeStream = fs.createWriteStream(audioPath);
    audioStream.pipe(writeStream);
    writeStream.on('finish', () => resolve(`/audio/${path.basename(audioPath)}`));
  });
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

    const audioPath = path.join('audio', `${voiceId}-${Date.now()}.mp3`);
    const audioStream = await elevenlabs.generate({
      voice: voiceId,
      text: text,
      model_id: "eleven_multilingual_v2",
      stream: true
    });

    const writeStream = fs.createWriteStream(audioPath);
    audioStream.pipe(writeStream);

    writeStream.on('finish', () => {
      res.json({ url: `/audio/${path.basename(audioPath)}` });
    });
  } catch (err) {
    console.error('Synthesis error:', err);
    res.status(500).json({ error: 'Speech synthesis failed' });
  }
});

// AI Processing
app.post('/ask-ai', async (req, res) => {
  try {
    const { messages, voiceId } = req.body;
    const voices = await getAvailableVoices();
    
    if (!voices.some(v => v.voice_id === voiceId)) {
      return res.status(400).json({ error: 'Invalid voiceId' });
    }

    const prompt = messages.map(m => m.content).join('\n');
    const ollamaResponse = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'llama3',
        prompt: prompt,
        stream: false
      })
    });

    const data = await ollamaResponse.json();
    const responseText = data.response || "No response from AI";

    const audioResponse = await fetch(
      `${req.protocol}://${req.get('host')}/synthesize?text=${
        encodeURIComponent(responseText)
      }&voiceId=${voiceId}`
    );
    const audioData = await audioResponse.json();

    res.json({
      text: responseText,
      audioUrl: audioData.url,
      voiceId: voiceId
    });
  } catch (err) {
    console.error('AI processing error:', err);
    res.status(500).json({ error: 'AI processing failed' });
  }
});

// Voice Management
app.get('/voices', async (req, res) => {
  try {
    const voices = await getAvailableVoices();
    res.json({ voices });
  } catch (error) {
    console.error('Fetch voices error:', error);
    res.status(500).json({ error: 'Failed to fetch voices' });
  }
});

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
  
  socket.on('join-room', (roomName) => {
    socket.join(roomName);
    console.log(`Client ${socket.id} joined room: ${roomName}`);
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

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Health Check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    activeRooms: activeRooms.size,
    totalParticipants: Array.from(roomParticipants.values()).reduce((sum, p) => sum + p.size, 0)
  });
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