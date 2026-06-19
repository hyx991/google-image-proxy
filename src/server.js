import express from "express";
import { GoogleGenAI } from "@google/genai";

const app = express();
app.use(express.json({ limit: process.env.JSON_LIMIT || "4mb" }));

const PORT = Number(process.env.PORT || 10000);
const DEFAULT_MODEL = process.env.GEMINI_MODEL || "gemini-3.1-flash-lite";
const MAX_VIDEO_BYTES = Number(process.env.MAX_VIDEO_BYTES || 80 * 1024 * 1024);
const FETCH_TIMEOUT_MS = Number(process.env.FETCH_TIMEOUT_MS || 120000);
const FILE_WAIT_TIMEOUT_MS = Number(process.env.FILE_WAIT_TIMEOUT_MS || 240000);
const FILE_POLL_MS = Number(process.env.FILE_POLL_MS || 3000);
const RELAY_AUTH_TOKEN = String(process.env.RELAY_AUTH_TOKEN || "").trim();

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type,Authorization,X-Goog-Api-Key");
}

function json(res, status, payload) {
  setCors(res);
  res.status(status).json(payload);
}

function getApiKey(req) {
  return String(
    req.get("x-goog-api-key") ||
      req.get("x-google-api-key") ||
      process.env.GOOGLE_API_KEY ||
      "",
  ).trim();
}

function ensureAuthorized(req, res) {
  if (!RELAY_AUTH_TOKEN) {
    return true;
  }

  const auth = String(req.get("authorization") || "");
  const token = auth.startsWith("Bearer ") ? auth.slice("Bearer ".length).trim() : "";
  if (token === RELAY_AUTH_TOKEN) {
    return true;
  }

  json(res, 401, {
    ok: false,
    error: "unauthorized",
    message: "Missing or invalid relay bearer token",
  });
  return false;
}

function truncate(value, max = 1600) {
  const text = String(value ?? "");
  return text.length <= max ? text : `${text.slice(0, max)}...<truncated>`;
}

function extractText(result) {
  if (typeof result?.text === "string" && result.text.trim()) {
    return result.text.trim();
  }

  const candidates = Array.isArray(result?.candidates) ? result.candidates : [];
  const parts = [];
  for (const candidate of candidates) {
    for (const part of candidate?.content?.parts || []) {
      if (typeof part?.text === "string" && part.text.trim()) {
        parts.push(part.text.trim());
      }
    }
  }
  return parts.join("\n").trim();
}

async function fetchVideo(videoUrl) {
  if (!/^https?:\/\//i.test(videoUrl)) {
    throw new Error("videoUrl must be an http/https URL");
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort("timeout"), FETCH_TIMEOUT_MS);
  try {
    const response = await fetch(videoUrl, { signal: controller.signal });
    if (!response.ok) {
      throw new Error(`video download failed: HTTP ${response.status}`);
    }

    const contentLength = Number(response.headers.get("content-length") || 0);
    if (contentLength > MAX_VIDEO_BYTES) {
      throw new Error(`video is too large: ${contentLength} bytes`);
    }

    const bytes = Buffer.from(await response.arrayBuffer());
    if (!bytes.byteLength) {
      throw new Error("video payload is empty");
    }
    if (bytes.byteLength > MAX_VIDEO_BYTES) {
      throw new Error(`video is too large: ${bytes.byteLength} bytes`);
    }

    return {
      bytes,
      mimeType: response.headers.get("content-type")?.split(";")[0] || "video/mp4",
    };
  } finally {
    clearTimeout(timer);
  }
}

async function waitForFileReady(ai, file) {
  const deadline = Date.now() + FILE_WAIT_TIMEOUT_MS;
  let current = file;

  while (Date.now() < deadline) {
    if (current?.state === "ACTIVE" || current?.state === "SUCCEEDED") {
      return current;
    }
    if (current?.state === "FAILED") {
      throw new Error("Gemini file processing failed");
    }

    await new Promise((resolve) => setTimeout(resolve, FILE_POLL_MS));
    current = await ai.files.get({ name: current.name });
  }

  throw new Error("Gemini file processing timed out");
}

function buildPrompt(userPrompt) {
  if (userPrompt && String(userPrompt).trim()) {
    return String(userPrompt).trim();
  }

  return [
    "你是短视频内容分析师。请理解这个视频，并输出结构化中文分析。",
    "请包含：",
    "1. 视频主题",
    "2. 画面/分镜摘要",
    "3. 可能的口播或字幕重点",
    "4. 爆点与情绪钩子",
    "5. 可复用脚本结构",
    "6. 适合本地门店老板改编的拍摄建议",
  ].join("\n");
}

app.options("*", (req, res) => {
  setCors(res);
  res.status(204).send();
});

app.get("/", (req, res) => {
  json(res, 200, {
    ok: true,
    service: "gemini-video-understand",
    routes: ["/", "/health", "/video-understand"],
    defaultModel: DEFAULT_MODEL,
    maxVideoBytes: MAX_VIDEO_BYTES,
    relayTokenRequired: Boolean(RELAY_AUTH_TOKEN),
  });
});

app.get("/health", (req, res) => {
  json(res, 200, {
    ok: true,
    service: "gemini-video-understand",
    uptimeSeconds: Math.round(process.uptime()),
  });
});

app.post("/video-understand", async (req, res) => {
  if (!ensureAuthorized(req, res)) {
    return;
  }

  const apiKey = getApiKey(req);
  if (!apiKey) {
    return json(res, 400, {
      ok: false,
      error: "missing_google_api_key",
      message: "Provide x-goog-api-key or configure GOOGLE_API_KEY",
    });
  }

  const videoUrl = String(req.body?.videoUrl || req.body?.url || "").trim();
  const prompt = buildPrompt(req.body?.prompt);
  const model = String(req.body?.model || DEFAULT_MODEL).trim() || DEFAULT_MODEL;
  if (!videoUrl) {
    return json(res, 400, {
      ok: false,
      error: "missing_video_url",
      message: "Provide videoUrl",
    });
  }

  try {
    const ai = new GoogleGenAI({ apiKey });
    const video = await fetchVideo(videoUrl);
    const file = await ai.files.upload({
      file: new Blob([video.bytes], { type: video.mimeType }),
      config: {
        mimeType: video.mimeType,
        displayName: `video-${Date.now()}`,
      },
    });
    const readyFile = await waitForFileReady(ai, file);
    const result = await ai.models.generateContent({
      model,
      contents: [
        {
          role: "user",
          parts: [
            {
              fileData: {
                mimeType: readyFile.mimeType || video.mimeType,
                fileUri: readyFile.uri,
              },
            },
            { text: prompt },
          ],
        },
      ],
    });

    const text = extractText(result);
    return json(res, 200, {
      ok: true,
      mode: "live",
      source: "gemini_files_video_understand",
      model,
      text,
      result: text,
      file: {
        name: readyFile.name,
        uri: readyFile.uri,
        mimeType: readyFile.mimeType || video.mimeType,
        state: readyFile.state,
      },
      usage: result?.usageMetadata || null,
    });
  } catch (error) {
    return json(res, 502, {
      ok: false,
      mode: "failed",
      source: "gemini_files_video_understand",
      error: "video_understand_failed",
      message: error instanceof Error ? error.message : String(error),
      detail: truncate(error?.stack || error),
    });
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`gemini-video-understand listening on :${PORT}`);
});
