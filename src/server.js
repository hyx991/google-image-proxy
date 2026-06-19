import express from "express";
import { GoogleGenAI } from "@google/genai";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { spawn } from "node:child_process";

const app = express();
app.use(express.json({ limit: "2mb" }));

const PORT = Number(process.env.PORT || 10000);
const MAX_INPUT_BYTES = Number(process.env.MAX_INPUT_BYTES || 120 * 1024 * 1024);
const MAX_CLIP_SECONDS = Number(process.env.MAX_CLIP_SECONDS || 600);
const FETCH_TIMEOUT_MS = Number(process.env.FETCH_TIMEOUT_MS || 120000);
const GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_IMAGE_MODEL =
  process.env.IMAGE_MODEL || "gemini-3.1-flash-image-preview";
const DEFAULT_VIDEO_MODEL =
  process.env.VIDEO_MODEL || "veo-3.1-generate-preview";
const DEFAULT_TRANSLATION_MODEL =
  process.env.TRANSLATION_MODEL || "gemini-2.5-flash";
const DEFAULT_TEXT_MODEL =
  process.env.TEXT_MODEL || "gemini-3.1-flash-lite";
const GOOGLE_IMAGE_TIMEOUT_MS = Number(process.env.GOOGLE_IMAGE_TIMEOUT_MS || 180000);
const GOOGLE_TEXT_TIMEOUT_MS = Number(process.env.GOOGLE_TEXT_TIMEOUT_MS || 180000);
const GOOGLE_VIDEO_TIMEOUT_MS = Number(process.env.GOOGLE_VIDEO_TIMEOUT_MS || 900000);
const GOOGLE_FILE_WAIT_TIMEOUT_MS = Number(process.env.GOOGLE_FILE_WAIT_TIMEOUT_MS || 240000);
const GOOGLE_POLL_MS = Number(process.env.GOOGLE_POLL_MS || 10000);
const RELAY_AUTH_TOKEN = String(process.env.RELAY_AUTH_TOKEN || "").trim();
const CJK_RE =
  /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\u31f0-\u31ff\uac00-\ud7af]/u;

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type,Authorization,X-Goog-Api-Key");
}

function json(res, status, payload) {
  setCors(res);
  res.status(status).json(payload);
}

function encodeMetadataHeader(metadata) {
  return encodeURIComponent(JSON.stringify(metadata));
}

function truncate(value, max = 1200) {
  const input = String(value ?? "");
  if (input.length <= max) {
    return input;
  }
  return `${input.slice(0, max)}...<truncated>`;
}

function normalizeImageModel(model) {
  if (!model) {
    return DEFAULT_IMAGE_MODEL;
  }
  if (model === "BANNER_2") {
    return "gemini-3.1-flash-image-preview";
  }
  return model;
}

function normalizeVideoModel(model) {
  if (!model) {
    return DEFAULT_VIDEO_MODEL;
  }
  if (model === "veo-3.0") {
    return "veo-3.1-generate-preview";
  }
  return model;
}

function normalizeVideoUnderstandModel(model) {
  return String(model || DEFAULT_TEXT_MODEL).trim() || DEFAULT_TEXT_MODEL;
}

function resolveGoogleApiKey(req) {
  return String(
    req.get("x-goog-api-key") ||
      req.get("x-google-api-key") ||
      process.env.GOOGLE_API_KEY ||
      "",
  ).trim();
}

function ensureRelayAuthorized(req, res) {
  if (!RELAY_AUTH_TOKEN) {
    return true;
  }

  const authorization = String(req.get("authorization") || "");
  const token = authorization.startsWith("Bearer ")
    ? authorization.slice("Bearer ".length).trim()
    : "";

  if (token !== RELAY_AUTH_TOKEN) {
    json(res, 401, {
      success: false,
      error: "unauthorized",
      message: "Missing or invalid relay bearer token",
    });
    return false;
  }

  return true;
}

function parseClipParams(req) {
  const src = req.method === "POST" ? req.body || {} : req.query || {};
  const videoUrl = String(src.video_url || src.videoUrl || "").trim();
  const start = Number(src.start ?? 0);
  const end = Number(src.end);
  return { videoUrl, start, end };
}

function validateClipParams({ videoUrl, start, end }) {
  if (!videoUrl || !/^https?:\/\//i.test(videoUrl)) {
    return "video_url must be an http/https URL";
  }
  if (!Number.isFinite(start) || !Number.isFinite(end)) {
    return "start/end must be numbers";
  }
  if (start < 0 || end <= 0 || start >= end) {
    return "invalid range: require 0 <= start < end";
  }
  if (end - start > MAX_CLIP_SECONDS) {
    return `clip length cannot exceed ${MAX_CLIP_SECONDS} seconds`;
  }
  return null;
}

async function downloadInput(videoUrl, targetFile) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const resp = await fetch(videoUrl, { signal: controller.signal });
    if (!resp.ok) {
      throw new Error(`video download failed, HTTP ${resp.status}`);
    }
    const contentLength = Number(resp.headers.get("content-length") || 0);
    if (contentLength && contentLength > MAX_INPUT_BYTES) {
      throw new Error("video is too large");
    }
    const bytes = new Uint8Array(await resp.arrayBuffer());
    if (!bytes.byteLength) {
      throw new Error("empty video payload");
    }
    if (bytes.byteLength > MAX_INPUT_BYTES) {
      throw new Error("video is too large");
    }
    await fs.writeFile(targetFile, bytes);
  } finally {
    clearTimeout(timer);
  }
}

async function runFfmpeg(args) {
  await new Promise((resolve, reject) => {
    const p = spawn("ffmpeg", args, { stdio: ["ignore", "pipe", "pipe"] });
    let err = "";
    p.stderr.on("data", (chunk) => {
      err += chunk.toString();
    });
    p.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(err || `ffmpeg exited with code ${code}`));
      }
    });
    p.on("error", (e) => reject(e));
  });
}

async function clipVideo({ inputFile, outputFile, start, end }) {
  try {
    await runFfmpeg([
      "-y",
      "-i",
      inputFile,
      "-ss",
      String(start),
      "-to",
      String(end),
      "-c:v",
      "copy",
      "-c:a",
      "copy",
      outputFile,
    ]);
    return "copy";
  } catch {
    await runFfmpeg([
      "-y",
      "-i",
      inputFile,
      "-ss",
      String(start),
      "-to",
      String(end),
      "-c:v",
      "libx264",
      "-preset",
      "veryfast",
      "-crf",
      "23",
      "-c:a",
      "aac",
      "-b:a",
      "128k",
      "-movflags",
      "+faststart",
      outputFile,
    ]);
    return "reencode";
  }
}

async function fetchWithTimeout(url, options = {}, timeoutMs = FETCH_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort("timeout"), timeoutMs);
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } finally {
    clearTimeout(timer);
  }
}

async function callGoogleJson(url, { apiKey, method = "GET", body, timeoutMs }) {
  try {
    const response = await fetchWithTimeout(
      url,
      {
        method,
        headers: {
          "Content-Type": "application/json; charset=utf-8",
          "x-goog-api-key": apiKey,
        },
        body: body ? JSON.stringify(body) : undefined,
      },
      timeoutMs,
    );

    const text = await response.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = null;
      }
    }

    return {
      ok: response.ok,
      status: response.status,
      text,
      payload,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const isTimeout = message === "timeout" || message.includes("aborted");
    return {
      ok: false,
      status: 0,
      text: message,
      payload: null,
      networkError: isTimeout ? "timeout" : "network_error",
    };
  }
}

function extractInlineImage(payload) {
  const candidates = Array.isArray(payload?.candidates) ? payload.candidates : [];
  for (const candidate of candidates) {
    const parts = Array.isArray(candidate?.content?.parts)
      ? candidate.content.parts
      : [];
    for (const part of parts) {
      if (part?.inlineData?.data) {
        return {
          data: part.inlineData.data,
          mimeType: part.inlineData.mimeType || "image/png",
        };
      }
    }
  }
  return null;
}

function extractTextParts(payload) {
  const candidates = Array.isArray(payload?.candidates) ? payload.candidates : [];
  const texts = [];
  for (const candidate of candidates) {
    const parts = Array.isArray(candidate?.content?.parts)
      ? candidate.content.parts
      : [];
    for (const part of parts) {
      if (typeof part?.text === "string" && part.text.trim()) {
        texts.push(part.text.trim());
      }
    }
  }
  return texts;
}

function normalizeTextModel(model) {
  return String(model || DEFAULT_TEXT_MODEL).trim() || DEFAULT_TEXT_MODEL;
}

function messagesToPrompt(messages) {
  if (!Array.isArray(messages)) {
    return "";
  }

  return messages
    .map((message) => {
      const role = String(message?.role || "user").trim();
      const content = message?.content;

      if (typeof content === "string") {
        return `${role}: ${content}`;
      }

      if (Array.isArray(content)) {
        const text = content
          .map((part) => {
            if (typeof part === "string") {
              return part;
            }
            if (typeof part?.text === "string") {
              return part.text;
            }
            return "";
          })
          .filter(Boolean)
          .join("\n");
        return text ? `${role}: ${text}` : "";
      }

      return "";
    })
    .filter(Boolean)
    .join("\n\n");
}

function buildTextPayload({ prompt, messages, temperature, maxOutputTokens }) {
  const text = String(prompt || "").trim() || messagesToPrompt(messages).trim();
  if (!text) {
    return null;
  }

  return {
    contents: [{ parts: [{ text }] }],
    generationConfig: {
      temperature: Number.isFinite(Number(temperature)) ? Number(temperature) : 0.7,
      maxOutputTokens: Number.isFinite(Number(maxOutputTokens))
        ? Number(maxOutputTokens)
        : 2048,
    },
  };
}

async function maybeRewritePrompt(prompt, apiKey) {
  if (!CJK_RE.test(prompt)) {
    return {
      originalPrompt: prompt,
      effectivePrompt: prompt,
      translationApplied: false,
      translationMode: "none",
    };
  }

  const rewriteInstruction = [
    "Rewrite the following image prompt into concise natural English for an image generation model.",
    "Preserve all visual details and intent.",
    "Return only the rewritten English prompt.",
    "",
    `User prompt: ${prompt}`,
  ].join("\n");

  const translationPayload = {
    contents: [{ parts: [{ text: rewriteInstruction }] }],
    generationConfig: {
      temperature: 0.2,
      topP: 0.8,
      maxOutputTokens: 120,
    },
  };

  const translation = await callGoogleJson(
    `${GOOGLE_API_BASE}/models/${encodeURIComponent(DEFAULT_TRANSLATION_MODEL)}:generateContent`,
    {
      apiKey,
      method: "POST",
      body: translationPayload,
      timeoutMs: FETCH_TIMEOUT_MS,
    },
  );

  if (!translation.ok) {
    return {
      error: translation.networkError || "translation_http_error",
      message: translation.networkError
        ? `Prompt rewrite failed: ${translation.text}`
        : `Prompt rewrite upstream ${translation.status}`,
    };
  }

  const translatedText = extractTextParts(translation.payload).join("\n").trim();
  if (!translatedText) {
    return {
      error: "translation_empty",
      message: "Prompt rewrite upstream returned no text",
    };
  }

  return {
    originalPrompt: prompt,
    effectivePrompt: translatedText,
    translationApplied: true,
    translationMode: "proxy-rewrite",
  };
}

async function fetchImageInlineData(imageUrl) {
  const response = await fetchWithTimeout(imageUrl, {}, FETCH_TIMEOUT_MS);
  if (!response.ok) {
    throw new Error(`Failed to fetch source image: ${response.status} ${response.statusText}`);
  }

  const mimeType = response.headers.get("content-type") || "image/png";
  const bytes = Buffer.from(await response.arrayBuffer());

  return {
    mimeType,
    data: bytes.toString("base64"),
  };
}

async function fetchVideoInlineData(videoUrl) {
  const response = await fetchWithTimeout(videoUrl, {}, FETCH_TIMEOUT_MS);
  if (!response.ok) {
    throw new Error(`Failed to fetch source video: ${response.status} ${response.statusText}`);
  }

  const mimeType = response.headers.get("content-type")?.split(";")[0] || "video/mp4";
  const bytes = Buffer.from(await response.arrayBuffer());
  if (!bytes.byteLength) {
    throw new Error("Source video is empty");
  }
  if (bytes.byteLength > MAX_INPUT_BYTES) {
    throw new Error(`Source video is too large: ${bytes.byteLength} bytes`);
  }

  return {
    mimeType,
    bytes,
  };
}

function buildVideoUnderstandPrompt(prompt) {
  const userPrompt = String(prompt || "").trim();
  if (userPrompt) {
    return userPrompt;
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

async function waitForGoogleFileReady(ai, file) {
  const deadline = Date.now() + GOOGLE_FILE_WAIT_TIMEOUT_MS;
  let current = file;

  while (Date.now() < deadline) {
    if (current?.state === "ACTIVE" || current?.state === "SUCCEEDED") {
      return current;
    }
    if (current?.state === "FAILED") {
      throw new Error("Google file processing failed");
    }

    await sleep(GOOGLE_POLL_MS);
    current = await ai.files.get({ name: current.name });
  }

  throw new Error("Google file processing timed out");
}

function normalizeRequestedVideoDuration(duration) {
  const parsed = Number(duration);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return 4;
  }
  return parsed <= 4 ? 4 : 8;
}

async function pollGoogleOperation(operationName, apiKey) {
  const deadline = Date.now() + GOOGLE_VIDEO_TIMEOUT_MS;

  while (Date.now() < deadline) {
    const poll = await callGoogleJson(
      `${GOOGLE_API_BASE}/${operationName}`,
      {
        apiKey,
        method: "GET",
        timeoutMs: FETCH_TIMEOUT_MS,
      },
    );

    if (!poll.ok) {
      throw new Error(
        poll.networkError
          ? `Failed to poll Google operation: ${poll.text}`
          : `Failed to poll Google operation: ${poll.status} ${truncate(poll.text)}`,
      );
    }

    if (poll.payload?.error?.message) {
      throw new Error(poll.payload.error.message);
    }

    if (poll.payload?.done) {
      return poll.payload;
    }

    await new Promise((resolve) => setTimeout(resolve, GOOGLE_POLL_MS));
  }

  throw new Error("Google video generation timed out.");
}

async function generateVideoWithSdk({
  apiKey,
  prompt,
  imageUrl,
  model,
  duration,
}) {
  const imageInput = await fetchImageInlineData(imageUrl);
  const ai = new GoogleGenAI({ apiKey });
  const config = {
    numberOfVideos: 1,
    aspectRatio: "16:9",
  };
  const durationSeconds = normalizeRequestedVideoDuration(duration);

  if (durationSeconds) {
    config.durationSeconds = durationSeconds;
  }

  let operation = await ai.models.generateVideos({
    model,
    prompt,
    image: {
      imageBytes: imageInput.data,
      mimeType: imageInput.mimeType,
    },
    config,
  });

  while (!operation.done) {
    await sleep(GOOGLE_POLL_MS);
    operation = await ai.operations.getVideosOperation({ operation });
  }

  if (operation.error) {
    throw new Error(JSON.stringify(operation.error));
  }

  const generatedVideo = operation.response?.generatedVideos?.[0]?.video;
  if (!generatedVideo) {
    throw new Error("Google video generation completed without a downloadable video.");
  }

  const downloadPath = path.join(os.tmpdir(), `veo-${crypto.randomUUID()}.mp4`);
  try {
    await ai.files.download({
      file: generatedVideo,
      downloadPath,
    });

    return {
      videoBytes: await fs.readFile(downloadPath),
      operationName: operation.name ?? null,
    };
  } finally {
    await fs.rm(downloadPath, { force: true });
  }
}

function extractGeneratedVideoUri(operationPayload) {
  return (
    operationPayload?.response?.generateVideoResponse?.generatedSamples?.[0]?.video?.uri ||
    operationPayload?.response?.generatedSamples?.[0]?.video?.uri ||
    operationPayload?.response?.videos?.[0]?.uri ||
    null
  );
}

app.options("*", (req, res) => {
  setCors(res);
  res.status(204).send();
});

app.get("/", (req, res) => {
  json(res, 200, {
    ok: true,
    service: "render-ffmpeg-clip-api",
    routes: [
      "/",
      "/health",
      "/clip",
      "/google/models",
      "/google/text",
      "/google/video-understand",
      "/v1/chat/completions",
      "/google/image",
      "/google/image-binary",
      "/google/video",
      "/google/video-binary",
    ],
    hasGoogleApiKeyEnv: Boolean(process.env.GOOGLE_API_KEY),
    relayTokenRequired: Boolean(RELAY_AUTH_TOKEN),
  });
});

app.get("/health", (req, res) => {
  json(res, 200, {
    ok: true,
    service: "render-ffmpeg-clip-api",
    uptimeSeconds: Math.round(process.uptime()),
  });
});

app.get("/google/models", async (req, res) => {
  if (!ensureRelayAuthorized(req, res)) {
    return;
  }

  const apiKey = resolveGoogleApiKey(req);
  if (!apiKey) {
    return json(res, 400, {
      success: false,
      error: "missing_google_api_key",
      message: "Provide x-goog-api-key or configure GOOGLE_API_KEY",
    });
  }

  const response = await callGoogleJson(
    `${GOOGLE_API_BASE}/models`,
    {
      apiKey,
      method: "GET",
      timeoutMs: FETCH_TIMEOUT_MS,
    },
  );

  if (!response.ok) {
    return json(res, response.networkError === "timeout" ? 504 : 502, {
      success: false,
      error: response.networkError || "google_models_failed",
      message: response.networkError ? response.text : `Google models ${response.status}`,
      debug: {
        upstreamStatus: response.status,
        upstreamBody: truncate(response.text),
      },
    });
  }

  return json(res, 200, response.payload || {});
});

app.post("/google/text", async (req, res) => {
  if (!ensureRelayAuthorized(req, res)) {
    return;
  }

  const apiKey = resolveGoogleApiKey(req);
  if (!apiKey) {
    return json(res, 400, {
      success: false,
      error: "missing_google_api_key",
      message: "Provide x-goog-api-key or configure GOOGLE_API_KEY",
    });
  }

  const model = normalizeTextModel(req.body?.model);
  const payload = buildTextPayload({
    prompt: req.body?.prompt,
    messages: req.body?.messages,
    temperature: req.body?.temperature,
    maxOutputTokens: req.body?.maxOutputTokens || req.body?.max_tokens,
  });

  if (!payload) {
    return json(res, 400, {
      success: false,
      error: "missing_prompt",
      message: "Provide prompt or messages",
    });
  }

  const response = await callGoogleJson(
    `${GOOGLE_API_BASE}/models/${encodeURIComponent(model)}:generateContent`,
    {
      apiKey,
      method: "POST",
      timeoutMs: GOOGLE_TEXT_TIMEOUT_MS,
      body: payload,
    },
  );

  if (!response.ok) {
    return json(res, response.networkError === "timeout" ? 504 : 502, {
      success: false,
      error: response.networkError || "google_text_failed",
      message: response.networkError
        ? response.text
        : `Google text generation failed: ${response.status}`,
      debug: {
        upstreamStatus: response.status,
        upstreamBody: truncate(response.text),
      },
    });
  }

  const text = extractTextParts(response.payload).join("\n").trim();
  return json(res, 200, {
    success: true,
    provider: "google",
    model,
    text,
    result: text,
    raw: response.payload,
  });
});

app.post("/v1/chat/completions", async (req, res) => {
  if (!ensureRelayAuthorized(req, res)) {
    return;
  }

  const apiKey = resolveGoogleApiKey(req);
  if (!apiKey) {
    return json(res, 400, {
      error: {
        message: "Provide x-goog-api-key or configure GOOGLE_API_KEY",
        type: "missing_google_api_key",
      },
    });
  }

  const model = normalizeTextModel(req.body?.model);
  const payload = buildTextPayload({
    messages: req.body?.messages,
    temperature: req.body?.temperature,
    maxOutputTokens: req.body?.max_tokens,
  });

  if (!payload) {
    return json(res, 400, {
      error: {
        message: "Provide messages",
        type: "missing_messages",
      },
    });
  }

  const response = await callGoogleJson(
    `${GOOGLE_API_BASE}/models/${encodeURIComponent(model)}:generateContent`,
    {
      apiKey,
      method: "POST",
      timeoutMs: GOOGLE_TEXT_TIMEOUT_MS,
      body: payload,
    },
  );

  if (!response.ok) {
    return json(res, response.networkError === "timeout" ? 504 : 502, {
      error: {
        message: response.networkError
          ? response.text
          : `Google text generation failed: ${response.status}`,
        type: response.networkError || "google_text_failed",
      },
    });
  }

  const text = extractTextParts(response.payload).join("\n").trim();
  return json(res, 200, {
    id: `chatcmpl-${crypto.randomUUID()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        finish_reason: "stop",
        message: {
          role: "assistant",
          content: text,
        },
      },
    ],
    usage: response.payload?.usageMetadata || undefined,
  });
});

app.post("/google/video-understand", async (req, res) => {
  if (!ensureRelayAuthorized(req, res)) {
    return;
  }

  const apiKey = resolveGoogleApiKey(req);
  if (!apiKey) {
    return json(res, 400, {
      success: false,
      error: "missing_google_api_key",
      message: "Provide x-goog-api-key or configure GOOGLE_API_KEY",
    });
  }

  const videoUrl = String(req.body?.videoUrl || req.body?.url || "").trim();
  const prompt = buildVideoUnderstandPrompt(req.body?.prompt);
  const model = normalizeVideoUnderstandModel(req.body?.model);
  if (!videoUrl) {
    return json(res, 400, {
      success: false,
      error: "missing_video_url",
      message: "Provide videoUrl",
    });
  }

  try {
    const ai = new GoogleGenAI({ apiKey });
    const video = await fetchVideoInlineData(videoUrl);
    const file = await ai.files.upload({
      file: new Blob([video.bytes], { type: video.mimeType }),
      config: {
        mimeType: video.mimeType,
        displayName: `video-${Date.now()}`,
      },
    });
    const readyFile = await waitForGoogleFileReady(ai, file);
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

    const text = extractTextParts(result).join("\n").trim() || String(result.text || "").trim();
    return json(res, 200, {
      success: true,
      ok: true,
      mode: "live",
      source: "gemini_files_video_understand",
      provider: "google",
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
      success: false,
      ok: false,
      mode: "failed",
      source: "gemini_files_video_understand",
      error: "google_video_understand_failed",
      message: error instanceof Error ? error.message : String(error),
      detail: truncate(error?.stack || error),
    });
  }
});

app.post("/google/image", async (req, res) => {
  if (!ensureRelayAuthorized(req, res)) {
    return;
  }

  const apiKey = resolveGoogleApiKey(req);
  if (!apiKey) {
    return json(res, 400, {
      success: false,
      error: "missing_google_api_key",
      message: "Provide x-goog-api-key or configure GOOGLE_API_KEY",
    });
  }

  const prompt = String(req.body?.prompt || "").trim();
  const model = normalizeImageModel(String(req.body?.model || "").trim());
  const aspectRatio = String(req.body?.aspectRatio || "16:9").trim() || "16:9";

  if (!prompt) {
    return json(res, 400, {
      success: false,
      error: "missing_prompt",
      message: "Missing prompt",
    });
  }

  const promptInfo = await maybeRewritePrompt(prompt, apiKey);
  if ("error" in promptInfo) {
    return json(res, 502, {
      success: false,
      error: promptInfo.error,
      message: promptInfo.message,
    });
  }

  const response = await callGoogleJson(
    `${GOOGLE_API_BASE}/models/${encodeURIComponent(model)}:generateContent`,
    {
      apiKey,
      method: "POST",
      timeoutMs: GOOGLE_IMAGE_TIMEOUT_MS,
      body: {
        contents: [{ parts: [{ text: promptInfo.effectivePrompt }] }],
        generationConfig: {
          responseModalities: ["TEXT", "IMAGE"],
          imageConfig: {
            aspectRatio,
          },
        },
      },
    },
  );

  const metadata = {
    provider: "google",
    model,
    originalPrompt: promptInfo.originalPrompt,
    effectivePrompt: promptInfo.effectivePrompt,
    translationApplied: promptInfo.translationApplied,
    translationMode: promptInfo.translationMode,
    relayMode: "render",
  };

  if (!response.ok) {
    return json(res, response.networkError === "timeout" ? 504 : 502, {
      success: false,
      error: response.networkError || "google_image_failed",
      message: response.networkError
        ? response.text
        : `Google image generation failed: ${response.status}`,
      metadata,
      debug: {
        upstreamStatus: response.status,
        upstreamBody: truncate(response.text),
      },
    });
  }

  const inlineData = extractInlineImage(response.payload);
  if (!inlineData?.data) {
    return json(res, 422, {
      success: false,
      error: "no_inline_image",
      message: "Google image generation returned no image data",
      metadata,
      debug: {
        upstreamBody: truncate(response.text),
        upstreamTextParts: extractTextParts(response.payload).slice(0, 3),
      },
    });
  }

  return json(res, 200, {
    success: true,
    provider: "google",
    model,
    mimeType: inlineData.mimeType,
    data: inlineData.data,
    metadata,
  });
});

app.post("/google/image-binary", async (req, res) => {
  if (!ensureRelayAuthorized(req, res)) {
    return;
  }

  const apiKey = resolveGoogleApiKey(req);
  if (!apiKey) {
    return json(res, 400, {
      success: false,
      error: "missing_google_api_key",
      message: "Provide x-goog-api-key or configure GOOGLE_API_KEY",
    });
  }

  const prompt = String(req.body?.prompt || "").trim();
  const model = normalizeImageModel(String(req.body?.model || "").trim());
  const aspectRatio = String(req.body?.aspectRatio || "16:9").trim() || "16:9";

  if (!prompt) {
    return json(res, 400, {
      success: false,
      error: "missing_prompt",
      message: "Missing prompt",
    });
  }

  const promptInfo = await maybeRewritePrompt(prompt, apiKey);
  if ("error" in promptInfo) {
    return json(res, 502, {
      success: false,
      error: promptInfo.error,
      message: promptInfo.message,
    });
  }

  const response = await callGoogleJson(
    `${GOOGLE_API_BASE}/models/${encodeURIComponent(model)}:generateContent`,
    {
      apiKey,
      method: "POST",
      timeoutMs: GOOGLE_IMAGE_TIMEOUT_MS,
      body: {
        contents: [{ parts: [{ text: promptInfo.effectivePrompt }] }],
        generationConfig: {
          responseModalities: ["TEXT", "IMAGE"],
          imageConfig: {
            aspectRatio,
          },
        },
      },
    },
  );

  const metadata = {
    provider: "google",
    model,
    originalPrompt: promptInfo.originalPrompt,
    effectivePrompt: promptInfo.effectivePrompt,
    translationApplied: promptInfo.translationApplied,
    translationMode: promptInfo.translationMode,
    relayMode: "render",
  };

  if (!response.ok) {
    return json(res, response.networkError === "timeout" ? 504 : 502, {
      success: false,
      error: response.networkError || "google_image_failed",
      message: response.networkError
        ? response.text
        : `Google image generation failed: ${response.status}`,
      metadata,
      debug: {
        upstreamStatus: response.status,
        upstreamBody: truncate(response.text),
      },
    });
  }

  const inlineData = extractInlineImage(response.payload);
  if (!inlineData?.data) {
    return json(res, 422, {
      success: false,
      error: "no_inline_image",
      message: "Google image generation returned no image data",
      metadata,
      debug: {
        upstreamBody: truncate(response.text),
        upstreamTextParts: extractTextParts(response.payload).slice(0, 3),
      },
    });
  }

  const bytes = Buffer.from(inlineData.data, "base64");
  setCors(res);
  res.setHeader("Content-Type", inlineData.mimeType || "image/png");
  res.setHeader("Cache-Control", "no-store");
  res.setHeader("X-Proxy-Metadata", encodeMetadataHeader(metadata));
  return res.status(200).send(bytes);
});

app.post("/google/video", async (req, res) => {
  if (!ensureRelayAuthorized(req, res)) {
    return;
  }

  const apiKey = resolveGoogleApiKey(req);
  if (!apiKey) {
    return json(res, 400, {
      success: false,
      error: "missing_google_api_key",
      message: "Provide x-goog-api-key or configure GOOGLE_API_KEY",
    });
  }

  const prompt = String(req.body?.prompt || "").trim();
  const imageUrl = String(req.body?.imageUrl || "").trim();
  const model = normalizeVideoModel(String(req.body?.model || "").trim());
  const duration = req.body?.duration ?? null;

  if (!prompt) {
    return json(res, 400, {
      success: false,
      error: "missing_prompt",
      message: "Missing prompt",
    });
  }

  if (!imageUrl) {
    return json(res, 400, {
      success: false,
      error: "missing_image_url",
      message: "Missing imageUrl",
    });
  }

  try {
    const { videoBytes, operationName } = await generateVideoWithSdk({
      apiKey,
      prompt,
      imageUrl,
      model,
      duration,
    });

    return json(res, 200, {
      success: true,
      provider: "google",
      model,
      mimeType: "video/mp4",
      data: Buffer.from(videoBytes).toString("base64"),
      metadata: {
        provider: "google",
        model,
        relayMode: "render",
        operationName,
        requestedDuration: duration,
      },
    });
  } catch (error) {
    return json(res, 502, {
      success: false,
      error: "google_video_failed",
      message: error instanceof Error ? error.message : String(error),
    });
  }
});

app.post("/google/video-binary", async (req, res) => {
  if (!ensureRelayAuthorized(req, res)) {
    return;
  }

  const apiKey = resolveGoogleApiKey(req);
  if (!apiKey) {
    return json(res, 400, {
      success: false,
      error: "missing_google_api_key",
      message: "Provide x-goog-api-key or configure GOOGLE_API_KEY",
    });
  }

  const prompt = String(req.body?.prompt || "").trim();
  const imageUrl = String(req.body?.imageUrl || "").trim();
  const model = normalizeVideoModel(String(req.body?.model || "").trim());
  const duration = req.body?.duration ?? null;

  if (!prompt) {
    return json(res, 400, {
      success: false,
      error: "missing_prompt",
      message: "Missing prompt",
    });
  }

  if (!imageUrl) {
    return json(res, 400, {
      success: false,
      error: "missing_image_url",
      message: "Missing imageUrl",
    });
  }

  try {
    const { videoBytes, operationName } = await generateVideoWithSdk({
      apiKey,
      prompt,
      imageUrl,
      model,
      duration,
    });
    const metadata = {
      provider: "google",
      model,
      relayMode: "render",
      operationName,
      requestedDuration: duration,
    };

    setCors(res);
    res.setHeader("Content-Type", "video/mp4");
    res.setHeader("Cache-Control", "no-store");
    res.setHeader("X-Proxy-Metadata", encodeMetadataHeader(metadata));
    return res.status(200).send(videoBytes);
  } catch (error) {
    return json(res, 502, {
      success: false,
      error: "google_video_failed",
      message: error instanceof Error ? error.message : String(error),
    });
  }
});

app.all("/clip", async (req, res) => {
  setCors(res);
  if (!["GET", "POST"].includes(req.method)) {
    return res.status(405).json({ error: "Method Not Allowed" });
  }

  const params = parseClipParams(req);
  const invalid = validateClipParams(params);
  if (invalid) {
    return res.status(400).json({ error: invalid });
  }

  const id = crypto.randomUUID();
  const workDir = path.join(os.tmpdir(), `clip-${id}`);
  const inputFile = path.join(workDir, "input.mp4");
  const outputFile = path.join(workDir, "output.mp4");

  try {
    await fs.mkdir(workDir, { recursive: true });
    await downloadInput(params.videoUrl, inputFile);
    const mode = await clipVideo({
      inputFile,
      outputFile,
      start: params.start,
      end: params.end,
    });
    const output = await fs.readFile(outputFile);
    res.setHeader("Content-Type", "video/mp4");
    res.setHeader("Content-Disposition", 'inline; filename="clip.mp4"');
    res.setHeader("X-Clip-Mode", mode);
    return res.status(200).send(output);
  } catch (error) {
    return res.status(500).json({ error: `clip failed: ${String(error.message || error)}` });
  } finally {
    await fs.rm(workDir, { recursive: true, force: true });
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`render relay api listening on :${PORT}`);
});

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
