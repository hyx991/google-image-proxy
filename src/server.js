import express from "express";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { spawn } from "node:child_process";

const app = express();
app.use(express.json({ limit: "1mb" }));

const PORT = Number(process.env.PORT || 10000);
const MAX_INPUT_BYTES = Number(process.env.MAX_INPUT_BYTES || 120 * 1024 * 1024);
const MAX_CLIP_SECONDS = Number(process.env.MAX_CLIP_SECONDS || 600);
const FETCH_TIMEOUT_MS = Number(process.env.FETCH_TIMEOUT_MS || 120000);

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type,Authorization");
}

function parseClipParams(req) {
  const src = req.method === "POST" ? req.body || {} : req.query || {};
  const videoUrl = String(src.video_url || src.videoUrl || "").trim();
  const start = Number(src.start ?? 0);
  const end = Number(src.end);
  return { videoUrl, start, end };
}

function validateParams({ videoUrl, start, end }) {
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
      outputFile
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
      outputFile
    ]);
    return "reencode";
  }
}

app.options("*", (req, res) => {
  setCors(res);
  res.status(204).send();
});

app.get("/", (req, res) => {
  setCors(res);
  res.json({
    ok: true,
    service: "render-ffmpeg-clip-api",
    usage: "/clip?video_url=...&start=0&end=5"
  });
});

app.all("/clip", async (req, res) => {
  setCors(res);
  if (!["GET", "POST"].includes(req.method)) {
    return res.status(405).json({ error: "Method Not Allowed" });
  }

  const params = parseClipParams(req);
  const invalid = validateParams(params);
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
      end: params.end
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
  console.log(`ffmpeg clip api listening on :${PORT}`);
});
