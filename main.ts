const API_BASE = "https://generativelanguage.googleapis.com/v1beta/models";
const GOOGLE_API_KEY = Deno.env.get("GOOGLE_API_KEY") || "";
const IMAGE_MODEL =
  Deno.env.get("IMAGE_MODEL") || "gemini-3.1-flash-image-preview";
const TRANSLATION_MODEL =
  Deno.env.get("TRANSLATION_MODEL") || "gemini-2.5-flash";
const DEFAULT_TRANSLATION_TIMEOUT_MS = 20000;
const DEFAULT_IMAGE_TIMEOUT_MS = 120000;
const MIN_IMAGE_TIMEOUT_MS = 120000;

function readTimeoutMs(
  name: string,
  fallback: number,
  minimum = 0,
) {
  const raw = Deno.env.get(name);
  const parsed = raw ? Number(raw) : fallback;
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback;
  }
  return Math.max(parsed, minimum);
}

const TRANSLATION_TIMEOUT_MS = readTimeoutMs(
  "TRANSLATION_TIMEOUT_MS",
  DEFAULT_TRANSLATION_TIMEOUT_MS,
);
const IMAGE_TIMEOUT_MS = readTimeoutMs(
  "IMAGE_TIMEOUT_MS",
  DEFAULT_IMAGE_TIMEOUT_MS,
  MIN_IMAGE_TIMEOUT_MS,
);

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

const CJK_RE =
  /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\u31f0-\u31ff\uac00-\ud7af]/u;

function jsonResponse(payload: unknown, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...CORS_HEADERS,
    },
  });
}

function truncate(value: string, max = 1200) {
  if (value.length <= max) return value;
  return `${value.slice(0, max)}...<truncated>`;
}

function extractTextParts(body: any): string[] {
  const candidates = Array.isArray(body?.candidates) ? body.candidates : [];
  const texts: string[] = [];
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

function extractInlineImage(body: any) {
  const candidates = Array.isArray(body?.candidates) ? body.candidates : [];
  for (const candidate of candidates) {
    const parts = Array.isArray(candidate?.content?.parts)
      ? candidate.content.parts
      : [];
    for (const part of parts) {
      if (part?.inlineData?.data) {
        return part.inlineData;
      }
    }
  }
  return null;
}

async function callGoogleModel(model: string, payload: unknown, timeoutMs: number) {
  const url =
    `${API_BASE}/${encodeURIComponent(model)}:generateContent?key=${
      encodeURIComponent(GOOGLE_API_KEY)
    }`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort("timeout"), timeoutMs);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    const text = await response.text();
    let json = null;
    let parseError = null;

    if (text) {
      try {
        json = JSON.parse(text);
      } catch (error) {
        parseError = error instanceof Error ? error.message : String(error);
      }
    }

    return { ok: response.ok, status: response.status, url, text, json, parseError };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const isTimeout = message === "timeout" || message.includes("aborted");
    return {
      ok: false,
      status: 0,
      url,
      text: "",
      json: null,
      networkError: isTimeout ? "timeout" : "network_error",
      networkMessage: message,
    };
  } finally {
    clearTimeout(timer);
  }
}

async function rewritePromptIfNeeded(prompt: string) {
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

  const translation = await callGoogleModel(
    TRANSLATION_MODEL,
    translationPayload,
    TRANSLATION_TIMEOUT_MS,
  );

  if (translation.networkError) {
    return {
      error: `translation_${translation.networkError}`,
      message: `Prompt rewrite failed: ${translation.networkMessage}`,
      debug: {
        upstream_url: translation.url,
        upstream_status: translation.status,
      },
    };
  }

  if (translation.parseError) {
    return {
      error: "translation_parse_error",
      message: `Prompt rewrite response was not valid JSON: ${translation.parseError}`,
      debug: {
        upstream_url: translation.url,
        upstream_status: translation.status,
        upstream_body: truncate(translation.text),
      },
    };
  }

  if (!translation.ok) {
    return {
      error: "translation_http_error",
      message: `Prompt rewrite upstream ${translation.status}`,
      debug: {
        upstream_url: translation.url,
        upstream_status: translation.status,
        upstream_body: truncate(translation.text),
      },
    };
  }

  const translatedText = extractTextParts(translation.json).join("\n").trim();
  if (!translatedText) {
    return {
      error: "translation_empty",
      message: "Prompt rewrite upstream returned no text",
      debug: {
        upstream_url: translation.url,
        upstream_status: translation.status,
        upstream_body: truncate(translation.text),
      },
    };
  }

  return {
    originalPrompt: prompt,
    effectivePrompt: translatedText,
    translationApplied: true,
    translationMode: "proxy-rewrite",
    translationDebug: {
      upstream_url: translation.url,
      upstream_status: translation.status,
      model: TRANSLATION_MODEL,
    },
  };
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: CORS_HEADERS });
  }

  if (!GOOGLE_API_KEY) {
    return jsonResponse(
      {
        success: false,
        error: "missing_google_api_key",
        message: "GOOGLE_API_KEY is required",
      },
      500,
    );
  }

  const url = new URL(req.url);
  if (req.method === "GET" && url.pathname === "/") {
    return jsonResponse({
      success: true,
      service: "google-image-proxy",
      paths: ["/", "/proxy", "/generate-image"],
      hasApiKey: true,
      imageModel: IMAGE_MODEL,
      translationModel: TRANSLATION_MODEL,
      translationTimeoutMs: TRANSLATION_TIMEOUT_MS,
      imageTimeoutMs: IMAGE_TIMEOUT_MS,
      minImageTimeoutMs: MIN_IMAGE_TIMEOUT_MS,
    });
  }

  if (req.method !== "POST" || !["/", "/proxy", "/generate-image"].includes(url.pathname)) {
    return jsonResponse({ success: false, error: "not_found", message: "Not found" }, 404);
  }

  let body: any;
  let rawBody = "";
  try {
    rawBody = await req.text();
    body = JSON.parse(rawBody);
  } catch (error) {
    return jsonResponse(
      {
        success: false,
        error: "invalid_json",
        message: error instanceof Error ? error.message : String(error),
        debug: {
          request_body: truncate(rawBody),
        },
      },
      400,
    );
  }

  const prompt = String(body?.prompt || "").trim();
  const aspectRatio = String(body?.aspectRatio || "9:16").trim() || "9:16";

  if (!prompt) {
    return jsonResponse(
      { success: false, error: "missing_prompt", message: "Missing prompt" },
      400,
    );
  }

  const promptInfo = await rewritePromptIfNeeded(prompt);
  if ("error" in promptInfo) {
    return jsonResponse(
      {
        success: false,
        error: promptInfo.error,
        message: promptInfo.message,
        metadata: {
          originalPrompt: prompt,
          effectivePrompt: prompt,
          translationApplied: true,
          translationMode: "proxy-rewrite",
          provider: "google",
          model: IMAGE_MODEL,
        },
        debug: promptInfo.debug,
      },
      502,
    );
  }

  const imagePayload = {
    contents: [{ parts: [{ text: promptInfo.effectivePrompt }] }],
    generationConfig: {
      responseModalities: ["TEXT", "IMAGE"],
      imageConfig: {
        aspectRatio,
      },
    },
  };

  const imageResult = await callGoogleModel(IMAGE_MODEL, imagePayload, IMAGE_TIMEOUT_MS);
  const metadata = {
    originalPrompt: promptInfo.originalPrompt,
    effectivePrompt: promptInfo.effectivePrompt,
    translationApplied: promptInfo.translationApplied,
    translationMode: promptInfo.translationMode,
    provider: "google",
    model: IMAGE_MODEL,
  };

  if (imageResult.networkError) {
    return jsonResponse(
      {
        success: false,
        error: imageResult.networkError,
        message: imageResult.networkError === "timeout"
          ? `Google image call timed out after ${IMAGE_TIMEOUT_MS}ms`
          : `Google image call failed: ${imageResult.networkMessage}`,
        metadata,
        debug: {
          upstream_url: imageResult.url,
          upstream_status: imageResult.status,
        },
      },
      imageResult.networkError === "timeout" ? 504 : 502,
    );
  }

  if (imageResult.parseError) {
    return jsonResponse(
      {
        success: false,
        error: "image_parse_error",
        message: `Image upstream returned invalid JSON: ${imageResult.parseError}`,
        metadata,
        debug: {
          upstream_url: imageResult.url,
          upstream_status: imageResult.status,
          upstream_body: truncate(imageResult.text),
        },
      },
      502,
    );
  }

  if (!imageResult.ok) {
    return jsonResponse(
      {
        success: false,
        error: "image_http_error",
        message: `Image upstream ${imageResult.status}`,
        metadata,
        debug: {
          upstream_url: imageResult.url,
          upstream_status: imageResult.status,
          upstream_body: truncate(imageResult.text),
        },
      },
      502,
    );
  }

  const inlineData = extractInlineImage(imageResult.json);
  if (!inlineData?.data) {
    const textParts = extractTextParts(imageResult.json);
    return jsonResponse(
      {
        success: false,
        error: "no_inline_image",
        message: textParts.length
          ? "Upstream returned text only, not image data"
          : "Upstream returned no inline image data",
        metadata,
        debug: {
          upstream_url: imageResult.url,
          upstream_status: imageResult.status,
          upstream_body: truncate(imageResult.text),
          upstream_text_parts: textParts.slice(0, 3),
        },
      },
      422,
    );
  }

  return jsonResponse({
    ...imageResult.json,
    proxyMetadata: metadata,
  });
});
