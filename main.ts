const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function json(payload: unknown, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      ...corsHeaders,
      "Content-Type": "application/json; charset=utf-8",
    },
  });
}

Deno.serve(async (req: Request) => {
  const url = new URL(req.url);

  if (req.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: corsHeaders,
    });
  }

  if (req.method === "GET") {
    return json({
      success: true,
      service: "google-image-proxy",
      paths: ["/", "/proxy", "/generate-image"],
      hasApiKey: Boolean(Deno.env.get("GOOGLE_API_KEY")),
    });
  }

  if (!["/", "/proxy", "/generate-image"].includes(url.pathname)) {
    return json({ success: false, error: "Not found" }, 404);
  }

  if (req.method !== "POST") {
    return json({ success: false, error: "Method not allowed" }, 405);
  }

  const apiKey = Deno.env.get("GOOGLE_API_KEY");
  if (!apiKey) {
    return json({ success: false, error: "GOOGLE_API_KEY is not configured" }, 500);
  }

  try {
    const body = await req.json();
    const prompt = String(body?.prompt ?? "").trim();
    const aspectRatio = String(body?.aspectRatio ?? "9:16").trim() || "9:16";

    if (!prompt) {
      return json({ success: false, error: "Missing prompt" }, 400);
    }

    const upstream = await fetch(
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": apiKey,
        },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: {
            responseModalities: ["TEXT", "IMAGE"],
            imageConfig: { aspectRatio },
          },
        }),
      },
    );

    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: {
        ...corsHeaders,
        "Content-Type": "application/json; charset=utf-8",
      },
    });
  } catch (error) {
    return json(
      {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      },
      500,
    );
  }
});
