# gemini-video-understand

Minimal Render-friendly service for Gemini video understanding.

## Routes

- `GET /health`
- `POST /video-understand`

## Request

```json
{
  "videoUrl": "https://example.com/video.mp4",
  "model": "gemini-3.1-flash-lite",
  "prompt": "请分析这个视频"
}
```

Pass Google AI Studio key either as:

```text
x-goog-api-key: <GOOGLE_API_KEY>
```

or configure Render environment variable:

```text
GOOGLE_API_KEY=<GOOGLE_API_KEY>
GEMINI_MODEL=gemini-3.1-flash-lite
MAX_VIDEO_BYTES=83886080
```

Optional protection:

```text
RELAY_AUTH_TOKEN=<your relay token>
```

