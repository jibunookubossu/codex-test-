# Mount Fuji history video generator

`generate_fuji_videos.py` sends four historical Mount Fuji prompts to a
text-to-video model through the Hugging Face Inference API and saves the
responses as MP4 files. It uses only the Python standard library and writes
downloads atomically. Responses are checked for an MP4 file signature rather
than trusting the server's `Content-Type` header.

## Usage

Create a Hugging Face access token and keep it in the environment:

```bash
export HF_TOKEN='hf_...'
python3 generate_fuji_videos.py --output-dir output
```

Inspect all filenames and prompts without making API calls:

```bash
python3 generate_fuji_videos.py --dry-run
```

The default model is `ali-vilab/text-to-video-ms-1.7b`. It can be overridden:

```bash
python3 generate_fuji_videos.py --model organization/model-name
```

## Service limitations

Hugging Face may provide small complimentary inference credits, but hosted
inference is an external service: it requires an account token, model/provider
availability varies, and free usage is not guaranteed. Resolution and clip
length are decided by the selected model; writing “8k” in a prompt does not make
an endpoint return native 8K video. This repository does not contain model
weights, and local generation requires separately downloading a video model and
having adequate GPU memory.
