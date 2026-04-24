#!/bin/bash
set -euo pipefail

# vLLM Qwen3-ASR Smoke Test
# Validates model loads and serves transcription correctly via vLLM.
# Reference: https://huggingface.co/Qwen/Qwen3-ASR-1.7B#deployment-with-vllm
#
# Usage: vllm_asr_smoke_test.sh <model_dir> <model_name> [extra_args...]

MODEL_DIR="${1:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
MODEL_NAME="${2:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
shift 2
EXTRA_ARGS="$*"

VLLM_PORT=8000
HEALTH_TIMEOUT=600
HEALTH_INTERVAL=10

# Public test audio from Qwen3-ASR repo
TEST_AUDIO_URL="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"

echo "=== Qwen3-ASR Smoke Test: ${MODEL_NAME} ==="
echo "=== Model directory: ${MODEL_DIR} ==="
ls -la "${MODEL_DIR}"

echo "=== Starting vLLM server ==="
# shellcheck disable=SC2086
vllm serve "${MODEL_DIR}" \
  --port "${VLLM_PORT}" \
  ${EXTRA_ARGS} &
VLLM_PID=$!

cleanup() {
  echo "=== Stopping vLLM server ==="
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Waiting for health check ==="
elapsed=0
while [ "${elapsed}" -lt "${HEALTH_TIMEOUT}" ]; do
  if curl -sf http://localhost:${VLLM_PORT}/health >/dev/null 2>&1; then
    echo "Server healthy after ${elapsed}s"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "ERROR: vLLM process died"
    exit 1
  fi
  sleep "${HEALTH_INTERVAL}"
  elapsed=$((elapsed + HEALTH_INTERVAL))
done

if [ "${elapsed}" -ge "${HEALTH_TIMEOUT}" ]; then
  echo "ERROR: Health check timed out after ${HEALTH_TIMEOUT}s"
  exit 1
fi

# Test 1: Chat completions with audio_url (primary vLLM recipe)
echo "=== Test 1: /v1/chat/completions with audio_url ==="
RESPONSE=$(curl -sf http://localhost:${VLLM_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_DIR}\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [{
        \"type\": \"audio_url\",
        \"audio_url\": {\"url\": \"${TEST_AUDIO_URL}\"}
      }]
    }]
  }")

echo "Response: ${RESPONSE}"

python3 -c "
import json, sys
r = json.loads('''${RESPONSE}''')
content = r['choices'][0]['message']['content']
if not content or len(content.strip()) == 0:
    print('FAIL: empty response from chat completions')
    sys.exit(1)
print(f'ASR output: {content}')
print('PASS: chat completions with audio works')
"

# Test 2: OpenAI transcription API
echo "=== Test 2: /v1/audio/transcriptions ==="
pip install -q httpx > /dev/null 2>&1
python3 -c "
import httpx, json, sys

audio_url = '${TEST_AUDIO_URL}'
audio_data = httpx.get(audio_url).content

resp = httpx.post(
    'http://localhost:${VLLM_PORT}/v1/audio/transcriptions',
    files={'file': ('test.wav', audio_data, 'audio/wav')},
    data={'model': '${MODEL_DIR}'},
    timeout=120,
)
r = resp.json()
print(f'Transcription response: {json.dumps(r, indent=2)}')
if 'text' not in r or not r['text'].strip():
    print('FAIL: empty transcription')
    sys.exit(1)
print(f'Transcription: {r[\"text\"]}')
print('PASS: transcription endpoint works')
"

echo "=== PASSED: ${MODEL_NAME} smoke test ==="
