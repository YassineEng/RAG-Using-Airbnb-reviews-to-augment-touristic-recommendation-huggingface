from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="qwen/qwen3-next-80b-a3b-thinking",
  messages=[{"role":"user","content":""}],
  temperature=0.6,
  top_p=0.7,
  max_tokens=4096,
  stream=True
)
for chunk in completion:
  reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
  if reasoning:
    print(reasoning, end="")
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

