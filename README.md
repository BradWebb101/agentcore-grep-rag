# grepRag

Answers questions from GitHub READMEs by copying them onto a session filesystem and grepping them. There is no vector index.

The corpus is not in git. Download it from [markdown-dataset](https://github.com/davidmyersdev/markdown-dataset) before deploy. CDK uploads that directory to S3. At question time the agent checks its session disk, uses `aws s3 ls` to find object names that match the question, copies those objects only, and greps them.

## Approach

A RAG stack embeds chunks and hopes the nearest neighbours contain the line you need. This stack keeps the source as files.

```
Question → list session disk → file present → grep
                              → file missing → aws s3 ls by name → aws s3 cp those objects → grep
         → answer with path and line number
```

Session storage is an AgentCore mount at `/mnt/workspace`, one tree per runtime session id. `aws s3 cp`, `find`, and `grep` see a normal directory. The same session id keeps the files across later invokes. A runtime version update starts that session with an empty disk again. The mount holds up to 1 GB and expires after 14 idle days.

The agent decides. It is not told whether the files are already local. Listing is how it looks before it copies. Grep is the authority for the line. S3 is the authority for the corpus. A file on the session disk can be older than the object in the bucket.

Kimi K3 is called with Bedrock `InvokeModel` (`global.moonshotai.kimi-k3` outside the US and Canada, `us.moonshotai.kimi-k3` there). The container is `linux/arm64` and includes the AWS CLI. The runtime role can `s3:GetObject` and `s3:ListBucket` on the corpus bucket. `s3_cp` refuses the bucket root and the `corpus/` prefix, so a copy is one object key returned by `s3_ls`.

## Download the corpus

`corpus/` is gitignored. Clone does not create it. Deploy publishes whatever is in that directory, so download first.

The upstream file is [`data/markdown.json`](https://github.com/davidmyersdev/markdown-dataset/blob/main/data/markdown.json): READMEs of popular MIT-licensed GitHub repositories, stored as base64 on each record (`markdownEncoded`, `repoName`). The raw URL is:

`https://raw.githubusercontent.com/davidmyersdev/markdown-dataset/main/data/markdown.json`

This writes one markdown file per repository, `corpus/<owner>/<repo>.md`. Duplicate records in the JSON are the same bytes, so each repository is written once (917 files, about 16 MB).

```bash
curl -fsSL -o /tmp/markdown.json \
  https://raw.githubusercontent.com/davidmyersdev/markdown-dataset/main/data/markdown.json

python3 - <<'PY'
import base64, json, shutil
from pathlib import Path

records = json.load(open("/tmp/markdown.json"))
root = Path("corpus")
if root.exists():
    shutil.rmtree(root)
seen = set()
for record in records:
    key = (record["repoName"], record["markdownSource"])
    if key in seen:
        continue
    seen.add(key)
    owner, repo = record["repoName"].split("/", 1)
    dest = root / owner / f"{repo}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(record["markdownEncoded"]))
print(f"wrote {len(seen)} files under {root}")
PY
```

## Deploy

Enable Kimi K3 in the account, with `corpus/` already populated.

```bash
npx cdk deploy
```

The bucket deployment syncs `corpus/` to `s3://<bucket>/corpus/`. Note `CorpusBucketName`, `AgentRuntimeArn`, and `RuntimeLogGroupName`.

## Ask

The CLI writes the response body to the outfile. `-` prints it on the terminal. The session id must be at least 33 characters. A second question with the same id should list the session disk and grep, without copying again.

```bash
aws bedrock-agentcore invoke-agent-runtime \
  --agent-runtime-arn "$AGENT_RUNTIME_ARN" \
  --runtime-session-id "grep-rag-session-0000000000000002" \
  --qualifier DEFAULT \
  --content-type application/json \
  --accept application/json \
  --cli-binary-format raw-in-base64-out \
  --payload '{"prompt":"For imgaug, which Python versions does it support, and how do I install the latest code straight from GitHub?"}' \
  -
```

The response is `{"answer": "..."}`.

## Logs

Application and usage logs go to `/aws/vendedlogs/bedrock-agentcore/grep_rag` (`RuntimeLogGroupName`). Each step is one JSON line: `list_session`, `s3_cp`, `grep`, `command.start`, `command.end`, `request.completed`.

```bash
aws logs tail "$RUNTIME_LOG_GROUP_NAME" --follow
```
