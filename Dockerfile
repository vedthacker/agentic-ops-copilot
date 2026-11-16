FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir -e .
ENV OPS_STATE_DIR=/data/state
# mount a volume at /data/state so checkpoints (and paused approvals) survive container restarts
ENTRYPOINT ["python", "-m", "opscopilot.cli"]
CMD ["eval"]
