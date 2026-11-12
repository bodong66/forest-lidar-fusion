# GPU training / inference image. Build: docker build -t forestfusion .
# Run:   docker run --gpus all -v $PWD/data:/app/data -v $PWD/runs:/app/runs forestfusion \
#          python -m forestfusion.train_seg --config configs/seg_forinstance.yaml
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[edge]"
COPY configs ./configs
COPY scripts ./scripts

CMD ["python", "-m", "forestfusion.train_seg", "--help"]
