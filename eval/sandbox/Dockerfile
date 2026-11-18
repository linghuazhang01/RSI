# Code-execution sandbox for EvalPlus (HumanEval+/MBPP+) and LiveCodeBench grading.
# Build context = eval/  (so third_party/LiveCodeBench at its pinned commit can be copied in):
#   docker build -t rsi-eval-sandbox:0.1 -f eval/sandbox/Dockerfile eval
# The container is always run with --network none, read-only rootfs, dropped capabilities,
# non-root user, memory/pid limits (see evallib/graders/code.py::docker_cmd). Network is only
# used at *build* time to install pinned packages.
FROM python:3.11.10-slim-bookworm

RUN pip install --no-cache-dir \
      evalplus==0.3.1 \
      numpy==1.26.4 \
      tqdm==4.67.1 \
      pebble==5.1.0

# LiveCodeBench evaluator only (lcb_runner.evaluation needs numpy/tqdm); repo pinned at
# 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24 in eval/third_party/LiveCodeBench.
COPY third_party/LiveCodeBench/lcb_runner /opt/lcb/lcb_runner
COPY sandbox/inside_grade.py /opt/grader/inside_grade.py

ENV PYTHONPATH=/opt/lcb \
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1

USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python", "/opt/grader/inside_grade.py"]
