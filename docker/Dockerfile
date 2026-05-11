FROM docker.io/pavelcz/yrc-bench-procgen-base:py38-cu121

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PROCGEN_CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/cmake/Qt5
ENV MAKEFLAGS=-j4

COPY requirements.txt /opt/src/yrc-bench/requirements.txt
RUN python -m pip install -r /opt/src/yrc-bench/requirements.txt || exit 1

COPY lib/acs /opt/src/yrc-bench/lib/acs
RUN python -m pip install /opt/src/yrc-bench/lib/acs || exit 1

COPY lib/procgen /opt/src/yrc-bench/lib/procgen
RUN python -m pip install /opt/src/yrc-bench/lib/procgen || exit 1

# pyod has internal `from pyod...` imports, so it must be installed as a
# top-level site-packages module rather than just being on PYTHONPATH.
COPY lib/pyod /opt/src/yrc-bench/lib/pyod
RUN python -m pip install /opt/src/yrc-bench/lib/pyod || exit 1

RUN rm -rf /opt/src/yrc-bench

WORKDIR /workspace
CMD ["bash"]
