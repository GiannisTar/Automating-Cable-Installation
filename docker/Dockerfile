
# --- Dockerfile: Ubuntu 22.04 + Python 3.11 + ROS2 Humble + MuJoCo (headless, CPU-only) ---
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-distutils python3.11-dev \
    build-essential git wget unzip ca-certificates \
    python3-pip python3-venv \
    curl gnupg2 lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Set python3 to python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN python3 --version

# Install ROS2 Humble and colcon extensions
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list && \
    apt-get update && apt-get install -y ros-humble-ros-base ros-humble-xacro ros-humble-moveit python3-colcon-common-extensions && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies (requirements.txt must exist in context)
COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# MuJoCo install (latest free version)
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz && \
    mkdir -p /root/.mujoco && \
    tar -xzf /tmp/mujoco.tar.gz -C /root/.mujoco && \
    rm /tmp/mujoco.tar.gz
ENV MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=$MUJOCO_PATH/bin:${LD_LIBRARY_PATH}


# Copy scripts, models, and scenes into image
COPY src/control /workspace/scripts
COPY src/cable /workspace/scripts
COPY models/ur5e /workspace/models/ur5e
COPY models/table /workspace/models/table
COPY models/scenes /workspace/scenes
COPY models/cable_scene.xml /workspace/models/cable_scene.xml

# Ensure output directory exists
RUN mkdir -p /workspace/output

# Final setup
WORKDIR /workspace
ENV PYTHONPATH=/workspace/scripts:$PYTHONPATH

# Health check script
RUN echo '#!/bin/bash\n[ -f /workspace/models/ur5e/ur5e.xml ] && echo "MJCF exists" || echo "MJCF missing"' > /usr/local/bin/check_models && chmod +x /usr/local/bin/check_models

# Entrypoint
CMD ["/bin/bash"]