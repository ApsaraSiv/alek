FROM osrf/ros:humble-desktop AS base
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# 0. HTTPS for apt
RUN sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g' /etc/apt/sources.list

# 1. Tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-vcstool python3-colcon-common-extensions python3-rosdep \
    python3-pip git wget curl lsb-release gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# 2. Gazebo Harmonic
RUN wget https://packages.osrfoundation.org/gazebo.gpg \
        -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
        http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update && apt-get install -y --no-install-recommends gz-harmonic \
    && rm -rf /var/lib/apt/lists/*

# 3. ROS 2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-gzharmonic \
    ros-humble-moveit \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-nav2-bringup \
    ros-humble-navigation2 \
    ros-humble-slam-toolbox \
    ros-humble-joint-state-publisher \
    ros-humble-xacro \
    ros-humble-robot-state-publisher \
    ros-humble-rviz2 \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-controller-manager \
    ros-humble-joint-trajectory-controller \
    ros-humble-diff-drive-controller \
    ros-humble-position-controllers \
    ros-humble-velocity-controllers \
    ros-humble-effort-controllers \
    ros-humble-realsense2-description \
    && rm -rf /var/lib/apt/lists/*

# 3.5. OCR (erc_perception's shelf_number_detector reads the overhead
# shelf-column marker digits via tesseract)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir pytesseract

# 4. Workspace directory (source is mounted from host at runtime)
ENV GZ_VERSION=harmonic
RUN mkdir -p /opt/erc_ws/src

# 5. Entrypoint
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
WORKDIR /opt/erc_ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
