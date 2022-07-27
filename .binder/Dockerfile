FROM zivgitlab.wwu.io/pymor/pymor/binder:2022-1-0

# binder wants to set the NB_ vars anyways, so we use it to service both setups
ARG NB_USER
ARG NB_UID

USER root
COPY .binder/entrypoint.bash /usr/local/bin/
RUN rm -rf /presentation \
    && useradd -d /presentation --shell /bin/bash -u ${NB_UID} -o -c "" -m ${NB_USER}

ENV USER=${NB_USER} \
    HOME=/presentation

COPY requirements-dev.txt /tmp
RUN chown -R ${NB_USER} /presentation && python -m pip install --no-cache-dir -r /tmp/requirements-dev.txt
USER ${NB_USER}

COPY . /presentation/

ENTRYPOINT ["/usr/local/bin/entrypoint.bash"]
WORKDIR /presentation
