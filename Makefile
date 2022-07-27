NB_USER:=${USER}


presentation: pymor.ipynb qr_school_website.png qr_docs.png qr_self.png docker_image
	PYMOR_JUPYTER_TOKEN=mathmod $(DOCKER_COMPOSE) up -d
	firefox http://127.0.0.1:8888/tree/pymor.md?token=mathmod
	$(DOCKER_COMPOSE) down

pymor.ipynb:

pdf:
	$(DOCKER_COMPOSE) run jupyter jupyter nbconvert --to slides pymor.ipynb --post serve

qr_%.png: venv
	. venv/bin/activate && python render_qr.py

clean:
	rm qr_*png
	rm pymor.ipynb

DOCKER_COMPOSE=NB_USER=$(NB_USER) docker-compose  -f .binder/docker-compose.yml -p presentation


docker_image:
	$(DOCKER_COMPOSE) build

shell: docker_image
	$(DOCKER_COMPOSE) run jupyter bash
