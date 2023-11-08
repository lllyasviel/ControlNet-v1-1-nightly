
celery:
	celery -A celery_app worker --loglevel=info --concurrency=1 --pool=solo

server:
	python server.py

test:
	python client.py img.png mask.png --prompt "a red drawer"
