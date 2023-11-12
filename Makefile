
celery:
	celery -A celery_app worker --loglevel=info --concurrency=1 --pool=solo

server:
	python server.py

test:
	python client.py img.png mask.png --prompt "a boatsteg made of glass overlooking a large lake" --seed 1 --num_images 1 --resolution 512
