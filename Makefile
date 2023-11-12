
celery:
	celery -A celery_app worker --loglevel=info --concurrency=1 --pool=solo

server:
	python server.py

test:
	python client.py img.png mask.png --prompt "a boatsteg made of glass overlooking a large lake" --seed 1 --num_images 1 --resolution 512

.PHONY: cert
cert.pem key.pem cert.cer:
	# openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=c-marschner.de"
	openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -config openssl.cnf
	openssl x509 -in cert.pem -text -noout
	openssl x509 -in cert.pem -outform der -out cert.cer

cert: cert.pem
