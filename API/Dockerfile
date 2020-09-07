FROM python:3.7.2

ADD . /flask-deploy

WORKDIR /flask-deploy

RUN pip install -r requirements.txt
RUN pip install gunicorn[gevent]

EXPOSE 8080

CMD gunicorn --worker-class gevent --workers 2 --bind 0.0.0.0:8080 wsgi:webapp --max-requests 10000 --timeout 60 --keep-alive 120 --log-level info