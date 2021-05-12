FROM python:3.7

WORKDIR /home/shreyus_puthi/web

COPY requirements.txt /tmp/

RUN ls /tmp

# upgrade pip and install required python packages
RUN pip3 install -r /tmp/requirements.txt

RUN pip3 install --upgrade tensorflow-hub

# copy over our app code
COPY App/ .

CMD [ "python3", "./app.py" ]
