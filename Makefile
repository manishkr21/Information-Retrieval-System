run:
	./run.sh $(QUERIES)

preprocessing:
	python3 preprocessing.py

req:
	chmod +x *
	pip install nltk
	python3 req.py