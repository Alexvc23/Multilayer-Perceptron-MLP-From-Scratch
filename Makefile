.PHONY: all split train predict clean

all: split train predict

split:
	python3 split.py

train:
	python3 train.py

predict:
	python3 predict.py

clean:
	rm -rf data/*_split*.csv models/*.npy models/*.json __pycache__ src/__pycache__
