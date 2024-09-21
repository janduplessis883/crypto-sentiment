install:
	@pip install --upgrade pip
	@pip install -e .
	@echo "🌵 pip install -e . completed!"

clean:
	@rm -f */version.txt
	@rm -f .DS_Store
	@rm -f .coverage
	@rm -rf */.ipynb_checkpoints
	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc
	@echo "🧽 Cleaned up successfully!"

all: install clean

app:
	@streamlit run crypto_sentiment/app.py

git_merge:
	$(MAKE) clean
	@python crypto_sentiment/automation/git_merge.py
	@echo "👍 Git Merge (master) successfull!"

git_push:
	$(MAKE) clean
	@python crypto_sentiment/automation/git_push.py
	@echo "👍 Git Push (branch) successfull!"

data:
	@python crypto_sentiment/data.py
	@echo "👍 Make DATA successfull!"

test:
	@pytest -v tests

# Specify package name
lint:
	@black crypto_sentiment/
