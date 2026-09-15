.PHONY: install model test api web

install:
	python -m pip install -e '.[dev]'
	cd frontend && npm install

model:
	./scripts/setup_ollama.sh

test:
	pytest
	cd frontend && npm run build

api:
	uvicorn rag_assistant.api:app --reload

web:
	cd frontend && npm run dev
