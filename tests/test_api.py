import asyncio

import httpx
import numpy as np

from rag_assistant.api import create_app
from rag_assistant.index import DocumentIndex


class KeywordEmbedder:
    def encode(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts])


def test_health_and_empty_index_response():
    async def request():
        app = create_app(DocumentIndex(KeywordEmbedder()))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            health = await client.get("/api/health")
            question = await client.post(
                "/api/questions",
                json={"question": "What does the paper say?"},
            )
            return health, question

    health, question = asyncio.run(request())

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert question.status_code == 409
