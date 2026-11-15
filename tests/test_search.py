import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from app.features.search.collection import INDEX_FILE, load_collection
from tests.conftest import FAKE_MODEL


def ids(body: dict) -> list[str]:
    return [hit["photograph"]["id"] for hit in body["hits"]]


def png(colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(buffer, "PNG")
    return buffer.getvalue()


async def test_a_picture_search_finds_what_the_caption_does_not_say(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/search", params={"q": "anjing", "mode": "picture"})
    assert response.status_code == 200
    body = response.json()
    assert ids(body)[:2] == ["1", "2"]
    assert not any(hit["in_caption"] for hit in body["hits"])


async def test_a_caption_search_returns_only_captions_with_a_query_word(client: httpx.AsyncClient) -> None:
    body = (await client.get("/v1/search", params={"q": "dog", "mode": "caption"})).json()
    assert ids(body) == ["1"]
    assert body["hits"][0]["in_caption"] is True


async def test_hybrid_puts_a_match_in_both_first_and_keeps_picture_matches(client: httpx.AsyncClient) -> None:
    body = (await client.get("/v1/search", params={"q": "dog"})).json()
    assert body["mode"] == "hybrid"
    assert ids(body)[:2] == ["1", "2"]
    assert [hit["in_caption"] for hit in body["hits"][:2]] == [True, False]


async def test_results_are_capped_at_the_maximum(client: httpx.AsyncClient) -> None:
    body = (await client.get("/v1/search", params={"q": "dog", "mode": "picture", "limit": 50})).json()
    assert len(body["hits"]) == 3


async def test_an_uploaded_picture_finds_photographs_that_look_like_it(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/search/image", files={"file": ("query.png", png((0, 0, 255)), "image/png")})
    assert response.status_code == 200
    assert ids(response.json())[0] == "3"


async def test_a_file_that_is_not_a_picture_is_refused(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/search/image", files={"file": ("notes.txt", b"not a picture", "text/plain")})
    assert response.status_code == 422
    assert response.json()["title"] == "Unreadable image"


async def test_a_picture_over_the_limit_is_refused(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/search/image", files={"file": ("big.png", b"0" * 60_000, "image/png")})
    assert response.status_code == 413


async def test_similar_leaves_out_the_photograph_itself(client: httpx.AsyncClient) -> None:
    body = (await client.get("/v1/photographs/1/similar")).json()
    assert "1" not in ids(body)
    assert ids(body)[0] == "2"


async def test_similar_to_an_unknown_photograph_is_404(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/photographs/999/similar")
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"


async def test_an_unknown_mode_is_422(client: httpx.AsyncClient) -> None:
    assert (await client.get("/v1/search", params={"q": "dog", "mode": "fuzzy"})).status_code == 422


async def test_the_collection_counts_photographs_per_subject(client: httpx.AsyncClient) -> None:
    body = (await client.get("/v1/collection")).json()
    assert body["photographs"] == 4
    assert body["subjects"][0] == {"subject": "Dogs", "photographs": 2}
    assert body["dimensions"] == 3


async def test_search_needs_the_api_key(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/search", params={"q": "dog"}, headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


async def test_health_and_meta_keep_the_lab_contract(client: httpx.AsyncClient) -> None:
    assert (await client.get("/health")).json() == {"status": "ok"}
    meta = (await client.get("/meta")).json()
    assert meta["id"] == "lens"
    assert meta["engine"] == FAKE_MODEL


def test_an_index_from_another_model_stops_the_service(collection_dir: Path) -> None:
    with pytest.raises(RuntimeError, match="Rebuild it"):
        load_collection(collection_dir, "gemini-embedding-2")


def test_an_index_for_other_photographs_stops_the_service(collection_dir: Path) -> None:
    index = json.loads((collection_dir / INDEX_FILE).read_text(encoding="utf-8"))
    index["ids"] = index["ids"][:2]
    (collection_dir / INDEX_FILE).write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different photographs"):
        load_collection(collection_dir, FAKE_MODEL)


def test_a_missing_index_says_how_to_build_it(collection_dir: Path) -> None:
    (collection_dir / INDEX_FILE).unlink()
    with pytest.raises(RuntimeError, match="build_index.py"):
        load_collection(collection_dir, FAKE_MODEL)


async def test_uploading_a_picture_indexes_it_for_text_and_origin_search(
    client: httpx.AsyncClient, collection_dir: Path
) -> None:
    response = await client.post(
        "/v1/photographs",
        files=[("files", ("my-boat.png", png((0, 0, 255)), "image/png"))],
        data={"title": "Harbour skiff", "tags": "Boats, Custom"},
    )
    assert response.status_code == 201
    added = response.json()["added"]
    assert len(added) == 1
    upload_id = added[0]["id"]
    assert added[0]["origin"] == "upload"
    assert (collection_dir / "images" / added[0]["image"]).is_file()

    # Semantic search restricted to uploads finds the newly added boat picture
    search_resp = await client.get("/v1/search", params={"q": "perahu", "mode": "picture", "origin": "upload"})
    assert search_resp.status_code == 200
    assert ids(search_resp.json()) == [upload_id]

    # Listing recent uploads returns it directly
    browse_resp = await client.get("/v1/photographs", params={"origin": "upload"})
    assert ids(browse_resp.json()) == [upload_id]

    # Uploading without tags assigns the Untagged category and lets gallery browse by tag without search
    untagged_resp = await client.post(
        "/v1/photographs",
        files=[("files", ("untagged.png", png((0, 255, 0)), "image/png"))],
    )
    assert untagged_resp.status_code == 201
    untagged_added = untagged_resp.json()["added"][0]
    assert untagged_added["tags"] == ["Untagged"]
    untagged_browse = await client.get("/v1/photographs", params={"origin": "all", "tag": "Untagged"})
    assert ids(untagged_browse.json()) == [untagged_added["id"]]
    await client.delete(f"/v1/photographs/{untagged_added['id']}")

    # Deleting the uploaded picture removes it cleanly, while archive pictures stay protected
    assert (await client.delete("/v1/photographs/1")).status_code == 400
    delete_resp = await client.delete(f"/v1/photographs/{upload_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["total_photographs"] == 4


async def test_typesense_store_builds_pure_vector_and_neighbour_queries() -> None:
    from app.features.search.typesense_store import TypesensePhotographStore
    from tests.conftest import BOAT, DOG, PHOTOGRAPHS

    seen_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/documents/1"):
            return httpx.Response(200, json=PHOTOGRAPHS[0].model_dump())
        if request.method == "POST" and request.url.path == "/multi_search":
            payload = json.loads(request.content.decode("utf-8"))
            seen_bodies.append(payload)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "hits": [
                                {"document": PHOTOGRAPHS[1].model_dump(), "vector_distance": 0.08},
                            ]
                        }
                    ]
                },
            )
        return httpx.Response(404)

    mock_http = httpx.AsyncClient(base_url="http://typesense.local", transport=httpx.MockTransport(handler))
    store = TypesensePhotographStore(
        "http://typesense.local", "key", "lens_photographs", search_ef=200, client=mock_http
    )
    hits = await store.search_text("anjing", DOG, "picture", 0.8, 5, "upload", set())
    assert len(hits) == 1
    assert hits[0].score == 0.92
    assert seen_bodies[0]["searches"][0]["q"] == "*"
    assert seen_bodies[0]["searches"][0]["vector_query"].endswith("k:5, ef:200)")
    assert seen_bodies[0]["searches"][0]["filter_by"] == "origin:=`upload`"

    title, neighbours = await store.similar("1", 4)
    assert title == "[Dog]"
    assert len(neighbours) == 1
    assert "id: 1" in seen_bodies[1]["searches"][0]["vector_query"]
    _ = BOAT
    await mock_http.aclose()


async def test_gemini_embedder_encodes_text_and_images(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from app.engines.embedder import GeminiEmbedder
    from app.platform.errors import EmbedderUnavailable

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen.append(body)
        if "batchEmbedContents" in request.url.path:
            count = len(body["requests"])
            return httpx.Response(200, json={"embeddings": [{"values": [3.0, 4.0, 0.0]} for _ in range(count)]})
        return httpx.Response(200, json={"embedding": {"values": [3.0, 4.0, 0.0]}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
        embedder = GeminiEmbedder(model_id="gemini-embedding-2", dimensions=3, http_async_client=mock_http)
        text_vec = await embedder.text("anjing")
        assert np.allclose(text_vec, np.array([0.6, 0.8, 0.0], dtype=np.float32))
        assert np.array_equal(await embedder.text("anjing"), text_vec)  # the repeat comes from the cache
        batch_vecs = await embedder.images(
            [Image.new("RGB", (4, 4), (10, 20, 30)), Image.new("RGB", (4, 4), (40, 50, 60))]
        )
        assert batch_vecs.shape == (2, 3)
        assert np.allclose(batch_vecs[0], np.array([0.6, 0.8, 0.0], dtype=np.float32))
        await embedder.aclose()

    query_request = seen[0]["requests"][0]
    assert query_request["outputDimensionality"] == 3
    assert query_request["content"]["parts"][0]["text"] == "anjing"
    assert "taskType" not in query_request
    assert len(seen[1]["requests"]) == 2
    assert "text" not in json.dumps(seen[1]["requests"][0]["content"])

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(429, json={"error": {"code": 429, "message": "quota"}})
        )
    ) as fail_http:
        failing = GeminiEmbedder(model_id="gemini-embedding-2", dimensions=3, http_async_client=fail_http)
        with pytest.raises(EmbedderUnavailable, match=r"Gemini.*answered 429"):
            await failing.text("anjing")
        await failing.aclose()


async def test_embedder_rotates_to_backup_key_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from app.engines.embedder import GeminiEmbedder

    monkeypatch.setenv("GEMINI_API_KEY", "key-one, key-two")
    monkeypatch.setenv("GEMINI_API_KEY_3", "key-three")
    monkeypatch.delenv("GEMINI_API_KEY_2", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    used_keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers.get("x-goog-api-key", "")
        used_keys.append(key)
        if key == "key-one":
            return httpx.Response(429, json={"error": {"code": 429, "message": "daily quota exceeded"}})
        return httpx.Response(200, json={"embeddings": [{"values": [0.0, 3.0, 4.0]}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
        embedder = GeminiEmbedder(model_id="gemini-embedding-2", dimensions=3, http_async_client=mock_http)
        vec1 = await embedder.text("surfing")
        vec2 = await embedder.text("volcano")
        await embedder.aclose()

    assert np.allclose(vec1, np.array([0.0, 0.6, 0.8], dtype=np.float32))
    assert np.allclose(vec2, np.array([0.0, 0.6, 0.8], dtype=np.float32))
    # First call tries key-one -> 429 -> key-two -> 200; second call stays on key-two
    assert used_keys == ["key-one", "key-two", "key-two"]


async def test_typesense_bootstrap_recreates_collection_when_model_changes() -> None:
    from app.features.search.typesense_store import TypesensePhotographStore
    from tests.conftest import PHOTOGRAPHS, VECTORS

    ops: list[tuple[str, str, dict | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/collections/lens_photographs":
            ops.append(("GET", request.url.path, None))
            return httpx.Response(
                200,
                json={
                    "num_documents": len(PHOTOGRAPHS),
                    "fields": [{"name": "embedding", "num_dim": 3}],
                    "metadata": {"model": "google/embeddinggemma-2"},
                },
            )
        if request.method == "DELETE":
            ops.append(("DELETE", request.url.path, None))
            return httpx.Response(200, json={})
        if request.method == "POST" and request.url.path == "/collections":
            body = json.loads(request.content.decode("utf-8"))
            ops.append(("POST", request.url.path, body))
            return httpx.Response(201, json=body)
        if request.method == "POST" and request.url.path.endswith("/documents/import"):
            ops.append(("IMPORT", request.url.path, None))
            lines = "\n".join('{"success": true}' for _ in PHOTOGRAPHS)
            return httpx.Response(200, text=lines)
        return httpx.Response(404)

    mock_http = httpx.AsyncClient(base_url="http://typesense.local", transport=httpx.MockTransport(handler))
    store = TypesensePhotographStore(
        "http://typesense.local", "key", "lens_photographs", search_ef=200, client=mock_http
    )
    await store.bootstrap(PHOTOGRAPHS, VECTORS, model="gemini-embedding-2")
    assert [op[0] for op in ops] == ["GET", "DELETE", "POST", "IMPORT"]
    assert ops[2][2] is not None and ops[2][2]["metadata"] == {"model": "gemini-embedding-2"}
    await mock_http.aclose()


async def test_picture_cutoff_and_daily_caps(collection_dir: Path) -> None:
    from app.engines.keyword import KeywordIndex
    from app.features.search.collection import load_collection
    from app.features.search.service import Lens
    from config.settings import Settings
    from tests.conftest import FakeEmbedder, make_app

    collection = load_collection(collection_dir, FAKE_MODEL)
    lens_with_cutoff = Lens(
        collection,
        FakeEmbedder(),
        KeywordIndex(collection.photographs),
        fusion_k=60,
        min_picture_score=0.7,
        collection_dir=collection_dir,
    )
    settings = Settings(
        _env_file=None,
        environment="test",
        store_backend="memory",
        api_keys="test-key",
        collection_dir=collection_dir,
        min_picture_score=0.7,
        daily_requests_per_caller=3,
        daily_uploads_per_caller=1,
    )
    transport = httpx.ASGITransport(app=make_app(settings, lens_with_cutoff))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={"X-API-Key": "test-key"}
    ) as client:
        unrelated = await client.get("/v1/search", params={"q": "spaceship", "mode": "picture"})
        assert unrelated.status_code == 200
        assert unrelated.json()["hits"] == []

        relevant = await client.get("/v1/search", params={"q": "dog", "mode": "picture"})
        assert ids(relevant.json()) == ["1", "2"]

        upload1 = await client.post("/v1/photographs", files=[("files", ("a.png", png((255, 0, 0)), "image/png"))])
        assert upload1.status_code == 201
        upload2 = await client.post("/v1/photographs", files=[("files", ("b.png", png((255, 0, 0)), "image/png"))])
        assert upload2.status_code == 429
        # Fourth /v1 call hits daily_requests_per_caller=3 on the next request
        blocked = await client.get("/v1/collection")
        assert blocked.status_code == 429
