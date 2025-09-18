from src.load_articles import (
    initialize_pinecone,
    fetch_articles,
    process_feeds_with_cache,
    load_cached_urls,
    save_cached_urls,
)
from src import load_articles as la
from src.data_models import Article


def test_initialize_pinecone_creates_when_missing(mocker):
    mock_pc = mocker.MagicMock()
    mock_pc.list_indexes.return_value.names.return_value = []

    mocker.patch("src.load_articles.Pinecone", return_value=mock_pc)

    idx = initialize_pinecone("k", None, "idx", 384)
    assert mock_pc.create_index.called
    assert idx == mock_pc.Index.return_value


def test_initialize_pinecone_skips_when_exists(mocker):
    mock_pc = mocker.MagicMock()
    mock_pc.list_indexes.return_value.names.return_value = ["idx"]

    mocker.patch("src.load_articles.Pinecone", return_value=mock_pc)

    idx = initialize_pinecone("k", None, "idx", 384)
    mock_pc.create_index.assert_not_called()
    assert idx == mock_pc.Index.return_value


def test_main_flow_invokes_initialize_and_process(mocker):
    called = {"init": False, "proc": False}

    def fake_init(*args, **kwargs):
        called["init"] = True
        return mocker.MagicMock()

    def fake_proc(*args, **kwargs):
        called["proc"] = True

    mocker.patch.object(la, "initialize_pinecone", side_effect=fake_init)
    mocker.patch.object(
        la,
        "SentenceTransformer",
        return_value=mocker.MagicMock(
            encode=mocker.MagicMock(
                return_value=type("V", (), {"tolist": lambda self: [0.1]})()
            )
        ),
    )
    mocker.patch.object(la, "process_feeds_with_cache", side_effect=fake_proc)

    la.main()
    assert called["init"] and called["proc"]


def test_fetch_articles_parses_feed_entries(mocker):
    mock_feed = mocker.MagicMock()
    mock_feed.entries = [
        mocker.MagicMock(
            title="T1",
            link="U1",
            published="P1",
            summary="S1",
            get=lambda *args, **kwargs: [{"value": "C1"}],
        )
    ]

    mocker.patch("src.load_articles.feedparser.parse", return_value=mock_feed)

    articles = fetch_articles("http://feed")
    assert len(articles) == 1
    assert articles[0].title == "T1"
    assert articles[0].link == "U1"


def test_process_feeds_with_cache_skips_cached_and_upserts_new(tmp_path, mocker):
    feeds = {"cat": "http://feed"}
    model = mocker.MagicMock()

    class _Vec:
        def tolist(self):
            return [0.1, 0.2, 0.3]

    model.encode.return_value = _Vec()

    mock_index = mocker.MagicMock()

    # One cached URL, one new
    cached_file = tmp_path / "cache.pkl"

    # Mock helpers and fetcher
    mocker.patch("src.load_articles.load_cached_urls", return_value={"U1"})
    save_mock = mocker.patch("src.load_articles.save_cached_urls")
    mocker.patch(
        "src.load_articles.fetch_articles",
        return_value=[
            Article(title="T1", summary="S1", link="U1", published="P1"),
            Article(title="T2", summary="S2", link="U2", published="P2"),
        ],
    )

    process_feeds_with_cache(feeds, model, mock_index, str(cached_file))

    # Should upsert only the new one
    assert mock_index.upsert.call_count == 1
    # Cache saved
    assert save_mock.called


def test_initialize_pinecone_handles_list_indexes_exception(mocker):
    mock_pc = mocker.MagicMock()
    mock_pc.list_indexes.side_effect = Exception("boom")

    mocker.patch("src.load_articles.Pinecone", return_value=mock_pc)

    _ = initialize_pinecone("k", None, "idx2", 384)
    assert mock_pc.create_index.called


def test_load_and_save_cached_urls(tmp_path):
    p = tmp_path / "cache.pkl"
    # Initially empty -> set()
    assert load_cached_urls(str(p)) == set()
    # Save and load back
    urls = {"a", "b"}
    save_cached_urls(urls, str(p))
    assert load_cached_urls(str(p)) == urls


def test_process_feeds_with_cache_index_none(mocker):
    # Should early return with no exception
    process_feeds_with_cache({}, mocker.MagicMock(), None, "cache.pkl")


def test_process_feeds_with_cache_missing_title_or_summary(tmp_path, mocker):
    feeds = {"cat": "http://feed"}
    model = mocker.MagicMock()

    class _Vec:
        def tolist(self):
            return [0.1]

    model.encode.return_value = _Vec()
    mock_index = mocker.MagicMock()
    cached_file = tmp_path / "cache.pkl"

    mocker.patch("src.load_articles.load_cached_urls", return_value=set())
    mocker.patch("src.load_articles.save_cached_urls")

    # Missing title
    mocker.patch(
        "src.load_articles.fetch_articles",
        return_value=[
            Article(title="", summary="S", link="U3", published="P"),
        ],
    )
    process_feeds_with_cache(feeds, model, mock_index, str(cached_file))

    # Missing summary
    mocker.patch(
        "src.load_articles.fetch_articles",
        return_value=[
            Article(title="T", summary="", link="U4", published="P"),
        ],
    )
    process_feeds_with_cache(feeds, model, mock_index, str(cached_file))

    # No upserts should have happened
    mock_index.upsert.assert_not_called()
