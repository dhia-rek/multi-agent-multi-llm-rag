
from src.chunking_embeddings.chunker import chunk_documents
from src.data_ingestion.loader import RawDocument


def _doc(pages, framework="fw", title="t"):
    return RawDocument(source="/x.pdf", framework=framework, title=title, pages=pages)


def test_chunks_respect_chunk_size():
    paras = "\n\n".join(f"paragraph {i} " + "word " * 10 for i in range(30))
    chunks = chunk_documents([_doc([paras])], chunk_size=200, overlap=40)
    assert len(chunks) > 1
    # a flushed chunk may exceed chunk_size only by the carried overlap + one paragraph
    assert all(len(c.text) <= 200 + 40 + 80 for c in chunks)


def test_small_page_is_single_chunk():
    chunks = chunk_documents([_doc(["alpha\n\nbeta"])], chunk_size=900)
    assert [c.text for c in chunks] == ["alpha\nbeta"]


def test_overlap_carries_tail_of_previous_chunk():
    a, b = "A" * 80, "B" * 80
    chunks = chunk_documents([_doc([f"{a}\n\n{b}"])], chunk_size=100, overlap=20)
    assert len(chunks) == 2
    assert chunks[1].text == "A" * 20 + "\n" + b


def test_zero_overlap_has_no_shared_text():
    a, b = "A" * 80, "B" * 80
    chunks = chunk_documents([_doc([f"{a}\n\n{b}"])], chunk_size=100, overlap=0)
    assert chunks[0].text == a
    assert "A" not in chunks[1].text


def test_metadata_is_carried_over():
    docs = [_doc(["", "page two text"], framework="dt_framework", title="canvas")]
    (chunk,) = chunk_documents(docs)
    assert (chunk.framework, chunk.title, chunk.page, chunk.source) == (
        "dt_framework",
        "canvas",
        2,
        "/x.pdf",
    )


def test_chunk_ids_are_sequential_across_documents():
    chunks = chunk_documents([_doc(["one"]), _doc(["two"])])
    assert [c.chunk_id for c in chunks] == ["c00000", "c00001"]


def test_empty_input():
    assert chunk_documents([]) == []
    assert chunk_documents([_doc([])]) == []
    assert chunk_documents([_doc(["", "   \n\n  "])]) == []
