import pytest

from localaudio.textchunk import chunk_text, split_sentences


def test_basic_sentences():
    assert split_sentences("Hello there. How are you? I am fine!") == [
        "Hello there.", "How are you?", "I am fine!",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Mr. Jones left early.",
        "Dr. Smith arrived at 9 a.m. on Monday.",
        "J. K. Rowling wrote many pages.",
        "Use a tool, e.g. a hammer, for this.",
        "The U.S. economy grew.",
    ],
)
def test_abbreviations_do_not_split(text):
    assert split_sentences(text) == [text]


def test_decimals_and_list_markers_do_not_split():
    assert split_sentences("Pi is 3.14 and e is 2.71.") == ["Pi is 3.14 and e is 2.71."]
    assert split_sentences("1. Introduction to the topic") == ["1. Introduction to the topic"]


def test_lowercase_after_terminator_continues_sentence():
    assert split_sentences('"Really?" she said. Then she left.') == ['"Really?" she said.', "Then she left."]
    assert split_sentences("Wait... what happened here?") == ["Wait... what happened here?"]


def test_chunks_respect_max_chars_and_never_cut_words():
    text = " ".join(f"This is sentence number {i} of the test script." for i in range(30))
    chunks = chunk_text(text, max_chars=120)
    assert len(chunks) > 5
    assert all(len(c.text) <= 120 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == text.split()  # nothing lost or cut


def test_long_sentence_is_split_at_clauses_then_words():
    sentence = ("This is a very long sentence, with many clauses; it keeps going and going: until it exceeds "
                "the limit, so we must split it somewhere sensible and not in the middle of a word at all.")
    chunks = chunk_text(sentence, max_chars=60)
    assert all(len(c.text) <= 60 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == sentence.split()
    giant_word = "x" * 200
    assert chunk_text(giant_word, max_chars=60)[0].text  # does not crash on unbreakable input


def test_pauses_paragraphs_longer_and_last_is_zero():
    chunks = chunk_text("One. Two.\n\nThree.", sentence_pause=0.2, paragraph_pause=0.7, max_chars=40)
    assert chunks[0].pause_after == pytest.approx(0.7) or chunks[0].pause_after == pytest.approx(0.2)
    assert any(c.pause_after == pytest.approx(0.7) and c.paragraph_end for c in chunks[:-1])
    assert chunks[-1].pause_after == 0.0 and chunks[-1].paragraph_end


def test_script_lines_without_punctuation_become_sentences():
    chunks = chunk_text("Welcome to the show\nToday we talk about rain\nLet us begin", max_chars=40)
    joined = " ".join(c.text for c in chunks)
    assert joined.count(".") >= 2


def test_hard_wrapped_prose_is_rejoined():
    text = "This sentence is wrapped\nacross two lines and ends here.\nThe next one follows."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert "wrapped across two lines" in chunks[0].text


def test_empty_and_whitespace():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_tiny_tail_is_merged():
    chunks = chunk_text("This is a longer opening sentence for the paragraph here. Ok.", max_chars=80, min_chars=20)
    assert len(chunks) == 1


def test_invalid_max_chars():
    with pytest.raises(ValueError):
        chunk_text("hello", max_chars=10)
