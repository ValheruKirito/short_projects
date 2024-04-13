"""Text recomendation system."""
import argparse
import nltk
import numpy as np


def parse_arguments():
    """
    Parse the input data.

    Returns
    -------
    The text to be edited.

    """
    parser = argparse.ArgumentParser(
        description="Receive text to be edited"
        )
    parser.add_argument(
        'text',
        metavar='input text',
        type=str
        )
    args = parser.parse_args()
    return args.text


def clean_input(text: str) -> str:
    """
    Cleanse the user input.

    Parameters
    ----------
    text : str
        User input text

    Returns
    -------
    str
        Sanitised text, cleansed from non ASCII characters

    """
    return str(text.encode().decode('ascii', errors='ignore'))


def preprocess_input(text: str) -> list:
    """
    Tokenize text to be further processed.

    Parameters
    ----------
    text : str
        Cleaned text to be tokenized.

    Returns
    -------
    list
        Text tokenized by sentences and words to be fed to analysis.

    """
    sentences = nltk.sent_tokenize(text)
    # using tokenizer that excludes punctiation from the tokens
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return tokens


def count_word_usage(sentence: list, is_word_used_list: list) -> int:
    """
    Count how many times any of the words in wordlist are used in a sentence.

    Parameters
    ----------
    sentence : list
        Sentence to be analysed.
    is_word_used_list : list
        List of words to be accounted for

    Returns
    -------
    num : int
        Number of times a word on a is_word_used_list is contained within a
        sentence.

    """
    num = sum(
        1 for word in sentence if word in is_word_used_list
        )
    return num


def compute_flesch_reading_ease(
        number_of_syllables: int,
        number_of_words: int,
        number_of_sentences: int):
    """
    Return Flesch reading ease score and its corresponding complexity level.

    Parameters
    ----------
    number_of_syllables : int
        number of syllables in a text
    number_of_words : int
        number of words in a text
    number_of_sentences : int
        number of sentences in a text

    Returns
    -------
    float
        Flesch reading ease score.

    """
    try:
        return (206.835
                - 1.015*(number_of_words/number_of_sentences)
                - 84.6*number_of_syllables/number_of_words)
    except ZeroDivisionError:
        return np.nan


def get_flesch_from_text(text: str) -> float:
    """
    Compute flesch score based on text input.

    Parameters
    ----------
    text : str
        Text to be scored.

    Returns
    -------
    float
        Flesch score of reading ease.

    """
    sentence_list = preprocess_input(text)
    num_words = count_total_words(sentence_list)
    num_syllab = count_total_syllables(sentence_list)
    num_sent = len(sentence_list)
    return compute_flesch_reading_ease(num_syllab, num_words, num_sent)


def get_reading_level_from_flesch(flesch: float) -> str:
    """
    Assing standardised reading level to Flesch score.

    Parameters
    ----------
    flesch : float
        Flesch reading ease score.

    Returns
    -------
    str
        Standardised reading level.

    """
    # construct thresholds
    thresholds = [90, 80, 70, 60, 50, 30, 10, 0]
    levels = ['5th grade', '6th grade', '7th grade', '8th & 9th grade',
              '10th to 12th grade', 'College', 'College graduate',
              'Professional'
              ]
    # check conditions from highest threshold to lowest (same result as a
    # sequence of elif statements)
    for thresh, lev in zip(thresholds, levels):
        if flesch >= thresh:
            return lev
        elif flesch < 0:
            return 'Professional'


def unpack_words(sentence_list: list) -> list:
    """
    Return simple list of words in all sentences.

    Parameters
    ----------
    sentence_list : list
        List of tokenised sentences – list of lists of tokens.

    Returns
    -------
    A simple list of words.

    """
    punctuation = '.,?!/'
    return [word for sentence in sentence_list for word in sentence
            if word not in punctuation]


def count_total_words(sentence_list: list) -> int:
    """
    Count number of words in the text.

    Parameters
    ----------
    sentence_list : list
        List of senteces – list of lists of tokens.

    Returns
    -------
    int
        Number of words in the text.

    """
    return len(unpack_words(sentence_list))


def compute_total_average_word_length(sentence_list: list) -> float:
    """
    Count average word length in the text.

    Parameters
    ----------
    sentence_list : list
        List of senteces – list of lists of tokens.

    Returns
    -------
    float
        Average word length.

    """
    word_list = unpack_words(sentence_list)
    # compute number of all characters using list comprehensions and the fact
    # that python treats strings as lists of characters
    num_char = sum(
        len(word) for word in word_list
        )
    # count number of words
    num_words = count_total_words(sentence_list)
    # return average length
    return num_char / num_words


def make_word_count_distribution(sentence_list: list) -> dict:
    """
    Create a dictionary of words with their counts in the text.

    Parameters
    ----------
    sentence_list : list
        List of senteces – list of lists of tokens.

    Returns
    -------
    dict
        Dictionary with individual words as keys, their counts as values.

    """
    word_list = unpack_words(sentence_list)
    # distribution for all tokens
    tokens_distrib = nltk.FreqDist(word_list)
    # filtering out numbers
    word_distrib = dict((word, count)
                        for word, count in tokens_distrib.items()
                        if word.isdigit() == 0)
    # return distribution of words
    return word_distrib


def compute_total_unique_words_fraction(sentence_list: list) -> float:
    """
    Compute fraction of all the words that are used only once.

    Parameters
    ----------
    sentence_list : list
        List of senteces – list of lists of tokens.

    Returns
    -------
    float
        Fraction of the words in the text, that are used exactly once.

    """
    # add 1 for each (word, count) pair, where count is 1
    word_distrib = make_word_count_distribution(sentence_list)
    num_unique_words = sum(
        1
        for (word, count) in word_distrib.items()
        if count == 1
        )
    # return fraction of unique words over all words
    return num_unique_words/count_total_words(sentence_list)


def count_total_syllables(sentence_list: list) -> int:
    """
    Count total number of syllables in the text.

    Parameters
    ----------
    sentence_list : list
        List of senteces – list of lists of tokens.

    Returns
    -------
    int
        Count of syllables in the text.

    """
    word_list = unpack_words(sentence_list)
    tk = nltk.tokenize.SyllableTokenizer()
    syllables = [syllable
                 for word in word_list
                 for syllable in tk.tokenize(word)]
    return len(syllables)


def get_suggestions(sentence_list: list) -> str:
    """
    Provide suggestions to improve the question based on tokenized input.

    Parameters
    ----------
    sentence_list : list
        List of senteces – list of lists of tokens.

    Returns
    -------
    str
        String containing our suggestions how to improve input.

    """
    # specific common words usage stats
    told_said_usage = sum(
        count_word_usage(
            tokens,
            ['told', 'said']
            )
        for tokens in sentence_list
        )
    but_and_usage = sum(
        count_word_usage(
            tokens,
            ['but', 'and']
            )
        for tokens in sentence_list
        )
    wh_adverbs_usage = sum(
        count_word_usage(
            tokens,
            ['when', 'where', 'why', 'whence',
             'whereby', 'wherein', 'whereupon']
            )
        for tokens in sentence_list
        )
    result_str = ''
    adverb_usage = f'Adverb usage: {told_said_usage} told/said, ' \
        f'{but_and_usage} but/and, {wh_adverbs_usage} wh adverbs.'
    result_str += adverb_usage

    # word stats
    average_word_length = compute_total_average_word_length(sentence_list)
    unique_words_fraction = compute_total_unique_words_fraction(sentence_list)
    word_stats = f'Average word length {average_word_length:2f}, ' \
        f'fraction of unique words {unique_words_fraction:2f}'
    result_str += '<\br>'
    result_str += word_stats

    # syllable stats
    number_of_syllables = count_total_syllables(sentence_list)
    number_of_words = count_total_words(sentence_list)
    number_of_sentences = len(sentence_list)
    syllable_counts = f'{number_of_syllables} syllables, ' \
        f'{number_of_words} words, {number_of_sentences} sentences'
    result_str += '<\br>'
    result_str += syllable_counts

    # flesch score
    flesch_score = compute_flesch_reading_ease(
        number_of_syllables, number_of_words, number_of_sentences
        )
    reading_level = get_reading_level_from_flesch(flesch_score)
    flesch = f'Flesch score: {flesch_score}, reading level: {reading_level}'
    result_str += '<\br>'
    result_str += flesch
    return result_str


def pipeline(text: str) -> str:
    """
    Automate user input –> recomnedation pipeline.

    Parameters
    ----------
    input_text : str
        Text to be enhanced via recomendations.

    Returns
    -------
    str
        Returns recomndation on the input.

    """
    processed = clean_input(text)
    tokenized_sentences = preprocess_input(processed)
    suggestions = get_suggestions(tokenized_sentences)
    return print(suggestions)


# =============================================================================
# Body of a code
# =============================================================================

if __name__ == '__main__':
    input_text = parse_arguments()
    print('\n\n', input_text)
    pipeline(input_text)
