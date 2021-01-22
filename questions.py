import nltk
import sys
import string
import numpy as np
import os


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    files=dict()
    dir=os.listdir(directory)
    for file in dir:
        loc=os.path.join(directory,file)
        files[file]=open(loc).read()

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words=[]


    for word in nltk. word_tokenize(document):
        w=word.casefold()
        if w not in nltk.corpus.stopwords.words("english"):
            ok=0
            for i in w:
                ok+=1 if i in string.punctuation else 0
            if not(len(w)==ok):
                words.append(w)

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    dic=dict()
    words=set()
    for l in documents.values():
        words.update(set(l))

    for word in words:
        dic[word]=0
        for l in documents.values():
            if word in l :
                dic[word]+=1
    N=len(documents.keys())
    X=lambda x: np.log(N/x)
    dfs=dict()
    for word in dic.keys():
        dfs[word]=X(dic[word])
    return dfs

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    rank0=list()
    for file in files.keys():
        tf_ids=0
        for word in query:
            if word in files[file]:
                tf_ids+=idfs[word]*list(files[file]).count(word)
        rank0.append((file,tf_ids))
    rank0=sorted(rank0, key=lambda x: x[1])
    rank0.reverse()
    rank=list()
    for  i in range(n):
        rank.append(rank0[i][0])
    return rank


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    rank0=list()
    for sentence in sentences.keys():
        tdfs=0
        sqt=0
        for word in query:
            if word in sentences[sentence]:
                sqt+=sentences[sentence].count(word)
                tdfs+=idfs[word]
        rank0.append((sentence,tdfs,sqt/len(sentences[sentence])))
    rank0 = sorted(rank0, key = lambda x: (x[1], x[2]))
    rank0.reverse()
    rank=list()
    for  i in range(n):
        rank.append(rank0[i][0])
    return rank


if __name__ == "__main__":
    main()
