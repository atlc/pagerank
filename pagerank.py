import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.
    """
    # 1. Create a dictionary to store probability distribution
    distribution = {}
    # 2. Get all possible pages from corpus
    pages = corpus.values()
    # 3. Get links from current page (corpus[page])
    links = corpus[page]
    
    # 4. If page has no links, distribute probability evenly among all pages
    #    - Each page gets probability 1/N where N is total number of pages
    N = len(corpus)
    LINKS_LEN = len(links)

    if LINKS_LEN == 0:
        probability = 1 / N
        for pg in corpus:
            distribution[pg] = probability
    else:
        # 5. Otherwise, calculate probabilities:
        #    - With probability (1 - damping_factor), randomly choose any page
        #      - Each page gets (1 - damping_factor)/N probability
        #    - With probability damping_factor, choose from linked pages
        #      - Each linked page gets damping_factor/L probability where L is number of links
        for pg in corpus:
            probability = (1 - damping_factor) / N
            if pg in links:
                probability += damping_factor / LINKS_LEN
            distribution[pg] = probability
    # 6. Return dictionary mapping page names to probabilities
    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values by sampling n pages according to transition model.
    """
    N = len(corpus)

    # 1. Initialize a dictionary to count visits to each page
    visits = {}

    for page in corpus.keys():
        visits[page] = 0

    # 2. Choose a random page to start
    rand = random.randint(0, N - 1)
    page = list(corpus.keys())[rand]

    # 3. Repeat n times:
    #    - Use transition_model to get probability distribution for current page
    #    - Choose next page based on probability distribution
    #    - Increment counter for chosen page
    for i in range(n):
        probability_dict = transition_model(corpus, page, damping_factor)
        next_page = random.choices(list(probability_dict.keys()), weights=list(probability_dict.values()), k=1)[0]
        visits[next_page] += 1
        page = next_page
    # 4. Convert counts to probabilities:
    #    - Divide each count by n to get probability
    #    - Ensure probabilities sum to 1
    for page in visits:
        visits[page] = visits[page] / n

    probability_sum = sum(visits.values())
    if abs(probability_sum - 1.0) > 0.001:
        raise ValueError(f"Probability check - values do not sum to 1. Value: {probability_sum}")
    
    # 5. Return dictionary mapping page names to PageRank values
    return visits


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values by iterative calculation until convergence.
    """
    N = len(corpus)
    # 1. Initialize PageRank values:
    #    - Each page starts with PR = 1/N where N is total number of pages
    ranks = {}

    for page in corpus.keys():
        ranks[page] = 1 / N

    # 2. Repeatedly update each page's PageRank according to formula:
    #    PR(p) = (1-d)/N + d * sum(PR(i)/NumLinks(i))
    #    where i represents pages linking to p    
    # 3. If page has no links, distribute its probability evenly among all pages
    # 4. Continue iterations until no PageRank value changes by more than 0.001
    while True:
        new_ranks = {}
        for page in corpus.keys():
            new_ranks[page] = (1 - damping_factor) / N
            for pg in corpus.keys():
                if page in corpus[pg]:
                    new_ranks[page] += damping_factor * ranks[pg] / len(corpus[pg])
                elif len(corpus[pg]) == 0:
                    new_ranks[page] += damping_factor * ranks[pg] / N

        if max(abs(new_ranks[page] - ranks[page]) for page in corpus.keys()) < 0.001:
            break

        ranks = new_ranks
    
    # 5. Return dictionary mapping page names to PageRank values
    return ranks


if __name__ == "__main__":
    main()
