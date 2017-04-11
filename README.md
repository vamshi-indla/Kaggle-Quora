# Kaggle-Quora
Kaggle_Quora Question Pairs Competition

Welcome to NLP wiki! 


1. Spell Checks
    1.1 Text Blob
    This is considered a light weight implementation for text mining. Its not primarily a NLP package.
    
    >>> from textblob import TextBlob
    >>> b = TextBlob("I havv goood speling!")
    >>> print(b.correct())
        I have good spelling!

    1.2 Google Spell Checker
        
        import requests
        import re
        import time
        from random import randint

        START_SPELL_CHECK="<span class=\"spell\">Showing results for</span>"
        END_SPELL_CHECK="<br><span class=\"spell_orig\">Search instead for"

        HTML_Codes = (
            ("'", '&#39;'),
            ('"', '&quot;'),
            ('>', '&gt;'),
            ('<', '&lt;'),
            ('&', '&amp;'),
        )

        def spell_check(s):
          q = '+'.join(s.split())
          time.sleep(  randint(0,2) ) #relax and don't let google be angry
          r = requests.get("https://www.google.co.uk/search?q="+q)
          content = r.text
          start=content.find(START_SPELL_CHECK) 
          if ( start > -1 ):
            start = start + len(START_SPELL_CHECK)
            end=content.find(END_SPELL_CHECK)
            search= content[start:end]
            search = re.sub(r'<[^>]+>', '', search)
            for code in HTML_Codes:
              search = search.replace(code[1], code[0])
            search = search[1:]
          else:
            search = s
          return search ;


        ###samples
        #searches = [ "metal plate cover gcfi", 'artric air portable", "roll roofing lap cemet", "basemetnt window", 
        #            "vynal grip strip", "lawn mower- electic" ]
        #for search in searches:
          #speel_check_search= spell_check(search)
          #print (search+"->" + speel_check_search)
    
2. Word Normalization
  
    2.1 Morphological Normalization
        Stemming - Truncates the word to a more static word. Ex: University reduces to Univers
        Lemmitization - Returns base word that is avaiable in Dictionary. Ex: Went to Go 
        
        Stemming makes the word invalid sometimes, which is not in lexicon.
        Lemmitization returns a word which is in Lexicon.
        
        Lemmitization converts
        > Verb forms are reduced to the infinitive.
        > Inflected forms of nouns are reduced to the nominative singular.
        > Comparatives and superlatives of gradable adjectives are reduced to the absolute form.
        
        Downnside:
        Lemmitization can reduce a noun and verb to same Lemma. Ex: Attached and attacks -> Attack
        
        
        
     2.9 References
         Normalization: http://www.aviarampatzis.com/Avi_Arampatzis/publications/HTMLized/encyclop/node5.html
         TextBlob: https://textblob.readthedocs.io/en/dev/quickstart.html#spelling-correction
         Google Spell Checker: https://www.kaggle.com/steubk/home-depot-product-search-relevance/fixing-typos

2. Tokenization

  2.1 Stemming
  
  2.2 Lemmitization
  
3. Fuzzy Wuzzy
3. Word Embeddings
  3.1 Word2vec
  3.2 Glove
  3.3 Spacy
  3.4 Google News
4. Models
  4.1 Keras
  4.2 Tensor flow
  4.3 Ensemble
  
  
  
# References

1. Fuzzywuzzy and Word2Vec Implementation by Abhishek Thakur

https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur


