import time
from loguru import logger
from nlpretext import Preprocessor
from nlpretext.basic.preprocess import (
    normalize_whitespace,
    remove_punct,
    remove_eol_characters,
    remove_stopwords,
    lower_text,
)
from nlpretext.social.preprocess import remove_mentions, remove_hashtag, remove_emoji



# = Decorators =========================================================================================================

def time_it(func):
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"🖥️    Started: '{func_name}'")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        func_name = func.__name__
        logger.info(f"✅ Completed: '{func_name}' ⚡️{elapsed_time:.6f} sec")
        return result

    return wrapper


# Universal functions ==================================================================================================

def text_preprocessing_nlpretext(text):
    '''
    Text pre-prcessing for NLP use with NLPretext
    Hash out the functions not needed.
    
    Suggested use: (skips empty rows)
    # data["text_column"] = data["text_column"].apply(lambda x: text_preprocessing(str(x)) if not pd.isna(x) else np.nan)
    '''
    logger.info("⭐️ Text Preprocesssing with *NLPretext")

    preprocessor = Preprocessor()
    preprocessor.pipe(lower_text)
    preprocessor.pipe(remove_mentions)
    preprocessor.pipe(remove_hashtag)
    preprocessor.pipe(remove_emoji)
    preprocessor.pipe(remove_eol_characters)
    # preprocessor.pipe(remove_stopwords, args={'lang': 'en'})
    preprocessor.pipe(remove_punct)
    preprocessor.pipe(normalize_whitespace)
    text = preprocessor.run(text)

    return text

