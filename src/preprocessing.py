import pandas as pd
import spacy
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor


nlp = spacy.load("en_core_web_sm")
executor = ThreadPoolExecutor()


async def lemmatize_text_spacy(text: str) -> str:
    loop = asyncio.get_running_loop()
    doc = await loop.run_in_executor(executor, lambda: nlp(text))
    lemmatized_words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    return " ".join(lemmatized_words)


async def preprocessing_company_data(data: pd.DataFrame) -> pd.DataFrame:
    if data.isnull().values.any():
        data.dropna(inplace=True)

    descriptions = data['description'].tolist()
    processed_descriptions = await asyncio.gather(*(lemmatize_text_spacy(text) for text in descriptions))
    data['description'] = processed_descriptions

    data['business_tags'] = (
        data['business_tags']
        .astype(str)
        .str.replace(r"[\[\]']", "", regex=True)
        .str.lower()
        .str.strip()
        .str.replace(",", "", regex=False)
    )

    data['sector'] = data['sector'].astype(str).str.lower().str.strip()
    data['category'] = data['category'].astype(str).str.lower().str.strip()
    data['niche'] = data['niche'].astype(str).str.lower().str.strip()

    data['extra_info'] = data['sector'] + " " + data['category'] + " " + data["niche"]
    data['full_text'] = data['description'] + " " + data['business_tags'] + " " + data['extra_info']
    data['full_text'] = data['full_text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    data.drop(columns=['description', 'business_tags', 'sector', 'category', 'niche', 'extra_info'], inplace=True)
    result = str(data)

    return result


