from openai import OpenAI
from dotenv import load_dotenv
import os
from google.oauth2 import service_account
from google.cloud import translate_v2 as translate

# Load the environment variables from .env file
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Translate text to a target language.
def translate_text(text, target_language, credentials_path="text_manipulation/credentials.json"):
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    client = translate.Client(credentials=credentials)
    result = client.translate(text, target_language=target_language)
    return result['translatedText']

def generate_text(sequence, myClient, language="en"):
    print("Sequence: " + sequence)
    try:
        response = myClient.chat.completions.create(
            model="gpt-3.5-turbo",  # Adjust model as needed
            messages=[
                {"role": "system", "content": "Your task is to assist in formatting text extracted from images for further processing. Convert sequences of characters into coherent English text. Ensure to maintain punctuation, correct typos, and do not add extraneous comments or questions. Process all input accordingly."},
                {"role": "user", "content": sequence}
            ]
        )

        result = response.choices[0].message.content.strip()
        return result, translate_text(result, language)
    except Exception as e:
        return str(e)