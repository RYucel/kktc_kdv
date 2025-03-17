from flask import Flask, request, render_template
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Securely retrieve API key from environment variable
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")

client = genai.Client(api_key=API_KEY)

# Path to the static PDF file
PDF_PATH = "KDVTuzuk.pdf"

@app.route('/')
def index():
    """Render the main page with the question input form."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Handle form submission, generate response, and display it."""
    user_input = request.form['question']

    # Upload the PDF file
    file = client.files.upload(file=PDF_PATH)

    # Construct the content with example and user question
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=file.uri,
                    mime_type=file.mime_type,
                ),
                types.Part.from_text(text="Süt KDV oranı nedir?"),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""The user is asking about the VAT rate for milk based on the provided document. I need to find the section in the document that mentions milk and check its VAT rate. Looking at the document, specifically page 4, I see item 5 under section "CETVEL I (Madde 3) YÜZDE SIFIR (% 0) VERGİ ORANI UYGULANACAK MAL VE HİZMETLER (A) MAL TESLİMLERİ:". Item 5 states: "5- Süt (konsantre edilmiş veya ilâve şeker veya diğer tatlandırıcı veya katkı maddeleri içeren sütler ile aromalandırılmış olanlar dahil);" This clearly states that milk, including various forms like concentrated or flavored milk, falls under the zero percent VAT rate. Therefore, the VAT rate for milk is 0%."""),
                types.Part.from_text(text="""Belgede Süt için KDV oranının %0 (Sıfır) olduğu belirtilmiştir. Belgenin 4. sayfasında, CETVEL I (Madde 3) YÜZDE SIFIR (% 0) VERGİ ORANI UYGULANACAK MAL VE HİZMETLER (A) MAL TESLİMLERİ: başlığı altında 5. madde şu şekildedir: 5- Süt (konsantre edilmiş veya ilâve şeker veya diğer tatlandırıcı veya katkı maddeleri içeren sütler ile aromalandırılmış olanlar dahil); Bu maddeye göre, süt (çeşitli formları dahil) %0 KDV oranı ile vergilendirilmektedir."""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_input),
            ],
        ),
    ]

    # Configuration for content generation
    generate_content_config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        max_output_tokens=65536,
        response_mime_type="text/plain",
    )

    # Generate the response (non-streaming)
    model = "gemini-2.0-flash-thinking-exp-01-21"
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    # Extract the response text
    response_text = response.candidates[0].content.parts[0].text

    return render_template('response.html', response=response_text)

if __name__ == '__main__':
    app.run(debug=True)