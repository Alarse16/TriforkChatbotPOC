# README

## Oversigt
Dette projekt er en proof-of-concept (POC) til en Retrieval-Augmented Generation (RAG) løsning. Systemet udfører følgende:
- Ekstraherer tekst fra billeder ved hjælp af OCR (Tesseract).
- Indekserer og søger i tekstdata med FAISS.
- Genererer svar på spørgsmål baseret på indlejrede data og en LLM (Generative AI).

## Arkitektur
### Komponenter
1. **ImageHandler.py**:
   - Ekstraherer tekst fra billeder i et angivet inputbibliotek og gemmer som tekstfiler.
   - Bruger EasyOCR og OpenCV.

2. **StorageHandler.py**:
   - Indlæser tekstdata, splitter i chunks, og skaber embeddings ved brug af en LLM.
   - Opretter og administrerer en FAISS-indeks til hurtige semantiske søgninger.

3. **Main.py**:
   - Hovedprogrammet med en GUI til spørgsmål/svar interaktioner.
   - Håndterer første kørsel ved at udføre initialisering og opsætning af data.

## Implementeringsdetaljer
- **Værktøjer og biblioteker**:
  - OCR: EasyOCR og OpenCV.
  - Indeksering: FAISS.
  - LLM: Google Generative AI via `google.generativeai` bibliotek.
  - GUI: Tkinter.
  - Teksthåndtering: `langchain_community`.

- **Teknologi**:
  - FAISS bruges til at udføre semantiske søgninger med embeddings.
  - Generative AI bruges til tekstembedding og generering af svar.

## Sådan køres POC’en
### Krav
- Python 3.10+ installeret.
- Installer alle biblioteker i requirements.txt.

### Trin
1. Kør **[Main.py](./Main.py)**. Den vil automatisk
   - Bruge OCR til at læse billederne i projektet og udvinde text
   - Indlæser tekstdata, splitter i chunks, og skaber embeddings
   - Opretter og administrerer en FAISS-indeks til hurtige semantiske søgninger
   - Åbne Chat windowed

3. Kør **[Tester.py](./Tester.py)** seperat for at fuldføre en kort unit test hvor en seperat LLM bruges til at validere svar givet af hoved LLM'en.
  
