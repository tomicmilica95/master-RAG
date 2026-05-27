import os
import re
import io
import urllib.request
from dotenv import load_dotenv
from sickle import Sickle
from pypdf import PdfReader
from neo4j import GraphDatabase
from openai import OpenAI

# 1. Inicijalizacija i ucitavanje lozinki
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

print("⏳ Povezujem se na NaRDuS repozitorijum...")
url = "https://cris.uns.ac.rs/api/export/OAIHandlerNaRDuS"
sickle = Sickle(url)
records = sickle.ListRecords(metadataPrefix='dim', set='theses')

# Ovde kontrolisemo koliko radova zelimo da uvezemo u ovom krugu
BROJ_RADOVA_ZA_UVOZ = 3 
brojac_rada = 0

# Funkcija za upis u Neo4j graf bazu
def upisi_u_graf(tx, naslov_rada, autor_rada, tekst_chanka, br_strane, embedding):
    query = """
    MERGE (a:Autor {ime: $autor_rada})
    MERGE (d:Disertacija {naslov: $naslov_rada})
    MERGE (a)-[:PISAO]->(d)
    CREATE (c:Chunk {tekst: $tekst_chanka, stranica: $br_strane, embedding: $embedding})
    WITH d, c
    MERGE (d)-[:IMA_DEO]->(c)
    """
    tx.run(query, autor_rada=autor_rada, naslov_rada=naslov_rada, tekst_chanka=tekst_chanka, br_strane=br_strane, embedding=embedding)

print(f"🚀 Pokrecem automatski uvoz {BROJ_RADOVA_ZA_UVOZ} radova...")

for record in records:
    if brojac_rada >= BROJ_RADOVA_ZA_UVOZ:
        break
        
    xml_tekst = record.raw
    
    # Izvlacenje osnovnih metapodataka pomocu regularnih izraza
    naslov_match = re.search(r'<dim:field[^>]*element=["\']title["\'][^>]*>(.*?)</dim:field>', xml_tekst, re.DOTALL)
    naslov = naslov_match.group(1).strip() if naslov_match else "Nepoznat naslov"
    
    autor_match = re.search(r'<dim:field[^>]*qualifier=["\']author["\'][^>]*>(.*?)</dim:field>', xml_tekst, re.DOTALL)
    autor = autor_match.group(1).strip() if autor_match else "Nepoznat autor"
    
    linkovi = re.findall(r'https?://cris\.uns\.ac\.rs/api/file/[^\s<>"]+\.pdf', xml_tekst)
    if not linkovi:
        continue # Preskaci radove koji nemaju dostupan PDF link
        
    pdf_url = linkovi[0]
    brojac_rada += 1
    
    print(f"\n📦 [{brojac_rada}/{BROJ_RADOVA_ZA_UVOZ}] Obrada: {naslov} (Autor: {autor})")
    
    try:
        # Preuzimanje PDF fajla sa interneta
        req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            pdf_citac = PdfReader(io.BytesIO(response.read()))
        
        # Ekstrakcija teksta i seckanje na pasuse (chunks)
        chunks = []
        trenutni_tekst = ""
        for br_strane, strana in enumerate(pdf_citac.pages[:15]): # Uzimamo prvih 15 strana radi optimizacije budzeta
            tekst_strane = strana.extract_text()
            if not tekst_strane:
                continue
            trenutni_tekst += f"\n[Stranica {br_strane+1}]\n" + tekst_strane
            
            while len(trenutni_tekst) >= 1000:
                chunks.append({"tekst": trenutni_tekst[:1000], "stranica": br_strane + 1})
                trenutni_tekst = trenutni_tekst[1000:]
        
        # Slanje podataka u bazu grafova
        with driver.session() as session:
            for chunk in chunks:
                response = client.embeddings.create(input=chunk["tekst"], model="text-embedding-3-small")
                embedding = response.data[0].embedding
                session.execute_write(upisi_u_graf, naslov, autor, chunk["tekst"], chunk["stranica"], embedding)
        print(f"   ✅ Uspesno uneto {len(chunks)} pasusa u graf.")
        
    except Exception as e:
        print(f"   ❌ Greska pri uvozu ovog rada: {e}")
        brojac_rada -= 1 # Ponistavamo brojac ako rad nije uspesno uvezen
        continue

driver.close()
print(f"\n🎉 UVOZ ZAVRŠEN! Graf baza je uspesno osvezena novim naucnim radovima.")