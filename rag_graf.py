import os
import re
import io
import time
import urllib.request
from dotenv import load_dotenv
from sickle import Sickle
from pypdf import PdfReader
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# Brisanje starih podataka i rekreiranje vektorskog indeksa sa novim dimenzijama (768 umesto 1536)
print("Cistim stare podatke i kreiram novi vektorski indeks...")
with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
    session.run("DROP INDEX chunk_embeddings IF EXISTS")
    session.run("""
        CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
    """)
print("Indeks spreman.")

print("Povezujem se na CRIS UNS repozitorijum...")
url = "https://cris.uns.ac.rs/api/export/OAIHandlerNaRDuS"
sickle = Sickle(url)
records = sickle.ListRecords(metadataPrefix='dim', set='theses')

# Ovde kontrolisemo koliko radova zelimo da uvezemo u ovom krugu
BROJ_RADOVA_ZA_UVOZ = 100
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

print(f"Pokrecem automatski uvoz {BROJ_RADOVA_ZA_UVOZ} radova...")
vreme_pocetka_ukupno = time.time()

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
    vreme_pocetka_rada = time.time()

    print(f"\n[{brojac_rada}/{BROJ_RADOVA_ZA_UVOZ}] Obrada: {naslov} (Autor: {autor})")

    try:
        # Preuzimanje PDF fajla sa interneta
        req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            pdf_citac = PdfReader(io.BytesIO(response.read()))

        # Ekstrakcija teksta i seckanje po paragrafima (structure-based chunking)
        chunks = []
        for br_strane, strana in enumerate(pdf_citac.pages):
            tekst_strane = strana.extract_text()
            if not tekst_strane:
                continue

            # Normalizacija artefakata PDF ekstrakcije:
            # 1. Spoji reci polomljene prelaskom u novi red (npr. "znač-\najne" -> "značajne")
            tekst_strane = re.sub(r'(\w)-\s*\n(\w)', r'\1\2', tekst_strane)
            # 2. Ukloni prelome reda unutar recenice (jedan \n koji nije granica pasusa)
            tekst_strane = re.sub(r'(?<!\n)\n(?!\n)', ' ', tekst_strane)
            # 3. Ukloni visestruke razmake nastale nakon normalizacije
            tekst_strane = re.sub(r' {2,}', ' ', tekst_strane)

            paragrafi = re.split(r'\n\s*\n', tekst_strane)
            for paragraf in paragrafi:
                paragraf = paragraf.strip()
                if len(paragraf) < 100:
                    continue  # preskaci zaglavlja, brojeve strana, kratke fragmente
                if len(paragraf) <= 1500:
                    chunks.append({"tekst": paragraf, "stranica": br_strane + 1})
                else:
                    # predugacak paragraf — seci po recenicama
                    recenice = re.split(r'(?<=[.!?])\s+', paragraf)
                    trenutni = ""
                    for recenica in recenice:
                        if len(trenutni) + len(recenica) + 1 <= 1500:
                            trenutni += (" " if trenutni else "") + recenica
                        else:
                            if len(trenutni) >= 100:
                                chunks.append({"tekst": trenutni, "stranica": br_strane + 1})
                            trenutni = recenica
                    if len(trenutni) >= 100:
                        chunks.append({"tekst": trenutni, "stranica": br_strane + 1})

        # Slanje podataka u bazu grafova
        with driver.session() as session:
            for chunk in chunks:
                response = client.embeddings.create(input=chunk["tekst"], model="nomic-embed-text")
                embedding = response.data[0].embedding
                session.execute_write(upisi_u_graf, naslov, autor, chunk["tekst"], chunk["stranica"], embedding)
        print(f"   Uspesno uneto {len(chunks)} pasusa u graf.")
        vreme_rada = time.time() - vreme_pocetka_rada
        proteklo_ukupno = time.time() - vreme_pocetka_ukupno
        prosek = proteklo_ukupno / brojac_rada
        preostalo = prosek * (BROJ_RADOVA_ZA_UVOZ - brojac_rada)
        print(f"   Vreme za ovaj rad: {vreme_rada:.0f}s | Prosek: {prosek:.0f}s/rad | Preostalo: ~{preostalo/60:.1f} min")

    except Exception as e:
        print(f"   Greska pri uvozu ovog rada: {e}")
        brojac_rada -= 1 # Ponistavamo brojac ako rad nije uspesno uvezen
        continue

driver.close()
ukupno_vreme = time.time() - vreme_pocetka_ukupno
print(f"\nUVOZ ZAVRSEN! Graf baza je uspesno osvezena novim naucnim radovima.")
print(f"Ukupno vreme: {ukupno_vreme/60:.1f} minuta | Prosek po radu: {ukupno_vreme/max(brojac_rada,1):.0f} sekundi")
