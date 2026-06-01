import os
import re
import unicodedata
import uuid
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

st.set_page_config(page_title="CRIS UNS GraphRAG Asistent", page_icon=None, layout="centered")

load_dotenv()

@st.cache_resource
def inicijalizuj_veze():
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    return client, driver

client, driver = inicijalizuj_veze()

st.title("CRIS UNS - GraphRAG Asistent")
st.markdown("Postavite pitanje u vezi sa dokumentima koji se nalaze u bazi znanja.")
st.info("Sistem koristi hibridnu pretragu kroz graf bazu kako bi pronasao tacne odgovore.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}  # {poruka_id: "pozitivan" | "negativan"}


def sacuvaj_feedback(poruka_id, pitanje, odgovor, ocena):
    with driver.session() as session:
        session.run(
            """
            CREATE (f:Feedback {
                id: $id,
                pitanje: $pitanje,
                odgovor: $odgovor,
                ocena: $ocena,
                vreme: datetime()
            })
            """,
            {"id": poruka_id, "pitanje": pitanje, "odgovor": odgovor, "ocena": ocena},
        )


def prikazi_feedback_dugmice(poruka_id, pitanje, odgovor):
    if poruka_id in st.session_state.feedback:
        ocena = st.session_state.feedback[poruka_id]
        ikona = "👍" if ocena == "pozitivan" else "👎"
        st.caption(f"{ikona} Hvala na povratnoj informaciji!")
    else:
        col1, col2, _ = st.columns([1, 1, 12])
        with col1:
            if st.button("👍", key=f"up_{poruka_id}", help="Korisno"):
                sacuvaj_feedback(poruka_id, pitanje, odgovor, "pozitivan")
                st.session_state.feedback[poruka_id] = "pozitivan"
                st.rerun()
        with col2:
            if st.button("👎", key=f"down_{poruka_id}", help="Nije korisno"):
                sacuvaj_feedback(poruka_id, pitanje, odgovor, "negativan")
                st.session_state.feedback[poruka_id] = "negativan"
                st.rerun()


# Prikaz istorije razgovora
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("izvori") and "ne mogu da odgovorim" not in message["content"].lower():
                with st.expander("Pogledaj izvore iz baze"):
                    for izvor in message["izvori"]:
                        st.write(izvor)
            prikazi_feedback_dugmice(
                message["id"], message["pitanje"], message["content"]
            )


def je_meta_pitanje(pitanje):
    p = pitanje.lower()
    kljucne_reci = [
        "koji radovi", "koje disertacije", "koji dokumenti", "sta je u bazi",
        "sta se nalazi u bazi", "popis radova", "lista radova", "koje radove",
        "koliko radova", "sta pretrazujes", "koje radove pretrazujes",
        "koji su radovi", "koje su disertacije", "sta imas u bazi",
        "koji se radovi", "koji dokumenti su",
    ]
    return any(kljuc in p for kljuc in kljucne_reci)


def lista_disertacija():
    with driver.session() as session:
        rezultati = session.run("""
            MATCH (a:Autor)-[:PISAO]->(d:Disertacija)-[:IMA_DEO]->(:Chunk)
            RETURN DISTINCT d.naslov AS naslov, a.ime AS autor
            ORDER BY d.naslov
        """)
        radovi = [(r["naslov"], r["autor"]) for r in rezultati]

    if not radovi:
        return "Baza znanja je trenutno prazna.", []

    odgovor = f"U bazi znanja trenutno se nalaze **{len(radovi)}** doktorske disertacije:\n\n"
    for i, (naslov, autor) in enumerate(radovi, 1):
        odgovor += f"{i}. **{naslov}**  \n   *Autor: {autor}*\n\n"
    return odgovor, []


def izvuci_naslov_iz_pitanja(pitanje):
    navodnici = re.findall(r'[""„\'"](.*?)[""„\'"]', pitanje)
    if navodnici:
        return max(navodnici, key=len)
    cirilica = re.findall(r'[Ѐ-ӿ][Ѐ-ӿ\s,]*[Ѐ-ӿ]', pitanje)
    if cirilica:
        return max(cirilica, key=len).strip()
    return None


_CIR_U_LAT = [
    ('љ','lj'),('Љ','Lj'),('њ','nj'),('Њ','Nj'),('џ','dž'),('Џ','Dž'),
    ('а','a'),('б','b'),('в','v'),('г','g'),('д','d'),('ђ','đ'),
    ('е','e'),('ж','ž'),('з','z'),('и','i'),('ј','j'),('к','k'),
    ('л','l'),('м','m'),('н','n'),('о','o'),('п','p'),('р','r'),
    ('с','s'),('т','t'),('ћ','ć'),('у','u'),('ф','f'),('х','h'),
    ('ц','c'),('ч','č'),('ш','š'),
    ('А','A'),('Б','B'),('В','V'),('Г','G'),('Д','D'),('Ђ','Đ'),
    ('Е','E'),('Ж','Ž'),('З','Z'),('И','I'),('Ј','J'),('К','K'),
    ('Л','L'),('М','M'),('Н','N'),('О','O'),('П','P'),('Р','R'),
    ('С','S'),('Т','T'),('Ћ','Ć'),('У','U'),('Ф','F'),('Х','H'),
    ('Ц','C'),('Ч','Č'),('Ш','Š'),
]

def normalizuj_za_pretragu(tekst):
    for cir, lat in _CIR_U_LAT:
        tekst = tekst.replace(cir, lat)
    return unicodedata.normalize('NFD', tekst.lower()).encode('ascii', 'ignore').decode('ascii')


def pronadji_disertaciju_po_naslovu(naslov_fragment):
    with driver.session() as session:
        # Prvo pokušaj direktno poređenje (ćirilica = ćirilica)
        rezultat = session.run(
            "MATCH (d:Disertacija) WHERE toLower(d.naslov) CONTAINS toLower($fragment) RETURN d.naslov AS naslov LIMIT 1",
            {"fragment": naslov_fragment[:100]},
        )
        zapis = rezultat.single()
        if zapis:
            return zapis["naslov"]

        # Fuzzy: transliteraj ćirilične naslove u latinicu, ukloni dijakritike, poređaj reči
        svi = session.run("MATCH (d:Disertacija) RETURN d.naslov AS naslov")
        svi_naslovi = [r["naslov"] for r in svi]

    if not svi_naslovi:
        return None

    fragment_norm = normalizuj_za_pretragu(naslov_fragment)
    fragment_reci = set(fragment_norm.split())

    najbolji_naslov, najbolji_skor = None, 0.0
    for naslov in svi_naslovi:
        naslov_reci = set(normalizuj_za_pretragu(naslov).split())
        if not naslov_reci:
            continue
        skor = len(fragment_reci & naslov_reci) / len(naslov_reci)
        if skor > najbolji_skor:
            najbolji_skor, najbolji_naslov = skor, naslov

    return najbolji_naslov if najbolji_skor >= 0.5 else None


def generisi_odgovor(pitanje_korisnika):
    response_emb = client.embeddings.create(
        input=pitanje_korisnika,
        model="nomic-embed-text",
    )
    pitanje_embedding = response_emb.data[0].embedding

    # Ako pitanje imenuje konkretan rad (ćirilica ili navodnici), pretraži unutar njega
    naslov_fragment = izvuci_naslov_iz_pitanja(pitanje_korisnika)
    naslov_disertacije = pronadji_disertaciju_po_naslovu(naslov_fragment) if naslov_fragment else None

    if naslov_disertacije:
        # Graf pretraga: direktno povuci chunkove iz imenovanog rada
        upit_za_bazu = """
        MATCH (d:Disertacija {naslov: $naslov})-[:IMA_DEO]->(c:Chunk)
        RETURN c.tekst AS tekst, c.stranica AS stranica, d.naslov AS naslov, 1.0 AS score
        ORDER BY c.stranica ASC LIMIT 5
        """
        params = {"naslov": naslov_disertacije}
    else:
        # Standardna vektorska pretraga
        upit_za_bazu = """
        CALL db.index.vector.queryNodes('chunk_embeddings', 3, $pitanje_embedding)
        YIELD node, score
        MATCH (d:Disertacija)-[:IMA_DEO]->(node)
        RETURN node.tekst AS tekst, node.stranica AS stranica, d.naslov AS naslov, score
        """
        params = {"pitanje_embedding": pitanje_embedding}

    pronadjeni_pasusi = []
    with driver.session() as session:
        rezultati = session.run(upit_za_bazu, params)
        for zapis in rezultati:
            pronadjeni_pasusi.append(zapis)

    if not naslov_disertacije and (not pronadjeni_pasusi or max(p["score"] for p in pronadjeni_pasusi) < 0.70):
        return "Na osnovu dokumenata u bazi, ne mogu da pronadjem odgovor.", []

    if not pronadjeni_pasusi:
        return f"Rad '{naslov_fragment}' nije pronađen u bazi znanja.", []

    kontekst = ""
    vidljivi_radovi = {}
    for i, pasus in enumerate(pronadjeni_pasusi):
        naslov = pasus['naslov']
        if naslov not in vidljivi_radovi or pasus['score'] > vidljivi_radovi[naslov]:
            vidljivi_radovi[naslov] = pasus['score']
        kontekst += (
            f"\n--- IZVOR {i+1} ---\n"
            f"Naslov disertacije: {naslov}\n"
            f"Stranica: {pasus['stranica']}\n"
            f"Tekst:\n{pasus['tekst']}\n"
        )

    izvori = [
        f"Rad: '{naslov}' (Slicnost: {round(score*100, 1)}%)"
        for naslov, score in vidljivi_radovi.items()
    ]

    sistemski_prompt = (
        "Ti si strucni asistent za univerzitetske doktorske disertacije. "
        "Odgovori na korisnikovo pitanje koristeci iskljucivo prilozeni kontekst — "
        "ukljucujuci i naslove disertacija i tekst pasusa, jer naslovi mogu sadrzati kljucne informacije. "
        "Odgovori na srpskom jeziku (koristi kvacice u recima), budi profesionalan i precizan. "
        "Ako kontekst sadrzi delimicno relevantne informacije, upotrebi ih i jasno naznaci na osnovu cega odgovaras. "
        "Samo ako prilozen kontekst UOPSTE nije relevantan za pitanje, reci tacno: "
        "'Na osnovu trenutnih dokumenata u bazi ne mogu da odgovorim na to pitanje.'"
    )
    korisnicki_prompt = f"Tekst iz disertacije:\n{kontekst}\n\nPitanje: {pitanje_korisnika}"

    odgovor_ai = client.chat.completions.create(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": sistemski_prompt},
            {"role": "user", "content": korisnicki_prompt},
        ],
        temperature=0.2,
    )

    return odgovor_ai.choices[0].message.content, izvori


# Polje za unos pitanja
if pitanje := st.chat_input("Pitajte nesto..."):
    with st.chat_message("user"):
        st.markdown(pitanje)
    st.session_state.messages.append({"role": "user", "content": pitanje})

    with st.chat_message("assistant"):
        with st.spinner("Pretrazujem graf znanja..."):
            if je_meta_pitanje(pitanje):
                odgovor, izvori = lista_disertacija()
            else:
                odgovor, izvori = generisi_odgovor(pitanje)
            poruka_id = str(uuid.uuid4())

            st.markdown(odgovor)

            if izvori and "ne mogu da odgovorim" not in odgovor.lower():
                with st.expander("Pogledaj izvore iz baze"):
                    for izvor in izvori:
                        st.write(izvor)

            prikazi_feedback_dugmice(poruka_id, pitanje, odgovor)

    st.session_state.messages.append({
        "role": "assistant",
        "content": odgovor,
        "id": poruka_id,
        "pitanje": pitanje,
        "izvori": izvori,
    })
