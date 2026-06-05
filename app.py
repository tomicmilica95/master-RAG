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

# Full-text indeks — kreira se jednom, idempotentna operacija
with driver.session() as _s:
    _s.run("CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.tekst]")

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
        "izlistaj", "sve disertacije", "sve radove", "nabrojati", "nabroj",
        "prikazi sve", "prikaži sve", "lista disertacija", "lista radova",
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
    # Pokusaj da nadjes naslov rada pomenut bez navodnika (lat. fraza iza "u radu", "iz rada", "rad ")
    bez_navodnika = re.search(
        r'(?:u radu|iz rada|rad[au]?|in paper|in the paper|paper)\s+([A-ZŠĐČĆŽ][^\?\.]{10,})',
        pitanje, re.IGNORECASE
    )
    if bez_navodnika:
        return bez_navodnika.group(1).strip()
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


_LAT_U_CIR = [
    ('lj','љ'),('LJ','Љ'),('Lj','Љ'),('nj','њ'),('NJ','Њ'),('Nj','Њ'),
    ('dž','џ'),('DŽ','Џ'),('Dž','Џ'),('dj','ђ'),('DJ','Ђ'),('Dj','Ђ'),
    ('š','ш'),('Š','Ш'),('č','ч'),('Č','Ч'),('ć','ћ'),('Ć','Ћ'),
    ('ž','ж'),('Ž','Ж'),('đ','ђ'),('Đ','Ђ'),
    ('a','а'),('b','б'),('v','в'),('g','г'),('d','д'),('e','е'),
    ('z','з'),('i','и'),('j','ј'),('k','к'),('l','л'),('m','м'),
    ('n','н'),('o','о'),('p','п'),('r','р'),('s','с'),('t','т'),
    ('u','у'),('f','ф'),('h','х'),('c','ц'),
    ('A','А'),('B','Б'),('V','В'),('G','Г'),('D','Д'),('E','Е'),
    ('Z','З'),('I','И'),('J','Ј'),('K','К'),('L','Л'),('M','М'),
    ('N','Н'),('O','О'),('P','П'),('R','Р'),('S','С'),('T','Т'),
    ('U','У'),('F','Ф'),('H','Х'),('C','Ц'),
]

_STOPWORDS = {
    "je", "su", "i", "u", "na", "za", "se", "a", "o", "ali", "da",
    "ili", "od", "do", "iz", "po", "sa", "koji", "koja", "koje",
    "sta", "kako", "kada", "gde", "ovo", "ova", "ovaj", "ne", "ni",
    "vec", "jos", "uvek", "samo", "ima", "nema", "the", "and", "of",
    "in", "for", "is", "are", "ovom", "ovim", "radu", "rad",
}

def u_cirilicu(tekst):
    for lat, cir in _LAT_U_CIR:
        tekst = tekst.replace(lat, cir)
    return tekst

def izvuci_kljucne_reci(pitanje):
    reci = re.findall(r'\b\w+\b', pitanje.lower())
    return [r for r in reci if r not in _STOPWORDS and len(r) > 3]


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
        pronadjeni_pasusi = []
        vidjena_tekst_graf = set()
        with driver.session() as session:
            # Vektorska pretraga: globalni top-100 filtrirani na konkretan rad
            for zapis in session.run("""
                CALL db.index.vector.queryNodes('chunk_embeddings', 100, $emb)
                YIELD node, score
                WHERE size(node.tekst) > 500
                MATCH (d:Disertacija {naslov: $naslov})-[:IMA_DEO]->(node)
                RETURN node.tekst AS tekst, node.stranica AS stranica, d.naslov AS naslov, score
            """, {"naslov": naslov_disertacije, "emb": pitanje_embedding}):
                if zapis["tekst"] not in vidjena_tekst_graf:
                    vidjena_tekst_graf.add(zapis["tekst"])
                    pronadjeni_pasusi.append({**dict(zapis), "tip": "graf"})
            # Full-text pretraga kljucnih reci unutar konkretnog rada
            kljucne_reci_grafa = izvuci_kljucne_reci(pitanje_korisnika)
            if kljucne_reci_grafa:
                cir_reci_grafa = [u_cirilicu(r) for r in kljucne_reci_grafa]
                ft_upit_graf = " OR ".join(
                    [re.sub(r'[+\-!(){}\[\]^"~*?:\\/]', ' ', r) for r in cir_reci_grafa + kljucne_reci_grafa]
                )
                try:
                    for zapis in session.run("""
                        CALL db.index.fulltext.queryNodes('chunk_fulltext', $upit, {limit: 20})
                        YIELD node, score
                        WHERE size(node.tekst) > 500
                        MATCH (d:Disertacija {naslov: $naslov})-[:IMA_DEO]->(node)
                        RETURN node.tekst AS tekst, node.stranica AS stranica, d.naslov AS naslov, score
                    """, {"upit": ft_upit_graf, "naslov": naslov_disertacije}):
                        if zapis["tekst"] not in vidjena_tekst_graf:
                            vidjena_tekst_graf.add(zapis["tekst"])
                            pronadjeni_pasusi.append({**dict(zapis), "tip": "graf"})
                except Exception:
                    pass
        # Sortiraj po scoru, top 3 za LLM
        pronadjeni_pasusi = sorted(pronadjeni_pasusi, key=lambda p: p["score"], reverse=True)[:3]
    else:
        pronadjeni_pasusi = []
        vidjena_tekst = set()

        with driver.session() as session:
            for z in session.run("""
                CALL db.index.vector.queryNodes('chunk_embeddings', 5, $emb)
                YIELD node, score
                WHERE size(node.tekst) > 500
                MATCH (d:Disertacija)-[:IMA_DEO]->(node)
                RETURN node.tekst AS tekst, node.stranica AS stranica, d.naslov AS naslov, score
            """, {"emb": pitanje_embedding}):
                if z["tekst"] not in vidjena_tekst:
                    vidjena_tekst.add(z["tekst"])
                    pronadjeni_pasusi.append({**dict(z), "tip": "vektor"})

            kljucne_reci = izvuci_kljucne_reci(pitanje_korisnika)
            if kljucne_reci:
                cir_reci = [u_cirilicu(r) for r in kljucne_reci]
                ft_upit = " OR ".join(
                    [re.sub(r'[+\-!(){}\[\]^"~*?:\\/]', ' ', r) for r in cir_reci + kljucne_reci]
                )
                try:
                    for z in session.run("""
                        CALL db.index.fulltext.queryNodes('chunk_fulltext', $upit, {limit: 5})
                        YIELD node, score
                        WHERE size(node.tekst) > 500
                        MATCH (d:Disertacija)-[:IMA_DEO]->(node)
                        RETURN node.tekst AS tekst, node.stranica AS stranica, d.naslov AS naslov, score
                    """, {"upit": ft_upit}):
                        if z["tekst"] not in vidjena_tekst:
                            vidjena_tekst.add(z["tekst"])
                            pronadjeni_pasusi.append({**dict(z), "tip": "keyword"})
                except Exception:
                    pass

    if not naslov_disertacije and (not pronadjeni_pasusi or max(p["score"] for p in pronadjeni_pasusi) < 0.60):
        return "Na osnovu dokumenata u bazi, ne mogu da pronadjem odgovor.", []

    if not pronadjeni_pasusi:
        return f"Rad '{naslov_fragment}' nije pronađen u bazi znanja.", []

    # Filtriraj TOC pasuse (sadrzaj sa mnogo tackica — tipican format sadrzaja)
    pronadjeni_pasusi = [p for p in pronadjeni_pasusi if p["tekst"].count("…") < 5 and p["tekst"].count("....") < 3]

    # Sortiraj po scoru i ograniči na top 3 pasusa koja idu u LLM kontekst
    pronadjeni_pasusi = sorted(pronadjeni_pasusi, key=lambda p: p["score"], reverse=True)[:3]

    # "Prati referencu": ako chunk pominje broj stranice, dovuci i cunkove sa te stranice
    brojevi_stranica = set()
    for pasus in pronadjeni_pasusi:
        pomenute = re.findall(r'\b(?:stranici?|strani?|page|str\.?)\s*(\d{1,4})\b', pasus["tekst"], re.IGNORECASE)
        for br in pomenute:
            brojevi_stranica.add((pasus["naslov"], int(br)))

    if brojevi_stranica:
        vidjena_tekst_ref = set(p["tekst"] for p in pronadjeni_pasusi)
        with driver.session() as session:
            for (naslov_ref, br_str) in brojevi_stranica:
                for zapis in session.run("""
                    MATCH (d:Disertacija {naslov: $naslov})-[:IMA_DEO]->(c:Chunk)
                    WHERE c.stranica = $stranica AND size(c.tekst) > 200
                    RETURN c.tekst AS tekst, c.stranica AS stranica, d.naslov AS naslov, 0.85 AS score
                """, {"naslov": naslov_ref, "stranica": br_str}):
                    if zapis["tekst"] not in vidjena_tekst_ref:
                        vidjena_tekst_ref.add(zapis["tekst"])
                        pronadjeni_pasusi.append({**dict(zapis), "tip": "referenca"})
        # Ponovo sortiraj sa novim pasusima, top 4 (malo vise jer sada imamo i referenciranu stranicu)
        pronadjeni_pasusi = sorted(pronadjeni_pasusi, key=lambda p: p["score"], reverse=True)[:4]

    kontekst = ""
    vidljivi_radovi = {}
    for pasus in pronadjeni_pasusi:
        naslov = pasus['naslov']
        tip = pasus.get('tip', 'vektor')
        if naslov not in vidljivi_radovi:
            vidljivi_radovi[naslov] = {"score": pasus['score'], "tip": tip}
        kontekst += (
            f"\n--- IZ RADA: {naslov} (stranica {pasus['stranica']}) ---\n"
            f"Tekst:\n{pasus['tekst']}\n"
        )

    izvori = []
    for naslov, info in vidljivi_radovi.items():
        if info["tip"] == "graf":
            izvori.append(f"Rad: '{naslov}' (Direktna pretraga po naslovu)")
        elif info["tip"] == "keyword":
            izvori.append(f"Rad: '{naslov}' (Pronađen keyword pretragom)")
        else:
            izvori.append(f"Rad: '{naslov}' (Semantička sličnost: {round(info['score']*100, 1)}%)")

    sistemski_prompt = (
        "Ti si strucni asistent za univerzitetske doktorske disertacije. "
        "Odgovori na korisnikovo pitanje koristeci iskljucivo prilozeni kontekst — "
        "ukljucujuci i naslove disertacija i tekst pasusa, jer naslovi mogu sadrzati kljucne informacije. "
        "Odgovori na srpskom jeziku koristeci iskljucivo standardne srpske reci — nemoj izmisljati reci niti koristiti strane reci kada postoji srpski ekvivalent. "
        "Pisi iskljucivo latinicom, nikada cirilicom. "
        "Budi profesionalan, precizan i koncizan. "
        "Ako kontekst sadrzi delimicno relevantne informacije, upotrebi ih i jasno naznaci iz kog rada i sa koje stranice. "
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
