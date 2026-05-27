import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

st.set_page_config(page_title="NaRDuS GraphRAG Asistent", page_icon="🎓", layout="centered")

load_dotenv()

@st.cache_resource
def inicijalizuj_veze():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    return client, driver

client, driver = inicijalizuj_veze()

st.title("🎓 NaRDuS - GraphRAG Asistent")
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
            if message.get("izvori"):
                with st.expander("🔍 Pogledaj izvore iz baze"):
                    for izvor in message["izvori"]:
                        st.write(f"📍 {izvor}")
            prikazi_feedback_dugmice(
                message["id"], message["pitanje"], message["content"]
            )


def generisi_odgovor(pitanje_korisnika):
    response_emb = client.embeddings.create(
        input=pitanje_korisnika,
        model="text-embedding-3-small"
    )
    pitanje_embedding = response_emb.data[0].embedding

    upit_za_bazu = """
    CALL db.index.vector.queryNodes('chunk_embeddings', 3, $pitanje_embedding)
    YIELD node, score
    MATCH (d:Disertacija)-[:IMA_DEO]->(node)
    RETURN node.tekst AS tekst, node.stranica AS stranica, d.naslov AS naslov, score
    """

    pronadjeni_pasusi = []
    with driver.session() as session:
        rezultati = session.run(upit_za_bazu, {"pitanje_embedding": pitanje_embedding})
        for zapis in rezultati:
            pronadjeni_pasusi.append(zapis)

    if not pronadjeni_pasusi:
        return "Na osnovu dokumenata u bazi, ne mogu da pronadjem odgovor.", []

    kontekst = ""
    izvori = []
    for i, pasus in enumerate(pronadjeni_pasusi):
        izvori.append(
            f"Rad: '{pasus['naslov']}' | Stranica {pasus['stranica']} "
            f"(Slicnost: {round(pasus['score']*100, 1)}%)"
        )
        kontekst += f"\n--- IZVOR {i+1} ---\n{pasus['tekst']}\n"

    sistemski_prompt = (
        "Ti si strucni asistent za univerzitetske doktorske disertacije. "
        "Odgovori na korisnikovo pitanje iskljucivo koristeci prilozeni tekst iz disertacije. "
        "Odgovori na srpskom jeziku (koristi kvacice u recima), budi profesionalan i precizan. "
        "Ako u tekstu nema odgovora, obavezno reci: "
        "'Na osnovu trenutnih dokumenata u bazi ne mogu da odgovorim na to pitanje.'"
    )
    korisnicki_prompt = f"Tekst iz disertacije:\n{kontekst}\n\nPitanje: {pitanje_korisnika}"

    odgovor_ai = client.chat.completions.create(
        model="gpt-4o-mini",
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
            odgovor, izvori = generisi_odgovor(pitanje)
            poruka_id = str(uuid.uuid4())

            st.markdown(odgovor)

            if izvori:
                with st.expander("🔍 Pogledaj izvore iz baze"):
                    for izvor in izvori:
                        st.write(f"📍 {izvor}")

            prikazi_feedback_dugmice(poruka_id, pitanje, odgovor)

    st.session_state.messages.append({
        "role": "assistant",
        "content": odgovor,
        "id": poruka_id,
        "pitanje": pitanje,
        "izvori": izvori,
    })
