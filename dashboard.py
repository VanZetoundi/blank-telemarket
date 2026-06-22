"""
Dashboard TP — Intelligence Artificielle 2 : Deep Learning
DIPES 2 · 4ème année · 2025-2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, warnings
from pathlib import Path
import pickle
import joblib

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TP Deep Learning · Dashboard",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: #0d0d0d;
    background: #fafaf8;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d0d0d !important;
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #fafaf8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stSidebarNav"] { display: none; }

/* Main bg */
.main .block-container { background: #fafaf8; padding: 2rem 2.5rem; max-width: 1200px; }

/* Label mono */
.label-track {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 8px;
}

/* KPI cards */
.kpi-grid { display: grid; gap: 1px; background: #e4e4e4; border: 1px solid #e4e4e4; margin-bottom: 1.5rem; }
.kpi-card {
    background: #fafaf8;
    padding: 1rem 1.2rem;
}
.kpi-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 500;
    color: #0d0d0d;
    margin-bottom: 2px;
}
.kpi-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Section header */
.section-header {
    border-top: 1px solid #e4e4e4;
    padding-top: 1.5rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.section-title {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #0d0d0d;
}

/* Tag badge */
.badge {
    display: inline-block;
    border: 1px solid #0d0d0d;
    padding: 2px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #0d0d0d;
    margin-right: 4px;
    margin-bottom: 4px;
}
.badge-ok { background: #0d0d0d; color: #fafaf8; }

/* Metric overrides */
[data-testid="stMetric"] {
    background: #fafaf8;
    border: 1px solid #e4e4e4;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    color: #888 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 22px !important;
    color: #0d0d0d !important;
}

/* Buttons */
.stButton>button {
    background: #0d0d0d;
    color: #fafaf8;
    border: none;
    border-radius: 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    padding: 0.5rem 1.2rem;
    letter-spacing: 0.05em;
}
.stButton>button:hover { background: #333; }

/* Divider */
hr { border: none; border-top: 1px solid #e4e4e4; margin: 1.5rem 0; }

/* Tab */
[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #888;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #0d0d0d;
    border-bottom: 2px solid #0d0d0d;
}

/* Dataframe */
.stDataFrame { border: 1px solid #e4e4e4; }

/* Info / Warning */
.stAlert { border-radius: 0 !important; border-left: 2px solid #0d0d0d !important; }

/* Select / Input */
.stSelectbox>div>div, .stNumberInput>div>div>input {
    border-radius: 0 !important;
    border: 1px solid #e4e4e4 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

P1_DATA    = BASE / "partie1/data/bank-additional-full.csv"
P1_FIG     = BASE / "partie1/outputs/figures"
P1_MDL     = BASE / "partie1/outputs/models"
P1_RPT     = BASE / "partie1/outputs/reports"
P2_FIG     = BASE / "partie2/outputs/figures"
P2_MDL     = BASE / "partie2/outputs/models"
P2_RPT     = BASE / "partie2/outputs/reports"
P3_FIG     = BASE / "partie3/outputs/figures"
P3_MDL     = BASE / "partie3/outputs/models"
P3_RPT     = BASE / "partie3/outputs/reports"

# ── Helpers ────────────────────────────────────────────────────────────────────
def  load_pkl(path):
    try:
        return joblib.load(path)

    except Exception:
        try:
            with open(path,"rb") as f:
                return pickle.load(f)

        except Exception as e:
            st.error(
                f"{path.name} impossible à charger : {e}"
            )
            return None

def show_fig(path, caption="", use_container_width=True):
    p = Path(path)
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=use_container_width)
    else:
        st.markdown(f'<div style="border:1px dashed #ccc;padding:1rem;color:#888;font-family:monospace;font-size:11px;">Figure non trouvée : {p.name}</div>', unsafe_allow_html=True)

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def kpi(val, lbl):
    return f'<div class="kpi-card"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>'

def section(title, subtitle=""):
    st.markdown(f"""
    <div class="section-header">
        <div class="label-track">{subtitle}</div>
        <div class="section-title">{title}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◆ TP Deep Learning")
    st.markdown("---")
    page = st.radio(
        "",
        [
            "Accueil",
            "Partie 1 — EDA",
            "Partie 1 — Modèles & CV",
            "Partie 1 — Ensembles & ROC",
            "Partie 2 — Réseaux de Neurones",
            "Partie 3 — Deep Learning Fashion",
            "Inférence",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown('<div style="font-size:10px;color:#666;line-height:1.6">DIPES 2 · S2 2025-2026<br>Stéphane C.K. TÉKOUABOU</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if page == "Accueil":
    st.markdown('<div class="label-track">Intelligence Artificielle 2 · Deep Learning · DIPES 2</div>', unsafe_allow_html=True)
    st.markdown("# TD-TP Réseaux de Neurones & Deep Learning")
    st.markdown('<div style="color:#555;font-size:15px;max-width:620px;line-height:1.7;margin-bottom:2rem">Tableau de bord complet couvrant l\'analyse exploratoire, les modèles classiques, les forêts aléatoires, le boosting, les réseaux de neurones artificiels et le deep learning sur Fashion-MNIST.</div>', unsafe_allow_html=True)

    # KPIs globaux
    df_comp = load_csv(P1_RPT / "comparaison_modeles.csv")
    df_auc  = load_csv(P1_RPT / "auc_results.csv")
    df_top3 = load_csv(P1_RPT / "comparaison_top3.csv")
    df_ann  = load_csv(P2_RPT / "comparaison_ann.csv")
    df_fash = load_csv(P3_RPT / "comparaison_fashion_mnist.csv")

    best_acc = "—"
    if df_comp is not None:
        acc_col = [c for c in df_comp.columns if "acc" in c.lower() or "roc" in c.lower() or "auc" in c.lower()]
        if acc_col:
            best_acc = f"{df_comp[acc_col[0]].max():.3f}"

    best_auc = "—"
    if df_auc is not None:
        auc_col = [c for c in df_auc.columns if "auc" in c.lower()]
        if auc_col:
            best_auc = f"{df_auc[auc_col[0]].max():.4f}"

    fashion_acc = "—"
    if df_fash is not None:
        acc_col = [c for c in df_fash.columns if "acc" in c.lower() or "test" in c.lower()]
        if acc_col:
            fashion_acc = f"{df_fash[acc_col[0]].max():.3f}"

    n_models = len(list(P1_MDL.glob("*.pkl"))) if P1_MDL.exists() else 0

    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns:repeat(4,1fr)">
        {kpi(n_models, "Modèles entraînés")}
        {kpi(best_acc, "Meilleur score P1")}
        {kpi(best_auc, "Meilleur AUC P1")}
        {kpi(fashion_acc, "Meilleur acc Fashion")}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="label-track">Partie 1</div>', unsafe_allow_html=True)
        st.markdown("**Télémarketing Bancaire**")
        st.markdown('<div style="font-size:12px;color:#555;line-height:1.7">EDA · Prétraitement · SVM · KNN · DT · NB · LR · ANN · Bagging · RF · Boosting · ROC</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="label-track">Partie 2</div>', unsafe_allow_html=True)
        st.markdown("**Réseaux de Neurones ANN**")
        st.markdown('<div style="font-size:12px;color:#555;line-height:1.7">TensorFlow / Keras · Architecture Sequential · Courbes d\'apprentissage · Déploiement</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="label-track">Partie 3</div>', unsafe_allow_html=True)
        st.markdown("**Deep Learning Fashion-MNIST**")
        st.markdown('<div style="font-size:12px;color:#555;line-height:1.7">LeNet · VGG · ResNet · DenseNet · NasNet · Comparaison architectures</div>', unsafe_allow_html=True)

    st.markdown("---")
    section("Structure du projet", "Architecture")
    st.code("""
partie1/
  data/              bank-additional-full.csv
  outputs/
    figures/         01→18  visualisations EDA & modèles
    models/          12 classifieurs .pkl + scaler
    reports/         CSV résultats & métriques

partie2/
  outputs/
    figures/         courbes apprentissage ANN
    models/          best_ann.keras · telemarketing.pkl
    reports/         comparaison_ann.csv

partie3/
  outputs/
    figures/         Fashion-MNIST exemples & résultats
    models/          best_fashion.keras · bank-tel.pkl
    reports/         comparaison_fashion_mnist.csv
    """, language="")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Partie 1 — EDA":
    st.markdown('<div class="label-track">Partie 1 · Analyse Exploratoire</div>', unsafe_allow_html=True)
    st.markdown("# Analyse des Données")

    df = None
    if P1_DATA.exists():
        try:
            df = pd.read_csv(P1_DATA, sep=";")
        except Exception:
            df = None

    if df is not None:
        # Dataset stats
        n_cls   = df["y"].nunique() if "y" in df.columns else "—"
        n_feat  = df.shape[1] - 1
        n_inst  = df.shape[0]

        st.markdown(f"""
        <div class="kpi-grid" style="grid-template-columns:repeat(4,1fr)">
            {kpi(n_inst, "Instances")}
            {kpi(n_feat, "Features")}
            {kpi(n_cls, "Classes")}
            {kpi("Binaire", "Type apprentissage")}
        </div>
        """, unsafe_allow_html=True)

        tabs = st.tabs(["Aperçu", "Statistiques", "Distribution", "Corrélations", "Variables catégorielles", "Scatter Matrix"])

        with tabs[0]:
            st.markdown('<div class="label-track">Premières lignes</div>', unsafe_allow_html=True)
            st.dataframe(df.head(20), use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Types de colonnes**")
                dtype_df = pd.DataFrame({"Type": df.dtypes, "Non-nuls": df.notna().sum(), "Nuls": df.isna().sum()})
                st.dataframe(dtype_df, use_container_width=True)
            with col2:
                if "y" in df.columns:
                    vc = df["y"].value_counts()
                    st.markdown("**Distribution des classes**")
                    ratio = (vc.min() / vc.max() * 100)
                    st.dataframe(vc.rename("count").to_frame(), use_container_width=True)
                    st.markdown(f'<div class="badge">Ratio min/max : {ratio:.1f}%</div> <div class="badge">Déséquilibre de classes</div>', unsafe_allow_html=True)

        with tabs[1]:
            st.markdown('<div class="label-track">describe()</div>', unsafe_allow_html=True)
            st.dataframe(df.describe().T, use_container_width=True)

        with tabs[2]:
            col1, col2 = st.columns(2)
            with col1:
                show_fig(P1_FIG / "01_distribution_target.png", "Distribution de la variable cible")
            with col2:
                show_fig(P1_FIG / "02_distributions_num.png", "Distributions des variables numériques")

        with tabs[3]:
            show_fig(P1_FIG / "03_matrice_correlations.png", "Matrice de corrélations")
            st.markdown("""
            **Lecture :**
            - Les corrélations intra-variables numériques permettent d'identifier les redondances.
            - `emp.var.rate`, `euribor3m`, `nr.employed` sont fortement corrélées entre elles (contexte économique).
            - Peu de variables corrélées fortement avec la cible `y` → problème de classification non-linéaire.
            """)

        with tabs[4]:
            show_fig(P1_FIG / "04_cat_vs_target.png", "Variables catégorielles vs cible")

        with tabs[5]:
            col1, col2 = st.columns(2)
            with col1:
                show_fig(P1_FIG / "05_scatter_matrix.png", "Scatter matrix")
            with col2:
                show_fig(P1_FIG / "06_scatter_regression.png", "Droites de régression")

    else:
        st.info("Données non trouvées. Placez `bank-additional-full.csv` dans `partie1/data/`.")
        # Afficher les figures existantes quand même
        section("Visualisations disponibles", "Figures générées")
        cols = st.columns(2)
        figs = sorted(P1_FIG.glob("0*.png")) if P1_FIG.exists() else []
        for i, fig in enumerate(figs[:6]):
            with cols[i % 2]:
                show_fig(fig, fig.stem)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — Modèles & CV
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Partie 1 — Modèles & CV":
    st.markdown('<div class="label-track">Partie 1 · Modèles Classiques & Validation Croisée</div>', unsafe_allow_html=True)
    st.markdown("# Modèles & Validation Croisée")

    tabs = st.tabs(["Comparaison générale", "Arbre de Décision", "Résultats numériques"])

    with tabs[0]:
        show_fig(P1_FIG / "07_comparaison_modeles.png", "Comparaison des modèles (validation croisée)")
        show_fig(P1_FIG / "08_confusion_best_model.png", "Matrice de confusion — meilleur modèle")

        df_comp = load_csv(P1_RPT / "comparaison_modeles.csv")
        if df_comp is not None:
            section("Tableau des performances", "Métriques")
            # Highlight best per column
            st.dataframe(df_comp, use_container_width=True)

            # Best model badge
            num_cols = df_comp.select_dtypes(include="number").columns
            if len(num_cols):
                best_col = num_cols[0]
                best_row = df_comp[best_col].idxmax()
                best_name = df_comp.iloc[best_row, 0] if df_comp.columns[0] != best_col else "—"
                st.markdown(f'<div class="badge badge-ok">Meilleur modèle : {best_name}</div>', unsafe_allow_html=True)

    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            show_fig(P1_FIG / "09_arbre_decision_simple.png", "Arbre de décision simple")
        with col2:
            show_fig(P1_FIG / "11_arbre_optimise.png", "Arbre optimisé (GridSearchCV)")
        show_fig(P1_FIG / "10_depth_vs_auc.png", "AUC vs profondeur de l'arbre")

        st.markdown("""
        **Observations :**
        - Un arbre trop peu profond sous-apprend (biais élevé).
        - Un arbre trop profond sur-apprend (variance élevée).
        - La validation croisée identifie le `max_depth` optimal qui maximise l'AUC de test.
        """)

    with tabs[2]:
        df_top3 = load_csv(P1_RPT / "comparaison_top3.csv")
        if df_top3 is not None:
            st.dataframe(df_top3, use_container_width=True)
        show_fig(P1_FIG / "15_comparaison_top3.png", "Comparaison Top 3 classifieurs optimisés")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — Ensembles & ROC
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Partie 1 — Ensembles & ROC":
    st.markdown('<div class="label-track">Partie 1 · Méthodes d\'Ensemble & Évaluation</div>', unsafe_allow_html=True)
    st.markdown("# Méthodes d'Ensemble & Courbes ROC")

    tabs = st.tabs(["Bagging & Random Forest", "Boosting", "Importance des variables", "Courbes ROC"])

    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            show_fig(P1_FIG / "12_bagging_vs_B.png", "Performance Bagging vs nombre d'arbres B")
        with col2:
            show_fig(P1_FIG / "13_rf_gridsearch.png", "Random Forest — GridSearch sur p")
        st.markdown("""
        **Bagging :** `RandomForestClassifier(max_features=None)` — agrège B arbres complets.  
        **Random Forest :** sous-sélection aléatoire de `p` features à chaque nœud → décorrélation des arbres.  
        L'erreur OOB (*Out-Of-Bag*) est une estimation gratuite de l'erreur de généralisation.
        """)

    with tabs[1]:
        show_fig(P1_FIG / "14_gb_early_stopping.png", "Gradient Boosting — Early Stopping")
        st.markdown("""
        **Paramètres cruciaux GradientBoosting :**
        - `n_estimators` (B) : nombre d'arbres — sélectionné par early stopping
        - `learning_rate` (λ) : shrinkage — compromis biais/variance
        - `max_depth` (p) : complexité des arbres de base
        - `subsample` : fraction d'échantillons par arbre (Stochastic GB)
        
        **AdaBoost ≈ GradientBoosting avec** `loss='exponential'`, `max_depth=1`, `learning_rate=1`.
        """)

    with tabs[2]:
        show_fig(P1_FIG / "16_feature_importances.png", "Importance des variables — Top 3 modèles")
        df_fi = load_csv(P1_RPT / "feature_importances.csv")
        if df_fi is not None:
            st.dataframe(df_fi.head(20), use_container_width=True)

        feat_names = load_pkl(P1_MDL / "feature_names.pkl")
        top10 = load_pkl(P1_MDL / "top10_features.pkl")
        if top10 is not None:
            st.markdown('<div class="label-track">Top 10 features retenues pour le déploiement</div>', unsafe_allow_html=True)
            badges = " ".join([f'<span class="badge badge-ok">{f}</span>' for f in top10])
            st.markdown(badges, unsafe_allow_html=True)

    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            show_fig(P1_FIG / "17_roc_curves.png", "Courbes ROC — 3 classifieurs optimisés")
        with col2:
            show_fig(P1_FIG / "18_roc_zoom.png", "Zoom ROC (région haute performance)")

        df_auc = load_csv(P1_RPT / "auc_results.csv")
        if df_auc is not None:
            st.markdown('<div class="label-track">AUC par modèle</div>', unsafe_allow_html=True)
            st.dataframe(df_auc, use_container_width=True)
            auc_cols = [c for c in df_auc.columns if "auc" in c.lower()]
            if auc_cols:
                best_idx = df_auc[auc_cols[0]].idxmax()
                st.markdown(f'<div class="badge badge-ok">Meilleur AUC : {df_auc.iloc[best_idx, 0]} — {df_auc[auc_cols[0]].max():.4f}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — ANN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Partie 2 — Réseaux de Neurones":
    st.markdown('<div class="label-track">Partie 2 · Réseaux de Neurones Artificiels</div>', unsafe_allow_html=True)
    st.markdown("# Réseaux de Neurones — Télémarketing")

    tabs = st.tabs(["Architecture", "Entraînement", "Résultats", "Comparaison"])

    with tabs[0]:
        section("Architecture Sequential Keras", "TensorFlow / Keras")
        st.code("""
model = Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')   # sortie binaire
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)
        """, language="python")

        st.markdown("""
        **Choix de conception :**
        - Couches `Dense` + `ReLU` → approximation universelle
        - `BatchNormalization` → stabilise l'entraînement, accélère la convergence
        - `Dropout(0.3)` → régularisation contre le surapprentissage
        - Sortie `sigmoid` → probabilité de souscription ∈ [0,1]
        """)

    with tabs[1]:
        show_fig(P2_FIG / "19_ann_learning_curves.png", "Courbes d'apprentissage — loss & accuracy")
        st.markdown("""
        **Lecture des courbes :**
        - Convergence train/val → bon généralisation
        - Divergence croissante → surapprentissage (augmenter le dropout ou réduire la capacité)
        - Plateau → learning rate trop faible ou architecture insuffisante
        """)

    with tabs[2]:
        show_fig(P2_FIG / "20_ann_confusion_best.png", "Matrice de confusion — meilleur ANN")
        df_ann = load_csv(P2_RPT / "comparaison_ann.csv")
        if df_ann is not None:
            st.dataframe(df_ann, use_container_width=True)

    with tabs[3]:
        section("ANN vs Modèles Classiques", "Comparaison inter-parties")
        df_comp = load_csv(P1_RPT / "comparaison_modeles.csv")
        df_ann  = load_csv(P2_RPT / "comparaison_ann.csv")
        if df_comp is not None and df_ann is not None:
            st.markdown("Fusion des résultats Partie 1 & 2 pour comparaison directe.")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Partie 1 — Modèles sklearn**")
                st.dataframe(df_comp, use_container_width=True)
            with col2:
                st.markdown("**Partie 2 — ANN Keras**")
                st.dataframe(df_ann, use_container_width=True)
        elif df_ann is not None:
            st.dataframe(df_ann, use_container_width=True)

        # Model files
        section("Modèles sauvegardés", "Fichiers .keras / .pkl")
        models_p2 = list(P2_MDL.glob("*")) if P2_MDL.exists() else []
        for m in models_p2:
            st.markdown(f'<span class="badge">{m.name}</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — DEEP LEARNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Partie 3 — Deep Learning Fashion":
    st.markdown('<div class="label-track">Partie 3 · Deep Learning · Fashion-MNIST</div>', unsafe_allow_html=True)
    st.markdown("# Deep Learning — Fashion-MNIST")

    st.markdown("""
    <div style="background:#f5f5f3;border:1px solid #e4e4e4;padding:1rem 1.2rem;margin-bottom:1.5rem">
    <div class="label-track">Dataset</div>
    <b>Fashion-MNIST</b> · 70 000 images 28×28 niveaux de gris · 10 classes de vêtements<br>
    <span style="font-family:monospace;font-size:11px;color:#555">Train: 60 000 · Test: 10 000 · Classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot</span>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["Données", "Entraînement", "Comparaison architectures", "Prédictions"])

    with tabs[0]:
        show_fig(P3_FIG / "21_fashion_exemples.png", "Exemples d'images Fashion-MNIST par classe")

    with tabs[1]:
        show_fig(P3_FIG / "22_fashion_learning_curves.png", "Courbes d'apprentissage — meilleur modèle")
        section("Architectures testées", "Deep Learning")
        archs = {
            "LeNet-5":   "Conv→Pool→Conv→Pool→FC→FC → pionnier, 60K params",
            "VGGNet":    "Blocs Conv 3×3 empilés → profondeur, ~14M params",
            "AlexNet":   "Conv large→Pool→BN→FC → ImageNet 2012",
            "ResNet":    "Skip connections → réseaux très profonds sans dégradation",
            "DenseNet":  "Connexions denses → réutilisation maximale des features",
            "NasNet":    "Architecture cherchée par NAS → état de l'art efficace",
        }
        for name, desc in archs.items():
            st.markdown(f'<span class="badge badge-ok">{name}</span> <span style="font-size:12px;color:#555">{desc}</span><br>', unsafe_allow_html=True)

    with tabs[2]:
        show_fig(P3_FIG / "23_fashion_comparaison.png", "Comparaison des architectures")
        df_fash = load_csv(P3_RPT / "comparaison_fashion_mnist.csv")
        if df_fash is not None:
            st.dataframe(df_fash, use_container_width=True)
            num_cols = df_fash.select_dtypes(include="number").columns
            if len(num_cols):
                best_idx = df_fash[num_cols[0]].idxmax()
                best_name = df_fash.iloc[best_idx, 0]
                st.markdown(f'<div class="badge badge-ok">Meilleure architecture : {best_name}</div>', unsafe_allow_html=True)

    with tabs[3]:
        show_fig(P3_FIG / "24_fashion_predictions.png", "Prédictions du meilleur modèle")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE — INFÉRENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Inférence":
    st.markdown('<div class="label-track">Déploiement · Inférence Temps Réel</div>', unsafe_allow_html=True)
    st.markdown("# Inférence & Prédiction")

    # ── Choisir le modèle ──────────────────────────────────────────────────
    model_options = {
        "Random Forest optimisé (P1)":      P1_MDL / "best_rf.pkl",
        "Gradient Boosting optimisé (P1)":  P1_MDL / "best_gb.pkl",
        "CART optimisé (P1)":               P1_MDL / "best_cart.pkl",
        "Logistic Regression (P1)":         P1_MDL / "Logistic_Reg.pkl",
        "SVM RBF (P1)":                     P1_MDL / "SVM_RBF.pkl",
        "KNN k=5 (P1)":                     P1_MDL / "KNN_k=5.pkl",
        "MLP ANN sklearn (P1)":             P1_MDL / "MLP_ANN.pkl",
        "ANN Keras Télémarketing (P2)":     P2_MDL / "telemarketing.pkl",
    }

    available = {k: v for k, v in model_options.items() if v.exists()}

    if not available:
        st.warning("Aucun modèle .pkl trouvé. Vérifiez que les dossiers `outputs/models/` sont bien présents.")
        st.stop()

    col_sel, col_info = st.columns([2, 1])
    with col_sel:
        selected_name = st.selectbox("Modèle", list(available.keys()))
    with col_info:
        st.markdown('<div class="label-track">Fichier</div>', unsafe_allow_html=True)
        st.markdown(f'<code style="font-size:11px">{available[selected_name].name}</code>', unsafe_allow_html=True)

    model = load_pkl(available[selected_name])
    scaler = load_pkl(P1_MDL / "scaler.pkl")
    feat_names = load_pkl(P1_MDL / "feature_names.pkl")
    top10 = load_pkl(P1_MDL / "top10_features.pkl")

    if model is None:
        st.error(f"Impossible de charger le modèle : {available[selected_name]}")
        st.stop()

    st.markdown("---")

    # ── Mode d'entrée ──────────────────────────────────────────────────────
    mode = st.radio("Mode d'entrée", ["Formulaire manuel", "Upload CSV"], horizontal=True)

    if mode == "Formulaire manuel":
        section("Paramètres du prospect", "Campagne télémarketing")

        # Charger les données pour connaître les plages
        df_ref = None
        if P1_DATA.exists():
            try:
                df_ref = pd.read_csv(P1_DATA, sep=";")
            except Exception:
                pass

        with st.form("inference_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                age         = st.number_input("Age", 18, 95, 40)
                job         = st.selectbox("Job", ["admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown"])
                marital     = st.selectbox("Statut marital", ["divorced","married","single","unknown"])
                education   = st.selectbox("Éducation", ["basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown"])

            with col2:
                default     = st.selectbox("Défaut de crédit", ["no","yes","unknown"])
                housing     = st.selectbox("Prêt immobilier", ["no","yes","unknown"])
                loan        = st.selectbox("Prêt personnel", ["no","yes","unknown"])
                contact     = st.selectbox("Type de contact", ["cellular","telephone"])
                month       = st.selectbox("Mois", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])

            with col3:
                day_of_week = st.selectbox("Jour de la semaine", ["mon","tue","wed","thu","fri"])
                duration    = st.number_input("Durée appel (s)", 0, 5000, 200)
                campaign    = st.number_input("Nb contacts campagne", 1, 50, 2)
                pdays       = st.number_input("Jours depuis dernier contact", -1, 999, 999)
                previous    = st.number_input("Contacts précédents", 0, 20, 0)
                poutcome    = st.selectbox("Résultat précédent", ["failure","nonexistent","success"])

            st.markdown("**Indicateurs économiques**")
            c1, c2, c3, c4, c5 = st.columns(5)
            emp_var_rate  = c1.number_input("emp.var.rate",  -5.0, 5.0,  1.1, 0.1)
            cons_price_idx= c2.number_input("cons.price.idx",90.0,100.0,93.2, 0.1)
            cons_conf_idx = c3.number_input("cons.conf.idx",-55.0,-20.0,-42.7,0.1)
            euribor3m     = c4.number_input("euribor3m",    0.0,  6.0,   4.9,  0.1)
            nr_employed   = c5.number_input("nr.employed",  4900.,5300.,5191.,1.0)

            submitted = st.form_submit_button("◆ Prédire")

        if submitted:
            # Construire le vecteur de features brut (même pipeline que preprocessing)
            row = {
                "age": age, "job": job, "marital": marital, "education": education,
                "default": default, "housing": housing, "loan": loan, "contact": contact,
                "month": month, "day_of_week": day_of_week, "duration": duration,
                "campaign": campaign, "pdays": pdays, "previous": previous,
                "poutcome": poutcome, "emp.var.rate": emp_var_rate,
                "cons.price.idx": cons_price_idx, "cons.conf.idx": cons_conf_idx,
                "euribor3m": euribor3m, "nr.employed": nr_employed,
            }
            input_df = pd.DataFrame([row])

            # Encoder les catégorielles
            cat_cols = input_df.select_dtypes(include="object").columns.tolist()
            input_enc = pd.get_dummies(input_df, columns=cat_cols)

            # Aligner avec les features attendues
            if feat_names is not None:
                missing = set(feat_names) - set(input_enc.columns)
                for m in missing:
                    input_enc[m] = 0
                input_enc = input_enc.reindex(columns=feat_names, fill_value=0)
            elif top10 is not None:
                # Si on n'a que top10, tenter de filtrer
                common = [f for f in top10 if f in input_enc.columns]
                if common:
                    input_enc = input_enc[common]

            # Scaler si disponible
            if scaler is not None:
                try:
                    input_scaled = scaler.transform(input_enc)
                except Exception:
                    input_scaled = input_enc.values
            else:
                input_scaled = input_enc.values

            # Prédiction
            try:
                pred = model.predict(input_scaled)[0]
                prob = None
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(input_scaled)[0]

                st.markdown("---")
                res_col1, res_col2, res_col3 = st.columns(3)

                label = "✓ Souscription probable" if pred == 1 or pred == "yes" else "✗ Pas de souscription"
                color = "#0d0d0d" if pred == 1 or pred == "yes" else "#666"

                with res_col1:
                    st.markdown(f"""
                    <div style="border:2px solid {color};padding:1.2rem;text-align:center">
                        <div class="kpi-val" style="color:{color};font-size:18px">{label}</div>
                        <div class="kpi-lbl">Résultat</div>
                    </div>
                    """, unsafe_allow_html=True)
                with res_col2:
                    if prob is not None and len(prob) > 1:
                        st.metric("P(souscription)", f"{prob[1]:.1%}")
                    else:
                        st.metric("Prédiction brute", str(pred))
                with res_col3:
                    st.metric("Modèle", selected_name.split(" (")[0])

                if prob is not None and len(prob) > 1:
                    st.progress(float(prob[1]))
                    st.markdown(f'<div style="font-family:monospace;font-size:11px;color:#888">Score de confiance : {prob[1]:.4f}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
                st.markdown("*Vérifiez la compatibilité du scaler et des noms de features avec le modèle chargé.*")

    else:  # Upload CSV
        section("Prédiction par lot (CSV)", "Batch inference")
        uploaded = st.file_uploader("Charger un fichier CSV", type=["csv"])

        if uploaded:
            try:
                df_up = pd.read_csv(uploaded, sep=None, engine="python")
                st.markdown(f"**{len(df_up)} lignes chargées**")
                st.dataframe(df_up.head(), use_container_width=True)

                cat_cols = df_up.select_dtypes(include="object").columns.tolist()
                # retirer la cible si présente
                if "y" in cat_cols:
                    cat_cols.remove("y")
                    df_feat = df_up.drop(columns=["y"])
                else:
                    df_feat = df_up.copy()

                df_enc = pd.get_dummies(df_feat, columns=[c for c in cat_cols if c in df_feat.columns])

                if feat_names is not None:
                    missing = set(feat_names) - set(df_enc.columns)
                    for m in missing:
                        df_enc[m] = 0
                    df_enc = df_enc.reindex(columns=feat_names, fill_value=0)

                if scaler is not None:
                    try:
                        X = scaler.transform(df_enc)
                    except Exception:
                        X = df_enc.values
                else:
                    X = df_enc.values

                preds = model.predict(X)
                df_result = df_up.copy()
                df_result["prediction"] = preds

                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(X)
                    if probas.shape[1] > 1:
                        df_result["score_souscription"] = probas[:, 1]

                st.markdown("---")
                st.markdown('<div class="label-track">Résultats</div>', unsafe_allow_html=True)
                st.dataframe(df_result, use_container_width=True)

                csv_out = df_result.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Télécharger les prédictions", csv_out, "predictions.csv", "text/csv")

            except Exception as e:
                st.error(f"Erreur : {e}")

    # ── Infos modèle ──────────────────────────────────────────────────────
    section("Informations sur le modèle chargé", "Méta-données")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<span class="badge badge-ok">{type(model).__name__}</span>', unsafe_allow_html=True)
        if hasattr(model, "n_estimators"):
            st.markdown(f"**n_estimators :** {model.n_estimators}")
        if hasattr(model, "max_depth"):
            st.markdown(f"**max_depth :** {model.max_depth}")
        if hasattr(model, "best_params_"):
            st.markdown(f"**best_params :** `{model.best_params_}`")
    with col2:
        if feat_names is not None:
            st.markdown(f"**Features attendues :** {len(feat_names)}")
        if top10 is not None:
            st.markdown(f"**Top-10 features :** {', '.join(top10[:5])}…")