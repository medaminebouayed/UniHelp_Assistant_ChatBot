# Questions de test RAG — UniHelp (IIT / ISB)

## Groupe A : Plans d'études (content_type = study_plan)

| # | Question | Attendus |
|---|----------|----------|
| A1 | Quel est le plan d'études de la 1ère année GLSI (Licence Info) à l'IIT ? | Modules S1 : Algèbre 1, Analyse 1, Algorithmique & Prog1, Systèmes d'exploitation 1, Systèmes Logiques & Archi, etc. 30 crédits/semestre |
| A2 | Quels sont les modules de la 3ème année GLSI à l'IIT ? | BD, IA, Cloud & Big Data, Compilation & Test, Projet Génie Logiciel, etc. |
| A3 | Quelles sont les matières du semestre 2 de la 2ème année GLSI ? | Algèbre 2, Analyse 2, Algorithmique & Prog 2, etc. |
| A4 | Donne-moi le plan d'études du Génie Civil à l'IIT | Modules spécifiques au GC (plan-detude-2025-2026-GC) |
| A5 | Quels sont les modules en Génie Industriel à l'IIT ? | Modules de Plan EtudeIndus 1 |
| A6 | Quels sont les modules en Génie Mécanique à l'IIT ? | Modules de Plan EtudeMeca 1 |
| A7 | Quels sont les modules en Génie des Procédés à l'IIT ? | Modules de Plan EtudeProcedes 1 |

## Groupe B : Programmes & Formations (content_type = general)

| # | Question | Attendus |
|---|----------|----------|
| B1 | Quelles sont les filières ingénieur disponibles à l'IIT ? | Génie Informatique, Génie Civil, Génie Industriel, Génie Mécanique, Génie des Procédés |
| B2 | Quelle est la différence entre la licence et le mastère à l'IIT ? | Licence = 3 ans, Mastère = Bac+5, etc. |
| B3 | C'est quoi le cycle préparatoire à l'IIT ? | Accès via concours national ou licence info, 2 ans avant le cycle ingénieur |
| B4 | Quels sont les programmes de licence à l'ISB ? | LBA, LBI, etc. |
| B5 | Quels sont les programmes de master à l'ISB ? | MBA, MSc Finance, MSc Marketing, etc. |
| B6 | C'est quoi le Mastère en Comptabilité à l'ISB ? | Programme, durée, débouchés |

## Groupe C : Partenariats & Mobilité (content_type = mobility)

| # | Question | Attendus |
|---|----------|----------|
| C1 | Quels sont les partenaires internationaux de l'ISB ? | KEDGE, MBS, Westcliff, Wesford, ESC Clermont, EAE, Rennes SB, YSchools, Collège de Paris, etc. |
| C2 | Quels sont les partenaires de l'IIT à l'étranger ? | 3iL, ESIGELEC, CESI, UTT, EPF, etc. |
| C3 | C'est quoi le programme avec 3iL pour les étudiants IIT ? | Double diplôme, échange, alternance, coûts |
| C4 | Comment faire pour une mobilité internationale à l'IIT ? | Démarches, conditions, partenaires |
| C5 | C'est quoi le programme avec Westcliff University ? | Partenariat ISB - Westcliff California, double diplôme |
| C6 | Quels sont les frais du programme MBS pour les étudiants ISB ? | Coûts MSc, Master PGE, BIBA |

## Groupe D : Admission & Inscription (content_type = general)

| # | Question | Attendus |
|---|----------|----------|
| D1 | Comment s'inscrire à l'IIT ? | Procédure d'inscription, documents requis |
| D2 | Quels sont les frais d'inscription à l'IIT ? | Frais nationaux/internationaux |
| D3 | Quelles sont les conditions d'admission à l'IIT ? | Concours, bac, prérequis |
| D4 | Comment faire une pré-inscription à l'ISB ? | pré-inscription2025 |

## Groupe E : Vie étudiante & Services (content_type = general)

| # | Question | Attendus |
|---|----------|----------|
| E1 | Quels sont les services de la cellule d'écoute à l'IIT ? | Soutien psychologique, accompagnement |
| E2 | Comment obtenir une bourse à l'IIT ? | Types de bourses, conditions |
| E3 | Quels sont les clubs et activités à l'IIT ? | Vie estudiantine, clubs |
| E4 | Quels sont les services aux étudiants à l'ISB ? | Espace étudiant, services |

## Groupe F : Accréditations & Qualité (content_type = general)

| # | Question | Attendus |
|---|----------|----------|
| F1 | Est-ce que l'IIT est accrédité ? | ASIIN, Euro-Inf, etc. |
| F2 | Quelle est la politique qualité de l'IIT ? | Démarche qualité, ISO, etc. |

## Groupe G : Questions "piège" (test de filtrage dynamique)

| # | Question | Piège | Comportement attendu |
|---|----------|-------|---------------------|
| G1 | Plan d'études pour génie informatique | Des chunks mobility (3iL, coûts) parlent aussi de "génie informatique" | Doit utiliser UNIQUEMENT les chunks study_plan, ignorer les mobility |
| G2 | Combien coûte le programme 3iL ? | Des chunks study_plan parlent aussi de "coûts" dans leurs sections | Doit utiliser les chunks mobility/fees, pas les study_plan |
| G3 | Quels sont les débouchés après une licence à l'IIT ? | Beaucoup de chunks général sur "formations" et "programmes" | Doit synthétiser les infos des chunks general pertinents |
| G4 | Présente-moi le programme génie informatique à l'IIT | Mélange de study_plan (modules) et general (objectifs, compétences) | Doit combiner les deux types de façon cohérente |

## Groupe H : Questions négatives (hors base)

| # | Question | Attendu |
|---|----------|---------|
| H1 | Quel est le prix d'un sandwich à la cafétéria ? | "Je n'ai pas cette information" |
| H2 | Qui est le fondateur de l'IIT ? | "Je n'ai pas cette information" (si pas dans les données) |
| H3 | Quelle est la date des prochains examens ? | "Je n'ai pas cette information" |
| H4 | Comment postuler à Harvard ? | "Je n'ai pas cette information" |
