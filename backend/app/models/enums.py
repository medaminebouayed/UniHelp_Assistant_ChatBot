import enum

class SourceEnum(str, enum.Enum):
    reseaux_sociaux = "Réseaux sociaux"
    amis_parents = "Amis/Parents"
    site_web = "Site Web"
    sms = "SMS"
    press = "Press"
    foire = "Foire"
    tourne_lycee = "Tourné Lycée"
    evenement = "Evenement"
    email = "Email"
    radio_tv = "Radio/TV"

class NiveauEnum(str, enum.Enum):
    niveau_1 = "1"
    niveau_2 = "2"
    niveau_3 = "3"

class DiplomeEnum(str, enum.Enum):
    baccalaureat = "Baccalauréat"
    prepa = "Prépa"
    licence = "Licence"
    master = "Master"

class SexeEnum(str, enum.Enum):
    homme = "Homme"
    femme = "Femme"

class UniversiteEnum(str, enum.Enum):
    iit = "IIT"
    isb = "ISB"

class DepartementEnum(str, enum.Enum):
    informatique = "Informatique"
    genie = "Génie"
    finance = "Finance"

class CategorieEnum(str, enum.Enum):
    dossier = "Dossier"
    attestation = "Attestation"
    diplome = "Diplome"
    autre = "Autre"

class DiplomeSouhaiteEnum(str, enum.Enum):
    licence_business_eng = "Licence en Business Administration(ENG)"
    licence_bi = "Licence en Business Intelligence(Bilingue)"
    licence_gestion = "Licence Nationale en Gestion(FR)"
    licence_comptabilite = "Licence Nationale en Comptabilité(FR)"
    licence_droit_prive = "Licence Nationale en Droit Privé(FR)"
    licence_droit_affaires = "Licence Nationale en Droit des Affaires(FR)"
    licence_eco_business = "Licence en science économique : Business Economics (with AI)(FR)"
    licence_eco_transport = "Licence en science économique : Transport et Logistique(FR)"
    master_mba = "Master in Business Administration (MBA)(ENG)"
    master_marketing = "Mastère en Marketing Digital(FR)"
    master_comptabilite = "Master en Comptabilité(FR)"
    master_finance = "Mastère Finance d'Entreprise et Ingénierie Financière(FR)"
    msc_digital = "Master of Science (MSc) in Digital Marketing(ENG)"
    msc_finance = "Master of Science (MSc) in Finance(ENG)"
    master_droit = "Master en Droit(FR)"
    cycle_prep = "Cycle Préparatoire"
    licence_info = "Licence en Informatique : Génie Logiciel et Système d'Information"
    licence_genie = "Licence en Génie Industriel : Management des Systèmes Industriels"
    master_industrie = "Master Industrie 4.0"
    genie_civil = "Génie Civil"
    genie_procedes = "Génie des Procédés"
    genie_mecanique = "Génie Mécanique"
    genie_telecom = "Génie des Télécommunications"
    genie_info = "Génie Informatique"
    genie_industriel = "Génie Industriel"
    architecture = "Architecture"
