"""
Casablanca Stock Exchange Sector Classification
================================================
Best-effort ticker -> sector map for CSE-listed companies, used as a
fallback when a portfolio CSV doesn't supply a Category for a given ticker.

Ticker codes below were confirmed against the official instrument dropdown
on https://www.casablanca-bourse.com/market-data/cours (the "Cours" /
historical-quotes lookup tool) - these are the exact codes needed to query
MongoDB (see data_fetcher.py) and the site's own API. Sector labels are a
best-effort classification by company activity; a handful of smaller/less
familiar names are educated guesses - verify before relying on them for
anything beyond a rough sector breakdown, and correct/extend CSE_SECTOR_MAP
as you go.
"""

from typing import Optional

CSE_SECTOR_MAP = {
    # Banks
    'ATW': 'Banks',            # Attijariwafa Bank
    'BCP': 'Banks',            # Banque Centrale Populaire
    'BOA': 'Banks',            # Bank of Africa
    'CIH': 'Banks',            # CIH Bank
    'CDM': 'Banks',            # Credit du Maroc
    'BCI': 'Banks',            # BMCI
    'CFG': 'Banks',            # CFG Bank

    # Insurance
    'WAA': 'Insurance',        # Wafa Assurance
    'SAH': 'Insurance',        # Sanlam Maroc (formerly Saham Assurance)
    'AGM': 'Insurance',        # Agma
    'AFM': 'Insurance',        # Afma
    'ATL': 'Insurance',        # Atlantasanad Assurance

    # Telecommunications
    'IAM': 'Telecommunications',   # Maroc Telecom (Itissalat Al-Maghrib)

    # Real Estate
    'ADH': 'Real Estate',      # Douja Prom Addoha
    'RDS': 'Real Estate',      # Residences Dar Saada
    'ARD': 'Real Estate',      # Aradei Capital
    'IMO': 'Real Estate',      # Immorente Invest

    # Building Materials / Construction
    'LHM': 'Building Materials',   # Holcim Maroc
    'CMA': 'Building Materials',   # Ciments du Maroc
    'SID': 'Building Materials',   # Sonasid (steel/rebar)
    'JET': 'Building Materials',   # Jet Contractors
    'TGC': 'Building Materials',   # TGCC
    'GTM': 'Building Materials',   # SGTM
    'DLM': 'Building Materials',   # Delattre Levivier Maroc
    'STR': 'Building Materials',   # Stroc Industrie

    # Mining
    'MNG': 'Mining',           # Managem
    'CMT': 'Mining',           # Miniere Touissit
    'SMI': 'Mining',           # Societe Metallurgique d'Imiter
    'ZDJ': 'Mining',           # Zellidja

    # Food & Beverage / Agro-industry
    'CSR': 'Food & Beverage',  # Cosumar
    'LES': 'Food & Beverage',  # Lesieur Cristal
    'DRI': 'Food & Beverage',  # Dari Couspate
    'OUL': 'Food & Beverage',  # Oulmes
    'SBM': 'Food & Beverage',  # Societe des Boissons du Maroc
    'UMR': 'Food & Beverage',  # Unimer

    # Distribution / Retail
    'LBV': 'Distribution',     # Label Vie
    'MUT': 'Distribution',     # Mutandis
    'CAP': 'Distribution',     # Cash Plus
    'DYT': 'Distribution',     # Disty Technologies
    'DWY': 'Distribution',     # Disway

    # Automotive
    'ATH': 'Automotive',       # Auto Hall
    'NEJ': 'Automotive',       # Auto Nejma
    'NKL': 'Automotive',       # Ennakl

    # Leasing / Consumer Finance
    'MAB': 'Financial Services',   # Maghrebail
    'MLE': 'Financial Services',   # Maroc Leasing
    'EQD': 'Financial Services',   # Eqdom
    'SLF': 'Financial Services',   # Salafin
    'DIS': 'Financial Services',   # Diac Salaf

    # Healthcare / Pharma
    'AKT': 'Healthcare',       # Akdital
    'PRO': 'Healthcare',       # Promopharm
    'SOT': 'Healthcare',       # Sothema

    # Industrials
    'AFI': 'Industrials',      # Afric Industries
    'ALM': 'Industrials',      # Aluminium du Maroc
    'FBR': 'Industrials',      # Fenie Brossette
    'MOX': 'Industrials',      # Maghreb Oxygene
    'SNP': 'Industrials',      # Snep
    'CMG': 'Industrials',      # CMGP Group
    'MDP': 'Industrials',      # Med Paper
    'SRM': 'Industrials',      # Realisations Mecaniques
    'COL': 'Industrials',      # Colorado
    'CRS': 'Industrials',      # Cartier Saada
    'REB': 'Industrials',      # Rebab Company
    'SNA': 'Industrials',      # Stokvis Nord Afrique

    # Technology
    'HPS': 'Technology',       # HPS
    'IBC': 'Technology',       # IB Maroc.com
    'INV': 'Technology',       # Involys
    'MIC': 'Technology',       # Microdata
    'S2M': 'Technology',       # S.M Monetique
    'M2M': 'Technology',       # M2M Group

    # Hospitality / Tourism
    'RIS': 'Hospitality',      # Risma

    # Holding / Conglomerate
    'ADI': 'Holding',          # Alliances
    'DHO': 'Holding',          # Delta Holding
    'T2S': 'Holding',          # T2S Group Holding

    # Transport / Logistics
    'MSA': 'Transport & Logistics',    # Sodep-Marsa Maroc
    'CTM': 'Transport & Logistics',    # CTM

    # Energy
    'GAZ': 'Energy',           # Afriquia Gaz
    'SAM': 'Energy',           # Samir
    'TMA': 'Energy',           # TotalEnergies Marketing Maroc
    'TQM': 'Utilities',        # Taqa Morocco (power generation)

    # Cash & benchmarks
    'CASH': 'Cash',
    'MASI': 'Index',
    'MSI20': 'Index',
}


def classify_sector(ticker: str, category: Optional[str] = "") -> str:
    """
    Best-effort sector classification for a Casablanca-listed ticker.

    A non-empty `category` (typically supplied by the user in the portfolio
    CSV) always takes priority. Falls back to CSE_SECTOR_MAP, then to
    'Unclassified' if the ticker isn't in the map either.
    """
    category = (category or "").strip()
    if category:
        return category
    return CSE_SECTOR_MAP.get(str(ticker).strip().upper(), "Unclassified")
