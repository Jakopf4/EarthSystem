# Berechnung der In- oder Out-Degrees (monthly und yearly) für die Knoten im Netzwerk aus Pre-Processed Data
# Berechnung der Feed-Forward-Loops (monthly und yearly) für die Knoten im Netzwerk aus Pre-Processed Data
# Berechnung der Clustering Coefficients (monthly und yearly) für die Knoten im Netzwerk aus Pre-Processed Data
# Berechnung der MAP (yearly) für die Knoten im Netzwerk aus Pre-Processed Data


# Speichern der Daten in /processing/processed_data/ (als .csv)

# Variabilität der Thresholds gewährleisten

#processed-data/
#    year2030/month01/
#        threshold_1.14/
#            clustering
#            ffl
#            in-degrees
#        threshold_1.5/
#            ...
#        threshold_1/
#            ...

import xarray as xr
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path


def count_ffls_per_node(G):
    """Zählt, wie oft jeder Knoten der Ursprung (Source) eines Feed-Forward-Loops ist."""
    ffl_counts = {n: 0 for n in G.nodes()}
    for A in G.nodes():
        # Alle direkten Nachbarn von A
        successors_A = set(G.successors(A))
        for B in successors_A:
            # Alle direkten Nachbarn von B
            successors_B = set(G.successors(B))
            # Überschneidung: Knoten C wird sowohl von A als auch von B erreicht
            common_C = successors_A.intersection(successors_B)
            # Für jedes gefundene C einen FFL zu A hinzufügen
            ffl_counts[A] += len(common_C)
    return ffl_counts


def calculate_network_metrics(scenario: int, year: int, month: int, thresholds: list,
                              def_option: str = "Raw"):
    """Berechnet Metriken (inkl. FFLs) für alle Thresholds und speichert sie in EINER Master-CSV."""

    if def_option == "Raw":
        nc_filepath = f"../1-data/water/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    elif def_option == "BaU":
        nc_filepath = f"../2-pre-processing/pre-processed-data/water_BaU/scenario_ssp{scenario}_decade{year}_month{month:02d}_BaU.nc"
    elif def_option == "Gov":
        nc_filepath = f"../2-pre-processing/pre-processed-data/water_Gov/scenario_ssp{scenario}_decade{year}_month{month:02d}_Gov.nc"
    elif def_option == "Sec":
        nc_filepath = f"../2-pre-processing/pre-processed-data/water_Sec/scenario_ssp{scenario}_decade{year}_month{month:02d}_Sec.nc"

    ds = xr.open_dataset(nc_filepath)
    matrix = ds["network"].values

    lat_values = ds["lat"].values
    lon_values = ds["lon"].values

    prec_values = ds["prec"].values
    evap_values = ds["evap"].values

    # Eine leere Liste, um die Teilergebnisse zu sammeln
    all_threshold_data = []

    for threshold in thresholds:
        print(f"Berechne Threshold {threshold}...")
        binary_matrix = (matrix >= threshold).astype(int)
        G = nx.from_numpy_array(binary_matrix, create_using=nx.DiGraph)

        # Metriken berechnen
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        clustering_coeffs = nx.clustering(G)

        # --- NEU: Feed-Forward-Loops berechnen ---
        ffl_counts = count_ffls_per_node(G)

        # DataFrame für DIESEN Threshold erstellen
        df_metrics = pd.DataFrame({
            "lat": lat_values,
            "lon": lon_values,
            "threshold": threshold,
            "prec": prec_values,
            "evap": evap_values,
            "in_degree": pd.Series(in_degrees),
            "out_degree": pd.Series(out_degrees),
            "clustering": pd.Series(clustering_coeffs),
            "ffl": pd.Series(ffl_counts)
        })

        # DataFrame in unsere Liste packen, NICHT speichern!
        all_threshold_data.append(df_metrics)

    # --- SCHLEIFE BEENDET ---

    # Alle DataFrames in der Liste zu einem riesigen DataFrame zusammenkleben
    master_df = pd.concat(all_threshold_data)

    # Ordnerstruktur nur bis zum Monat erstellen (wir brauchen keine Threshold-Ordner mehr)
    if def_option == "Raw":
        output_filepath = f"./processed-data/metrics_Raw/network_metrics_ssp{scenario}_decade{year}_month{month:02d}_Raw.csv"
    elif def_option == "BaU":
        output_filepath = f"./processed-data/metrics_BaU/network_metrics_ssp{scenario}_decade{year}_month{month:02d}_BaU.csv"
    elif def_option == "Gov":
        output_filepath = f"./processed-data/metrics_Gov/network_metrics_ssp{scenario}_decade{year}_month{month:02d}_Gov.csv"
    elif def_option == "Sec":
        output_filepath = f"./processed-data/metrics_Sec/network_metrics_ssp{scenario}_decade{year}_month{month:02d}_Sec.csv"

    # Datei schreiben
    master_df.to_csv(output_filepath, index_label="node_id")

    ds.close()
    print(f"Master-CSV für Jahr {year}, Monat {month} erfolgreich gespeichert!")


def calculate_yearly_network_metrics(scenario: int, year: int, thresholds: list, def_option: str = "Raw"):
    """Liest die 12 monatlichen Master-CSVs ein, berechnet den Jahresdurchschnitt 
    der Metriken und speichert das Ergebnis in einer neuen CSV-Datei.
    """

    # Eine leere Liste, um die DataFrames der 12 Monate zu sammeln
    all_months_data = []

    # 1. Schleife über die 12 Monate, um die bereits berechneten CSVs einzulesen
    for month in range(1, 13):
        if def_option == "Raw":
            input_filepath = f"./processed-data/metrics_Raw/network_metrics_ssp{scenario}_decade{year}_month{month:02d}_Raw.csv"
        elif def_option == "BaU":
            input_filepath = f"./processed-data/metrics_BaU/network_metrics_ssp{scenario}_decade{year}_month{month:02d}_BaU.csv"
        elif def_option == "Gov":
            input_filepath = f"./processed-data/metrics_Gov/network_metrics_ssp{scenario}_decade{year}_month{month:02d}_Gov.csv"
        elif def_option == "Sec":
            input_filepath = f"./processed-data/metrics_Sec/network_metrics_ssp{scenario}_decade{year}_month{month:02d}_Sec.csv"

        try:
            # Monats-CSV einlesen und an die Liste anhängen
            df_month = pd.read_csv(input_filepath)
            all_months_data.append(df_month)
        except FileNotFoundError:
            print(f"Warnung: Datei {input_filepath} nicht gefunden. Bitte stelle sicher, dass die monatlichen Daten generiert wurden.")
            return

    # 2. Alle 12 Monate zu einem einzigen riesigen DataFrame zusammenfügen
    df_all_months = pd.concat(all_months_data)

    # 3. Den Durchschnitt berechnen (Die Pandas-Magie)
    # Wir gruppieren die Daten nach 'node_id' und 'threshold' und nehmen den Mittelwert (mean).
    # Das bedeutet: Jeder Knoten bekommt pro Threshold genau EINE Zeile mit den gemittelten Werten.
    df_yearly_avg = df_all_months.groupby(["node_id", "threshold"]).mean().reset_index()

    # (Optional) Die Thresholds filtern, falls in der CSV mehr drinnen sind, als übergeben wurden:
    # df_yearly_avg = df_yearly_avg[df_yearly_avg["threshold"].isin(thresholds)]

    # 4. Ordnerstruktur und Dateiname für das Jahresergebnis festlegen
    if def_option == "Raw":
        output_filepath = f"./processed-data/metrics_Raw/network_metrics_ssp{scenario}_decade{year}_Raw.csv"
    elif def_option == "BaU":
        output_filepath = f"./processed-data/metrics_BaU/network_metrics_ssp{scenario}_decade{year}_BaU.csv"
    elif def_option == "Gov":
        output_filepath = f"./processed-data/metrics_Gov/network_metrics_ssp{scenario}_decade{year}_Gov.csv"
    elif def_option == "Sec":
        output_filepath = f"./processed-data/metrics_Sec/network_metrics_ssp{scenario}_decade{year}_Sec.csv"

    # 5. Speichern
    # WICHTIG: Da wir mit 'groupby' gearbeitet haben, speichert Pandas den Index automatisch mit.
    # Wir nutzen index=False, damit keine überflüssige ID-Spalte entsteht.
    df_yearly_avg.to_csv(output_filepath, index=False)

    print(f"Jahresdurchschnitt für Jahr {year} ({def_option}) erfolgreich gespeichert!")


def calculate_yearly_network_metrics2(scenario: int, year: int, thresholds: list, def_option: str = "Raw"):
    """Berechnet Metriken für alle Thresholds und speichert sie in EINER Master-CSV."""

    if def_option == "Raw":
        nc_filepath = f"../2-pre-processing/pre-processed-data/water_Raw/scenario_ssp{scenario}_decade{year}_Raw.nc"
    elif def_option == "BaU":
        nc_filepath = f"../2-pre-processing/pre-processed-data/water_BaU/scenario_ssp{scenario}_decade{year}_BaU.nc"
    elif def_option == "Gov":
        nc_filepath = f"../2-pre-processing/pre-processed-data/water_Gov/scenario_ssp{scenario}_decade{year}_Gov.nc"
    elif def_option == "Sec":
        nc_filepath = f"../2-pre-processing/pre-processed-data/water_Sec/scenario_ssp{scenario}_decade{year}_Sec.nc"

    ds = xr.open_dataset(nc_filepath)
    matrix = ds["network"].values

    # NEU: Eine leere Liste, um die Teilergebnisse zu sammeln
    all_threshold_data = []

    for threshold in thresholds:
        print(f"Berechne Threshold {threshold}...")
        binary_matrix = (matrix >= threshold).astype(int)
        G = nx.from_numpy_array(binary_matrix, create_using=nx.DiGraph)

        # Metriken berechnen
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        clustering_coeffs = nx.clustering(G)

        # DataFrame für DIESEN Threshold erstellen
        df_metrics = pd.DataFrame({
            "in_degree": in_degrees,
            "out_degree": out_degrees,
            "clustering": clustering_coeffs
        })

        # Threshold-Spalte hinzufügen
        df_metrics["threshold"] = threshold

        # NEU: DataFrame in unsere Liste packen, NICHT speichern!
        all_threshold_data.append(df_metrics)

    # --- SCHLEIFE BEENDET ---

    # NEU: Alle DataFrames in der Liste zu einem riesigen DataFrame zusammenkleben
    master_df = pd.concat(all_threshold_data)

    # Ordnerstruktur nur bis zum Monat erstellen (wir brauchen keine Threshold-Ordner mehr)
    if def_option == "Raw":
        output_filepath = f"./processed-data/metrics_Raw/network_metrics_ssp{scenario}_decade{year}_Raw.csv"
    elif def_option == "BaU":
        output_filepath = f"./processed-data/metrics_BaU/network_metrics_ssp{scenario}_decade{year}_BaU.csv"
    elif def_option == "Gov":
        output_filepath = f"./processed-data/metrics_Gov/network_metrics_ssp{scenario}_decade{year}_Gov.csv"
    elif def_option == "Sec":
        output_filepath = f"./processed-data/metrics_Sec/network_metrics_ssp{scenario}_decade{year}_Sec.csv"

    master_df.to_csv(output_filepath, index_label="node_id")

    ds.close()
    print(f"Master-CSV für Jahr {year} erfolgreich gespeichert!")


def create_global_monthly_timeseries(scenario: int, years: list, def_option: str = "Raw"):
    """
    Liest die monatlichen Netzwerk-Metriken ein, berechnet den globalen Durchschnitt 
    (über alle 416 Knoten) und erstellt eine gesammelte monatliche Zeitreihen-Datei.
    """
    all_months_data = []

    for year in years:
        # NEU: Innere Schleife für die 12 Monate
        for month in range(1, 13):
            # 1. Dateipfad der monatlichen Datei bestimmen
            if def_option == "Raw":
                input_filepath = f"./processed-data/metrics_Raw/network_metrics_ssp{scenario}_decade{year}_month{month:02d}_Raw.csv"
            elif def_option == "BaU":
                input_filepath = f"./processed-data/metrics_BaU/network_metrics_ssp{scenario}_decade{year}_month{month:02d}_BaU.csv"
            elif def_option == "Gov":
                input_filepath = f"./processed-data/metrics_Gov/network_metrics_ssp{scenario}_decade{year}_month{month:02d}_Gov.csv"
            elif def_option == "Sec":
                input_filepath = f"./processed-data/metrics_Sec/network_metrics_ssp{scenario}_decade{year}_month{month:02d}_Sec.csv"

            try:
                # 2. Die monatliche Datei einlesen
                df_month = pd.read_csv(input_filepath)

                # 3. Den globalen Durchschnitt über alle Knoten berechnen
                df_global_avg = df_month.drop(columns=['node_id']).groupby('threshold').mean().reset_index()

                # 4. Jahr UND Monat als neue Spalten hinzufügen
                df_global_avg['year'] = year
                df_global_avg['month'] = month

                # Zur Liste hinzufügen
                all_months_data.append(df_global_avg)

            except FileNotFoundError:
                # Um die Konsole nicht zu überfluten, können wir die Warnung hier leiser stellen
                # print(f"Warnung: Datei {input_filepath} nicht gefunden. Überspringe...")
                continue

    # Wenn keine Daten gefunden wurden, brechen wir ab
    if not all_months_data:
        print(f"Keine monatlichen Daten für Szenario {scenario} ({def_option}) gefunden.")
        return

    # 5. Alle Monate zu einem DataFrame (der finalen Zeitreihe) zusammenfügen
    df_timeseries = pd.concat(all_months_data)

    # 6. Spalten sortieren: Jahr, Monat und Threshold ganz nach links für die Lesbarkeit
    cols = ['year', 'month', 'threshold'] + [col for col in df_timeseries.columns if col not in ['year', 'month', 'threshold']]
    df_timeseries = df_timeseries[cols]

    # 7. Datei im selben Timeseries-Ordner speichern, aber mit angepasstem Namen
    output_dir = Path(f"./processed-data/timeseries_{def_option}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # "_monthly_" im Dateinamen hinzugefügt, um es von der jährlichen Datei abzugrenzen
    output_filepath = output_dir / f"global_monthly_metrics_timeseries_ssp{scenario}_{def_option}.csv"

    df_timeseries.to_csv(output_filepath, index=False)
    print(f"Monatliche Zeitreihe erfolgreich erstellt: {output_filepath}")


def create_global_timeseries(scenario: int, years: list, def_option: str = "Raw"):
    """
    Liest die jährlichen Netzwerk-Metriken ein, berechnet den globalen Durchschnitt 
    (über alle 416 Knoten) und erstellt eine gesammelte Zeitreihen-Datei über alle Jahre.
    """
    all_years_data = []

    for year in years:
        # 1. Dateipfad der jährlichen Datei bestimmen
        if def_option == "Raw":
            input_filepath = f"./processed-data/metrics_Raw/network_metrics_ssp{scenario}_decade{year}_Raw.csv"
        elif def_option == "BaU":
            input_filepath = f"./processed-data/metrics_BaU/network_metrics_ssp{scenario}_decade{year}_BaU.csv"
        elif def_option == "Gov":
            input_filepath = f"./processed-data/metrics_Gov/network_metrics_ssp{scenario}_decade{year}_Gov.csv"
        elif def_option == "Sec":
            input_filepath = f"./processed-data/metrics_Sec/network_metrics_ssp{scenario}_decade{year}_Sec.csv"

        try:
            # 2. Die jährliche Datei einlesen
            df_year = pd.read_csv(input_filepath)

            # 3. Den globalen Durchschnitt über alle Knoten berechnen
            # Wir werfen die 'node_id' weg und gruppieren nur noch nach 'threshold'.
            df_global_avg = df_year.drop(columns=['node_id']).groupby('threshold').mean().reset_index()

            # 4. Das aktuelle Jahr als neue Spalte hinzufügen, damit wir später die Zeitreihe haben
            df_global_avg['year'] = year

            # Zur Liste hinzufügen
            all_years_data.append(df_global_avg)

        except FileNotFoundError:
            print(f"Warnung: Datei {input_filepath} nicht gefunden. Überspringe Jahr {year}.")
            continue

    # Wenn keine Daten gefunden wurden, brechen wir ab
    if not all_years_data:
        print(f"Keine Daten für Szenario {scenario} ({def_option}) gefunden.")
        return

    # 5. Alle Jahre zu einem DataFrame (der finalen Zeitreihe) zusammenfügen
    df_timeseries = pd.concat(all_years_data)

    # 6. Spalten für eine schöne Lesbarkeit sortieren (Jahr ganz nach links)
    cols = ['year', 'threshold'] + [col for col in df_timeseries.columns if col not in ['year', 'threshold']]
    df_timeseries = df_timeseries[cols]

    # 7. Neuen Ordner für Zeitreihen erstellen und Datei speichern
    output_dir = Path(f"./processed-data/timeseries_{def_option}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / f"global_metrics_timeseries_ssp{scenario}_{def_option}.csv"

    df_timeseries.to_csv(output_filepath, index=False)
    print(f"Globale Zeitreihe erfolgreich erstellt: {output_filepath}")


def execute_network_metrics():
    scenarios = [245, 370, 585]
    years_raw = range(2030, 2100)
    years_def = range(2030, 2051)
    months = range(1, 13)
    thresholds = [0.2367, 1.1514, 3.112]
    # thresholds = [1.1514, 3.112]
    # thresholds = [1.1514]

    for scenario in scenarios:
        for year in years_raw:
            for month in months:
                print(f"Verarbeite Szenario {scenario}, Jahr {year}, Monat {month} (Raw)")
                calculate_network_metrics(scenario, year, month, thresholds, def_option="Raw")

                print(f"Verarbeite Szenario {scenario}, Jahr {year}, Monat {month} (Sec)")
                calculate_network_metrics(scenario, year, month, thresholds, def_option="Sec")

        for year in years_def:
            for month in months:
                print(f"Verarbeite Szenario {scenario}, Jahr {year}, Monat {month} (BaU)")
                calculate_network_metrics(scenario, year, month, thresholds, def_option="BaU")

                print(f"Verarbeite Szenario {scenario}, Jahr {year}, Monat {month} (Gov)")
                calculate_network_metrics(scenario, year, month, thresholds, def_option="Gov")


def execute_yearly_network_metrics():
    scenarios = [245, 370, 585]
    years_raw = range(2030, 2100)
    years_def = range(2030, 2051)
    thresholds = [12*0.2367, 12*1.1514, 12*3.112]
    # thresholds = [12*1.1514]

    for scenario in scenarios:
        for year in years_raw:
            print(f"Verarbeite Szenario {scenario}, Jahr {year} (Raw)")
            calculate_yearly_network_metrics(scenario, year, thresholds, def_option="Raw")

            print(f"Verarbeite Szenario {scenario}, Jahr {year} (Sec)")
            calculate_yearly_network_metrics(scenario, year, thresholds, def_option="Sec")

        for year in years_def:
            print(f"Verarbeite Szenario {scenario}, Jahr {year} (BaU)")
            calculate_yearly_network_metrics(scenario, year, thresholds, def_option="BaU")

            print(f"Verarbeite Szenario {scenario}, Jahr {year} (Gov)")
            calculate_yearly_network_metrics(scenario, year, thresholds, def_option="Gov")


def execute_timeseries_creation():
    scenarios = [245, 370, 585]
    years_raw = range(2030, 2100)
    years_def = range(2030, 2051)

    for scenario in scenarios:
        # --- Jährliche Zeitreihen ---
        print(f"Erstelle jährliche Zeitreihen für Szenario {scenario}...")
        create_global_timeseries(scenario, list(years_raw), def_option="Raw")
        create_global_timeseries(scenario, list(years_def), def_option="BaU")
        create_global_timeseries(scenario, list(years_raw), def_option="Sec")
        create_global_timeseries(scenario, list(years_def), def_option="Gov")

        # --- Monatliche Zeitreihen ---
        print(f"Erstelle monatliche Zeitreihen für Szenario {scenario}...")
        create_global_monthly_timeseries(scenario, list(years_raw), def_option="Raw")
        create_global_monthly_timeseries(scenario, list(years_def), def_option="BaU")
        create_global_monthly_timeseries(scenario, list(years_raw), def_option="Sec")
        create_global_monthly_timeseries(scenario, list(years_def), def_option="Gov")


if __name__ == "__main__":
    execute_network_metrics()
    execute_yearly_network_metrics()
    execute_timeseries_creation()
    a = 3
