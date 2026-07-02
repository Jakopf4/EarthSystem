import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_ffl_multiple_scenarios(scenarios: list, target_threshold: float):
    """
    Erstellt ein Liniendiagramm für die FFLs über die Zeit und vergleicht 
    mehrere SSP-Szenarien sowie die Modelle 'Raw' und 'BaU'.
    """
    all_data = []

    # 1. Daten für alle übergebenen Szenarien sammeln
    for scenario in scenarios:
        path_raw = Path(f"../processing/processed-data/timeseries_Raw/global_metrics_timeseries_ssp{scenario}_Raw.csv")
        path_bau = Path(f"../processing/processed-data/timeseries_BaU/global_metrics_timeseries_ssp{scenario}_BaU.csv")

        # Raw-Daten laden und markieren
        if path_raw.exists():
            df_raw = pd.read_csv(path_raw)
            df_raw['Deforestation_Model'] = 'Raw (Keine Entw.)'
            df_raw['Scenario'] = f"SSP {scenario}"
            all_data.append(df_raw)
        else:
            print(f"Warnung: Raw-Daten für SSP {scenario} fehlen.")

        # BaU-Daten laden und markieren
        if path_bau.exists():
            df_bau = pd.read_csv(path_bau)
            df_bau['Deforestation_Model'] = 'BaU (Business as Usual)'
            df_bau['Scenario'] = f"SSP {scenario}"
            all_data.append(df_bau)
        else:
            print(f"Warnung: BaU-Daten für SSP {scenario} fehlen.")

    # Prüfen, ob überhaupt Daten gefunden wurden
    if not all_data:
        print("Fehler: Keine Dateien gefunden. Plotvorgang abgebrochen.")
        return

    # 2. Alle Daten zu einem großen DataFrame zusammenfügen
    df_combined = pd.concat(all_data)

    # 3. Nach dem gewünschten Threshold filtern
    df_plot = df_combined[df_combined['threshold'] == target_threshold]

    if df_plot.empty:
        print(f"Fehler: Keine Daten für den Threshold {target_threshold} gefunden.")
        return

    # Sicherheits-Check für die FFL-Spalte
    if 'ffl' not in df_plot.columns:
        print("Fehler: Die Spalte 'ffl' existiert nicht in den Daten.")
        return

    # --- AB HIER WIRD GEPLOTTET ---

    # Diagramm-Größe etwas breiter machen, da wir mehr Informationen haben
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")

    # Der Multi-Lineplot
    # hue='Scenario' macht 3 Farben, style='Deforestation_Model' macht 2 Linienarten
    sns.lineplot(
        data=df_plot, 
        x='year', 
        y='ffl', 
        hue='Scenario', 
        style='Deforestation_Model', 
        linewidth=2.5,
        markers=False # Ohne Punkte, da 6 Linien mit Punkten zu unruhig wirken
    )

    # Diagramm aufhübschen
    plt.title(f"Entwicklung der Feed-Forward-Loops (Threshold: {target_threshold})", fontsize=16, pad=15)
    plt.xlabel("Jahr", fontsize=12)
    plt.ylabel("Durchschnittliche Anzahl FFLs pro Knoten", fontsize=12)

    # Legende nach rechts außen verschieben, damit sie die Linien nicht verdeckt
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # x-Achse auf den Bereich 2030 bis 2100 limitieren
    plt.xlim(2030, 2100)

    # Das Layout anpassen, damit die Legende nicht abgeschnitten wird
    plt.tight_layout()

    plt.show()


def plot_ffl_multiple_thresholds(scenario: int, thresholds: list):
    """
    Erstellt ein Liniendiagramm für die FFLs, das verschiedene Thresholds 
    für ein spezifisches SSP-Szenario vergleicht (Raw vs. BaU).
    """
    # 1. Dateipfade für das gewählte Szenario
    path_raw = Path(f"../processing/processed-data/timeseries_Raw/global_metrics_timeseries_ssp{scenario}_Raw.csv")
    path_bau = Path(f"../processing/processed-data/timeseries_BaU/global_metrics_timeseries_ssp{scenario}_BaU.csv")

    all_data = []

    # Raw-Daten laden
    if path_raw.exists():
        df_raw = pd.read_csv(path_raw)
        df_raw['Deforestation_Model'] = 'Raw (Keine Entw.)'
        all_data.append(df_raw)
    else:
        print(f"Warnung: Raw-Daten für SSP {scenario} fehlen.")

    # BaU-Daten laden
    if path_bau.exists():
        df_bau = pd.read_csv(path_bau)
        df_bau['Deforestation_Model'] = 'BaU (Business as Usual)'
        all_data.append(df_bau)
    else:
        print(f"Warnung: BaU-Daten für SSP {scenario} fehlen.")

    if not all_data:
        print("Fehler: Keine Dateien gefunden. Plotvorgang abgebrochen.")
        return

    # 2. Daten zusammenfügen
    df_combined = pd.concat(all_data)

    # 3. Nur die übergebenen Thresholds herausfiltern
    df_plot = df_combined[df_combined['threshold'].isin(thresholds)].copy()

    if df_plot.empty:
        print(f"Fehler: Keine Daten für die Thresholds {thresholds} gefunden.")
        return

    if 'ffl' not in df_plot.columns:
        print("Fehler: Die Spalte 'ffl' existiert nicht in den Daten.")
        return

    # 4. Threshold als Text (Kategorie) formatieren, damit Seaborn schöne, 
    # voneinander abgesetzte Farben (und keinen Farbverlauf) wählt.
    df_plot['Threshold_Label'] = "Threshold: " + df_plot['threshold'].astype(str)

    # --- AB HIER WIRD GEPLOTTET ---

    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")

    # Der Plot: hue trennt die Thresholds farblich, style trennt Raw/BaU durch Strichelung
    sns.lineplot(
        data=df_plot, 
        x='year', 
        y='clustering', 
        hue='Threshold_Label', 
        style='Deforestation_Model', 
        linewidth=2.5,
        markers=False
    )

    # Diagramm aufhübschen
    plt.title(f"Sensitivitätsanalyse: Einfluss des Thresholds auf FFLs (SSP {scenario})", fontsize=16, pad=15)
    plt.xlabel("Jahr", fontsize=12)
    plt.ylabel("Durchschnittliche Anzahl FFLs pro Knoten", fontsize=12)

    # Legende nach rechts außen verschieben
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # x-Achse limitieren
    plt.xlim(2030, 2100)

    # Layout anpassen
    plt.tight_layout()

    # plt.savefig(f"plot_ffl_sensitivity_ssp{scenario}.png", dpi=300)
    plt.show()


def plot_ffl_relative_change_all_scenarios(scenarios: list, target_threshold: float):
    """
    Berechnet die prozentuale Veränderung der FFLs relativ zum Jahr 2030 
    und plottet 'Raw' vs 'BaU' für mehrere SSP-Szenarien gleichzeitig.
    """
    all_data = []

    for scenario in scenarios:
        path_raw = Path(f"../processing/processed-data/timeseries_Raw/global_metrics_timeseries_ssp{scenario}_Raw.csv")
        path_bau = Path(f"../processing/processed-data/timeseries_BaU/global_metrics_timeseries_ssp{scenario}_BaU.csv")

        # --- Raw-Daten verarbeiten ---
        if path_raw.exists():
            df_raw = pd.read_csv(path_raw)
            df_raw = df_raw[df_raw['threshold'] == target_threshold].copy()

            # Prüfen, ob wir einen Nullpunkt für 2030 haben
            if not df_raw.empty and 2030 in df_raw['year'].values:
                baseline_raw = df_raw[df_raw['year'] == 2030]['ffl'].values[0]
                df_raw['ffl_change_pct'] = ((df_raw['ffl'] - baseline_raw) / baseline_raw) * 100
                df_raw['Modell'] = 'Raw (Keine Entw.)'
                df_raw['Scenario'] = f"SSP {scenario}"
                all_data.append(df_raw)
            else:
                print(f"Warnung: Keine 2030-Baseline für Raw SSP {scenario} gefunden.")
        else:
            print(f"Warnung: Datei fehlt -> {path_raw}")

        # --- BaU-Daten verarbeiten ---
        if path_bau.exists():
            df_bau = pd.read_csv(path_bau)
            df_bau = df_bau[df_bau['threshold'] == target_threshold].copy()
            
            # Prüfen, ob wir einen Nullpunkt für 2030 haben
            if not df_bau.empty and 2030 in df_bau['year'].values:
                baseline_bau = df_bau[df_bau['year'] == 2030]['ffl'].values[0]
                df_bau['ffl_change_pct'] = ((df_bau['ffl'] - baseline_bau) / baseline_bau) * 100
                df_bau['Modell'] = 'BaU (Mit Entwaldung)'
                df_bau['Scenario'] = f"SSP {scenario}"
                all_data.append(df_bau)
            else:
                print(f"Warnung: Keine 2030-Baseline für BaU SSP {scenario} gefunden.")
        else:
            print(f"Warnung: Datei fehlt -> {path_bau}")

    # Wenn die Liste leer ist, brechen wir ab
    if not all_data:
        print("Fehler: Keine verarbeitbaren Daten gefunden. Plot abgebrochen.")
        return

    # Alle vorbereiteten Tabellen zu einem großen DataFrame verknüpfen
    df_plot = pd.concat(all_data)

    # --- AB HIER WIRD GEPLOTTET ---

    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")

    # Der Multi-Lineplot
    # hue='Scenario' macht die Farben, style='Modell' macht die Strichelung
    sns.lineplot(
        data=df_plot, 
        x='year', 
        y='ffl_change_pct', 
        hue='Scenario', 
        style='Modell', 
        linewidth=2.5,
        markers=False  # Keine Punkte, damit das Diagramm sauber bleibt
    )

    # Die Nulllinie (Startpunkt 2030 für alle Kurven)
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=0)

    # Diagramm aufhübschen
    plt.title(f"Relative Entwicklung der FFLs seit 2030: Klimawandel vs. Entwaldung (Threshold {target_threshold})", fontsize=15, pad=15)
    plt.xlabel("Jahr", fontsize=12)
    plt.ylabel("Veränderung der FFLs (%)", fontsize=12)

    # Legende nach rechts außen verschieben, um Überschneidungen zu vermeiden
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # x-Achse auf den Zeitraum 2030 bis 2050 limitieren
    plt.xlim(2030, 2050) 

    # Layout anpassen
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_ffl_multiple_scenarios(scenarios=[245, 370, 585], target_threshold=1.14)
    plot_ffl_multiple_thresholds(scenario=370, thresholds=[1.0, 1.14, 1.5])
    plot_ffl_relative_change_all_scenarios(scenarios=[245, 370, 585], target_threshold=1.14)