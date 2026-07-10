import os
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path


def process_historical_data():
    """
    Liest die historischen Niederschlags- und Verdunstungsdaten (1950-2014) aus 
    und wandelt sie in das standardisierte CSV-Format der Pipeline um.
    """
    historical_file = "../1-data/historical/e_p_moisture_network_amazon_historical.nc"

    # Neuer Ordner für die historischen Zeitenreihen
    out_dir = Path("../3-processing/processed-data/timeseries_historical")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(historical_file):
        print(f"FEHLER: Historische Datei nicht gefunden: {historical_file}")
        return

    print(f"Lade historische Daten aus {historical_file}...")

    yearly_records = []
    monthly_records = []

    with xr.open_dataset(historical_file) as ds:
        for year in range(1950, 2015):
            try:
                # --- 1. JÄHRLICHE DATEN ---
                # Wir berechnen den globalen räumlichen Durchschnitt (nanmean) über das Grid.
                ann_prec_sum = np.nanmean(ds[f"annual_prec_amazon_{year}"].values)
                ann_evap_sum = np.nanmean(ds[f"annual_evap_amazon_{year}"].values)

                # Da der Bug behoben ist, übernehmen wir die echten Jahressummen direkt.
                yearly_records.append({
                    "year": year,
                    "threshold": "None", # Kein Netzwerk vorhanden
                    "prec": ann_prec_sum,
                    "evap": ann_evap_sum,
                    "in_degree": np.nan,
                    "out_degree": np.nan,
                    "clustering": np.nan,
                    "ffl": np.nan
                })

                # --- 2. MONATLICHE DATEN ---
                mon_prec_array = ds[f"monthly_prec_amazon_{year}"].values
                mon_evap_array = ds[f"monthly_evap_amazon_{year}"].values

                for m in range(12):
                    monthly_records.append({
                        "year": year,
                        "month": m + 1,
                        "threshold": "None",
                        "prec": np.nanmean(mon_prec_array[m]),
                        "evap": np.nanmean(mon_evap_array[m]),
                        "in_degree": np.nan,
                        "out_degree": np.nan,
                        "clustering": np.nan,
                        "ffl": np.nan
                    })

            except KeyError as e:
                print(f"Warnung: Fehlende Variablen für das Jahr {year} ({e}).")
            except Exception as e:
                print(f"Unerwarteter Fehler bei Jahr {year}: {e}")

    # =================================================================
    # DATAFRAMES ERSTELLEN UND SPEICHERN
    # =================================================================
    if yearly_records:
        df_yearly = pd.DataFrame(yearly_records)
        df_monthly = pd.DataFrame(monthly_records)

        yearly_path = out_dir / "global_metrics_timeseries_historical.csv"
        monthly_path = out_dir / "global_monthly_metrics_timeseries_historical.csv"

        df_yearly.to_csv(yearly_path, index=False)
        df_monthly.to_csv(monthly_path, index=False)

        print("\nErfolgreich abgeschlossen!")
        print(f"-> Jahres-CSV erstellt: {yearly_path}")
        print(f"-> Monats-CSV erstellt: {monthly_path}")
    else:
        print("Es konnten keine historischen Daten verarbeitet werden.")


def process_historical_spatial_average():
    """
    Berechnet den gesamten historischen Durchschnitt (1950-2014) UND die
    Standardabweichungen (sowohl auf Basis der Jahressummen als auch unter
    Berücksichtigung aller monatlichen Schwankungen) für jede der 416 Gitterzellen.
    """
    historical_file = "../1-data/historical/e_p_moisture_network_amazon_historical.nc"
    out_dir = Path("../3-processing/processed-data/timeseries_historical")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(historical_file):
        print(f"FEHLER: Historische Datei nicht gefunden: {historical_file}")
        return

    print("\nBerechne räumliche Statistiken (inkl. monatlicher Schwankungen 1950-2014) pro Zelle...")

    years = range(1950, 2015)

    with xr.open_dataset(historical_file) as ds:
        try:
            # Koordinaten auslesen
            lats = ds["latitudes"].values
            lons = ds["longitudes"].values
        except KeyError:
            print("FEHLER: Konnte 'latitudes' oder 'longitudes' nicht in der Datei finden.")
            return

        # Listen für jährliche und monatliche Daten-Arrays vorbereiten
        all_prec_years = []
        all_evap_years = []
        all_prec_months = []
        all_evap_months = []

        for year in years:
            try:
                # 1. Jährliche Daten sammeln (Länge 416)
                all_prec_years.append(ds[f"annual_prec_amazon_{year}"].values)
                all_evap_years.append(ds[f"annual_evap_amazon_{year}"].values)
                
                # 2. Monatliche Daten sammeln (Shape jeweils 12, 416)
                all_prec_months.append(ds[f"monthly_prec_amazon_{year}"].values)
                all_evap_months.append(ds[f"monthly_evap_amazon_{year}"].values)
            except KeyError as e:
                print(f"Warnung: Daten für Jahr {year} fehlen ({e}).")

        # In numpy-Matrizen umwandeln
        # Jährliche Matrizen haben Shape (65, 416)
        prec_annual_matrix = np.array(all_prec_years)
        evap_annual_matrix = np.array(all_evap_years)
        
        # Monatliche Daten stapeln zu Shape (65 * 12, 416) -> (780, 416)
        prec_monthly_matrix = np.vstack(all_prec_months)
        evap_monthly_matrix = np.vstack(all_evap_months)

        # --- STATISTIKEN BERECHNEN ---
        # 1. Jährliche Mittelwerte
        prec_mean = np.mean(prec_annual_matrix, axis=0)
        evap_mean = np.mean(evap_annual_matrix, axis=0)

        # 2. Jährliche Standardabweichung (Schwankung von Jahr zu Jahr)
        prec_annual_std = np.std(prec_annual_matrix, axis=0)
        evap_annual_std = np.std(evap_annual_matrix, axis=0)

        # 3. NEU: Monatliche Standardabweichung (Schwankung inklusive Saisonalität Trocken-/Regenzeit)
        prec_monthly_std = np.std(prec_monthly_matrix, axis=0)
        evap_monthly_std = np.std(evap_monthly_matrix, axis=0)

        # DataFrame für jede einzelne Zelle erstellen mit allen Statistiken
        df_spatial = pd.DataFrame({
            "lat": lats,
            "lon": lons,
            "mean_annual_prec": prec_mean,
            "std_annual_prec": prec_annual_std,
            "std_monthly_prec": prec_monthly_std,   # <--- NEU!
            "mean_annual_evap": evap_mean,
            "std_annual_evap": evap_annual_std,
            "std_monthly_evap": evap_monthly_std    # <--- NEU!
        })

        out_path = out_dir / "historical_spatial_average_per_cell.csv"
        df_spatial.to_csv(out_path, index=False)

        print(f"-> Räumliche Statistik-CSV (jährlich & monatlich) erfolgreich erstellt: {out_path}")


if __name__ == "__main__":
    process_historical_data()
    process_historical_spatial_average()