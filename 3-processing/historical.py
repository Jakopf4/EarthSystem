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
                # Keine Teilung durch 12 mehr nötig!
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
                # Shape ist vermutlich (12, lat, lon) oder (12, 416)
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
    Berechnet den gesamten historischen Durchschnitt (1950-2014) 
    für jede der 416 Gitterzellen und speichert das Ergebnis 
    inklusive Koordinaten (Lat/Lon) als CSV.
    """
    historical_file = "../1-data/historical/e_p_moisture_network_amazon_historical.nc"
    out_dir = Path("../3-processing/processed-data/timeseries_historical")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(historical_file):
        print(f"FEHLER: Historische Datei nicht gefunden: {historical_file}")
        return
        
    print("\nBerechne räumlichen historischen Durchschnitt (1950-2014) pro Zelle...")
    
    years = range(1950, 2015)
    num_years = len(years)
    
    with xr.open_dataset(historical_file) as ds:
        try:
            # Koordinaten auslesen
            lats = ds["latitudes"].values
            lons = ds["longitudes"].values
        except KeyError:
            print("FEHLER: Konnte 'latitudes' oder 'longitudes' nicht in der Datei finden.")
            return
            
        num_cells = len(lats)
        prec_sum = np.zeros(num_cells)
        evap_sum = np.zeros(num_cells)
        
        for year in years:
            try:
                prec_sum += ds[f"annual_prec_amazon_{year}"].values
                evap_sum += ds[f"annual_evap_amazon_{year}"].values
            except KeyError as e:
                print(f"Warnung: Daten für Jahr {year} fehlen ({e}).")
                
        # Durchschnitt berechnen
        prec_mean = prec_sum / num_years
        evap_mean = evap_sum / num_years
        
        # DataFrame für jede einzelne Zelle erstellen
        df_spatial = pd.DataFrame({
            "lat": lats,
            "lon": lons,
            "mean_annual_prec": prec_mean,
            "mean_annual_evap": evap_mean
        })
        
        out_path = out_dir / "historical_spatial_average_per_cell.csv"
        df_spatial.to_csv(out_path, index=False)
        
        print(f"-> Räumliche Durchschnitts-CSV erfolgreich erstellt: {out_path}")

if __name__ == "__main__":
    process_historical_data()
    process_historical_spatial_average()