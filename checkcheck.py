import os
import xarray as xr
import numpy as np

def compare_network_scenarios(scenario: int = 245, year: int = 2030):
    """
    Vergleicht die 'network'-Matrix-Summen zwischen Raw, Sec und BaU
    für ein gesamtes Jahr, um zu prüfen, ob die Entwaldungseffekte logisch sind.
    """
    print(f"Starte Netzwerk-Vergleich für SSP{scenario}, Jahr {year}...\n")
    
    # Ordner an deine neue Struktur angepasst
    dirs = {
        "Raw": "1-data/water",
        "Sec": "1-data/water_secondary_vegetation",
        "BaU": "1-data/deforestation" 
    }
    
    # Speicher für die Jahressummen
    total_flow = {"Raw": 0.0, "Sec": 0.0, "BaU": 0.0}
    valid_months = {"Raw": 0, "Sec": 0, "BaU": 0}
    
    for month in range(1, 13):
        for key, folder in dirs.items():
            # Dateiname zusammenbauen (Anpassen falls BaU z.B. "_BaU.nc" am Ende hat!)
            filename = f"scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
            filepath = os.path.join(folder, filename)
            
            # Falls die BaU Dateien einen Suffix haben, probieren wir das als Fallback
            if key == "BaU" and not os.path.exists(filepath):
                filename = f"scenario_ssp{scenario}_decade{year}_month{month:02d}_BaU.nc"
                filepath = os.path.join(folder, filename)
                
            if os.path.exists(filepath):
                try:
                    with xr.open_dataset(filepath) as ds:
                        if "network" in ds:
                            # Summiere alle Flüsse im gesamten Netzwerk auf (ignoriere NaNs)
                            matrix_sum = np.nansum(ds["network"].values)
                            total_flow[key] += float(matrix_sum)
                            valid_months[key] += 1
                        else:
                            print(f"Warnung: 'network' fehlt in {filepath}")
                except Exception as e:
                    print(f"Fehler beim Laden von {filepath}: {e}")
            else:
                if month == 1: # Nur beim ersten Monat warnen, sonst spamt es die Konsole voll
                    print(f"Warnung: Datei nicht gefunden -> {filepath}")

    # =================================================================
    # AUSWERTUNG & FAZIT
    # =================================================================
    print("="*60)
    print(f" GESAMTER FEUCHTIGKEITSFLUSS IM NETZWERK (Summe über Jahr {year})")
    print("="*60)
    
    for key in ["Raw", "Sec", "BaU"]:
        if valid_months[key] > 0:
            print(f"{key: <5} ({valid_months[key]:>2} Monate): {total_flow[key]:>15,.2f} Einheiten")
        else:
            print(f"{key: <5} ( 0 Monate): Keine Daten gefunden!")
            
    print("-" * 60)
    
    # Logik-Check nur durchführen, wenn wir für alle drei Szenarien Daten haben
    if all(valid_months[k] == 12 for k in ["Raw", "Sec", "BaU"]):
        print("LOGIK-CHECK:")
        
        # Erwartung: Raw > Sec > BaU
        if total_flow["Sec"] < total_flow["BaU"]:
            print("⚠️ WARNUNG: Sec ist NIEDRIGER als BaU!")
            print("Das ist unlogisch! Business as Usual (BaU) sollte die stärkste Entwaldung")
            print("und damit die geringsten Flüsse haben. Möglicherweise wurden die Datensätze")
            print("Sec und BaU bei der Datenbereitstellung oder im Code vertauscht!")
        elif total_flow["Sec"] > total_flow["BaU"]:
            print("✅ Alles in Ordnung: Sec ist GRÖSSER als BaU.")
            print("Die Erholung der Sekundärvegetation stützt das Netzwerk erwartungsgemäß besser als BaU.")
            
        if total_flow["Raw"] < total_flow["Sec"] or total_flow["Raw"] < total_flow["BaU"]:
            print("⚠️ WARNUNG: Raw ist niedriger als ein Entwaldungsszenario!")
            print("Das ergibt keinen Sinn. Der intakte Wald sollte den höchsten Feuchtigkeitsfluss haben.")
    else:
        print("Logik-Check übersprungen, da nicht für alle Szenarien 12 Monate gefunden wurden.")
        print("Bitte überprüfe die Ordner- und Dateinamen im Skript.")

if __name__ == "__main__":
    compare_network_scenarios(scenario=245, year=2030)